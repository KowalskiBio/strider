import random

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from strider.thermo.differentiable import (
    ThermoParameters,
    BatchedPartitionFunction,
    batched_free_energy,
)
from strider import ThermoEngine

def test_differentiable_parameters_init():
    """Test that ThermoParameters initializes correctly with expected dimensions and default weights."""
    params_rna = ThermoParameters(material="rna")
    assert params_rna.material == "rna"
    # Full 4-base stack table (36 populated entries in a 256-slot tensor),
    # replacing the old 16-entry Watson-Crick dinucleotide table.
    assert params_rna.stack_table.shape == (256,)
    assert params_rna.term_mismatch.shape == (256,)
    assert params_rna.dangle_3.shape == (64,)
    assert params_rna.dangle_5.shape == (64,)
    
    # Test MLP initialization and that the final layer has zero weights/biases
    assert params_rna.mlp_1_1[-1].weight.norm() == 0.0
    assert params_rna.mlp_1_1[-1].bias.norm() == 0.0
    
    params_dna = ThermoParameters(material="dna")
    assert params_dna.material == "dna"


def test_differentiable_forward_pass():
    """Test the forward pass of BatchedPartitionFunction with a batch of sequences."""
    params = ThermoParameters(material="rna")
    model = BatchedPartitionFunction(params)
    
    sequences = ["GGGCUUCGGCCC", "AUGCAUGC", "GCGCGC"]
    
    # Try forward pass
    energies = model(sequences)
    
    assert energies.shape == (3,)
    assert energies.dtype == torch.float64
    assert not torch.isnan(energies).any()
    assert not torch.isinf(energies).any()
    
    # Verify that longer sequences generally have more negative free energies (stronger folding/stacking)
    # GC-rich hairpins should bind strongly
    assert energies[0] < 0.0


def test_differentiable_backward_gradients():
    """Test that gradients propagate successfully through all biophysical parameters and neural networks."""
    params = ThermoParameters(material="rna")
    model = BatchedPartitionFunction(params)
    
    sequences = ["GGGCUUCGGCCC", "AUGCAUGC", "GCGCGC"]
    energies = model(sequences)
    
    loss = torch.sum(energies)
    loss.backward()
    
    # Check that basic parameters have valid gradients
    assert params.stack_table.grad is not None
    assert not torch.isnan(params.stack_table.grad).any()
    
    assert params.term_mismatch.grad is not None
    assert not torch.isnan(params.term_mismatch.grad).any()
    
    # Check that MLP weights have gradients propagated
    assert params.mlp_1_1[-1].weight.grad is not None
    assert not torch.isnan(params.mlp_1_1[-1].weight.grad).any()
    
    assert params.mlp_1_2[-1].weight.grad is not None
    assert not torch.isnan(params.mlp_1_2[-1].weight.grad).any()
    
    # Assert specific parameters get non-zero updates
    assert params.stack_table.grad.norm() > 0.0


def test_toy_training_step():
    """Test a single epoch optimization step on a toy dataset to verify learning works smoothly."""
    params = ThermoParameters(material="rna")
    model = BatchedPartitionFunction(params)
    
    # Set up toy dataset
    sequences = ["GGGCUUCGGCCC", "AUGCAUGC", "GCGCGC"]
    true_energies = torch.tensor([-5.2, -0.05, -2.1], dtype=torch.float64)
    
    # Setup dual-rate optimizer: aggressive baseline physical vs conservative neural fine-tuning
    thermo_params = [v for k, v in model.named_parameters() if 'mlp' not in k]
    neural_params = [v for k, v in model.named_parameters() if 'mlp' in k]
    
    optimizer = optim.Adam([
        {'params': thermo_params, 'lr': 1e-2, 'weight_decay': 1e-5},
        {'params': neural_params, 'lr': 1e-3, 'weight_decay': 1e-4}
    ])
    
    criterion = nn.MSELoss()
    
    # Single optimization step
    optimizer.zero_grad()
    pred_energies = model(sequences)
    loss = criterion(pred_energies, true_energies)
    
    # Gentle physical regularizer to anchor baseline parameters
    reg_loss = 0.1 * torch.sum(params.stack_table ** 2)
    total_loss = loss + reg_loss
    
    total_loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def _random_seqs(alphabet, lengths, reps, seed):
    rng = random.Random(seed)
    return [
        "".join(rng.choice(alphabet) for _ in range(n))
        for n in lengths for _ in range(reps)
    ]


@pytest.mark.parametrize("material,alphabet", [("rna", "ACGU"), ("dna", "ACGT")])
def test_matches_native_within_tolerance(material, alphabet):
    """
    The differentiable engine must agree with the authoritative native engine.
    This is the alignment gate: wiring the full 36-entry stack table + the
    single-base-bulge stack-across term brought the residual on random sequences
    from ~0.7 (max ~4.5) kcal/mol down to ~0.3 (max ~1.0).
    """
    eng = ThermoEngine(material=material, celsius=37.0)
    seqs = _random_seqs(alphabet, (8, 12, 16, 20, 24, 30), reps=3, seed=0)
    diff = batched_free_energy(seqs, material=material).tolist()
    native = [eng.pfunc(s).free_energy for s in seqs]
    errs = [abs(d - n) for d, n in zip(diff, native)]
    mean_err = sum(errs) / len(errs)
    assert mean_err < 0.5, f"{material} mean |err|={mean_err:.3f}"
    assert max(errs) < 1.5, f"{material} max |err|={max(errs):.3f}"


def test_gu_wobble_helix_not_overstabilized():
    """
    Regression for the core bug: GU-wobble helices were scored with Watson-Crick
    dinucleotide stack energies (the old 16-entry table), over-stabilizing by
    several kcal/mol.  This sequence was ~4.5 kcal/mol too negative before the fix.
    """
    eng = ThermoEngine(material="rna", celsius=37.0)
    seq = "CAGUGGUAAUUGGAUAUUAG"
    diff = batched_free_energy([seq], material="rna").item()
    native = eng.pfunc(seq).free_energy
    assert abs(diff - native) < 1.0


def test_full_stack_table_has_wobble_entries():
    """The stack table is the full 36-entry NN table, not the 16 WC dinucleotides."""
    p = ThermoParameters(material="rna")
    assert p.stack_table.shape == (256,)
    # GU/UG-containing stacks must differ from the default fill (i.e. populated).
    default = -2.0
    populated = (p.stack_table != default).sum().item()
    assert populated >= 30  # 36 entries, minus any that happen to equal -2.0


def test_batched_free_energy_helper_matches_model():
    seqs = ["GGGCUUCGGCCC", "AUGCAUGC", "GCGCGC"]
    helper = batched_free_energy(seqs, material="rna").tolist()
    model = BatchedPartitionFunction(ThermoParameters(material="rna"))
    with torch.no_grad():
        direct = model(seqs).tolist()
    assert helper == pytest.approx(direct, abs=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    seqs = ["GGGCUUCGGCCCAAA", "AUGCAUGCUAGC", "GCGCGCAU"]
    cpu = batched_free_energy(seqs, material="rna", device="cpu").tolist()
    cuda = batched_free_energy(seqs, material="rna", device="cuda").cpu().tolist()
    assert cuda == pytest.approx(cpu, abs=1e-6)
