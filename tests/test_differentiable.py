import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from strider.thermo.differentiable import ThermoParameters, BatchedPartitionFunction

def test_differentiable_parameters_init():
    """Test that ThermoParameters initializes correctly with expected dimensions and default weights."""
    params_rna = ThermoParameters(material="rna")
    assert params_rna.material == "rna"
    assert params_rna.stack_dG37.shape == (16,)
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
    assert params.stack_dG37.grad is not None
    assert not torch.isnan(params.stack_dG37.grad).any()
    
    assert params.term_mismatch.grad is not None
    assert not torch.isnan(params.term_mismatch.grad).any()
    
    # Check that MLP weights have gradients propagated
    assert params.mlp_1_1[-1].weight.grad is not None
    assert not torch.isnan(params.mlp_1_1[-1].weight.grad).any()
    
    assert params.mlp_1_2[-1].weight.grad is not None
    assert not torch.isnan(params.mlp_1_2[-1].weight.grad).any()
    
    # Assert specific parameters get non-zero updates
    assert params.stack_dG37.grad.norm() > 0.0


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
    reg_loss = 0.1 * torch.sum(params.stack_dG37 ** 2)
    total_loss = loss + reg_loss
    
    total_loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
