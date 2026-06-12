"""
Example 11: Gradient-based sequence design (differentiable inverse folding)

Covers:
- The autodiff base-pair-probability matrix (∂F/∂ε_ij = P(i,j))
- DiffObjective: differentiable ensemble defect, ΔG target, GC band,
  toehold accessibility, forbidden motifs
- DifferentiableDesigner: Adam on simplex logits with a temperature schedule,
  then a simulated-annealing polish hand-off
- A multi-strand (duplex) differentiable design

Unlike the simulated-annealing designer in example 04, here the sequence itself
is a continuous variable: every position is a distribution over {A,C,G,U}, the
whole strand is folded by a differentiable McCaskill engine, and the design loss
is minimized by gradient descent.  The gradient solution is then rounded and
handed to the SA designer for a short discrete polish.

Run:  python examples/11_gradient_design.py
"""

import torch

from strider import ThermoEngine
from strider.thermo.diff_design import DiffObjective
from strider.design.objective import DesignObjective
from strider.design.optimizer import DomainSpec
from strider.design.diff_designer import DifferentiableDesigner

torch.manual_seed(0)
engine = ThermoEngine(material="rna", celsius=37.0)


def banner(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


# ─── 1. Inverse-fold a hairpin by gradient descent + SA polish ──────────────

banner("1. Hairpin inverse folding")

target = "(((((((....)))))))"          # 18 nt, 7-bp stem, 4-nt loop
n = len(target)

designer = DifferentiableDesigner(material="rna", engine=engine, seed=0)

# A composite differentiable objective: match the target structure, keep GC in a
# band, and avoid homopolymer runs that make synthesis / specificity hard.
objective = (
    DiffObjective.ensemble_defect(target)
    + 0.2 * DiffObjective.gc_band(0.4, 0.6)
    + 0.1 * DiffObjective.forbidden_motifs(["GGGG", "CCCC", "AAAA", "UUUU"])
)

# Discrete objective for the SA polish hand-off (same target).
sa_objective = DesignObjective.ensemble_defect(engine, "hp", target)

result = designer.design(
    {"hp": DomainSpec(length=n, material="rna")},
    objective,
    n_restarts=16,
    n_steps=250,
    lr=0.2,
    sa_polish=True,
    sa_objective=sa_objective,
    sa_iterations=200,
    verbose=True,
)

seq = result.sequences["hp"]
mfe = engine.mfe(seq)
print(f"\n  target   : {target}")
print(f"  designed : {seq}")
print(f"  MFE fold : {getattr(mfe, 'structure', mfe)}")
print(f"  native ensemble defect : {engine.ensemble_defect(seq, target, normalize=True):.4f}")
print(f"  objective breakdown    : "
      f"{{{', '.join(f'{k}={v:.3g}' for k, v in result.objective_breakdown.items())}}}")


# ─── 2. Hit a target hairpin stability (ΔG) ─────────────────────────────────

banner("2. Targeting a free energy")

target_dg = -8.0
obj_dg = (DiffObjective.free_energy_target(target_dg)
          + 0.2 * DiffObjective.gc_band(0.4, 0.6))
res_dg = designer.design(
    {"hp": DomainSpec(length=16, material="rna")},
    obj_dg, n_restarts=12, n_steps=180, lr=0.2, sa_polish=False,
)
seq_dg = res_dg.sequences["hp"]
print(f"  target ΔG_ens : {target_dg:+.2f} kcal/mol")
print(f"  designed      : {seq_dg}")
print(f"  native ΔG_ens : {engine.pfunc(seq_dg).free_energy:+.2f} kcal/mol")


# ─── 3. Multi-strand: design a duplex ───────────────────────────────────────

banner("3. Two-strand duplex design")

# Two 8-nt strands that should hybridize into a clean 8-bp duplex.
dup_target = "((((((((+))))))))"        # 8 '(' + nick + 8 ')'
nicks = [8]
obj_dup = DiffObjective.ensemble_defect(dup_target)
res_dup = designer.design(
    {"top": DomainSpec(length=8, material="rna"),
     "bot": DomainSpec(length=8, material="rna")},
    obj_dup, n_restarts=12, n_steps=180, lr=0.25, nicks=nicks, sa_polish=False,
)
top, bot = res_dup.sequences["top"], res_dup.sequences["bot"]
print(f"  top strand : 5'-{top}-3'")
print(f"  bot strand : 5'-{bot}-3'")
print(f"  native complex defect : "
      f"{engine.ensemble_defect((top, bot), dup_target, normalize=True):.4f}")
print(f"  native complex ΔG     : {engine.pfunc(top, bot).free_energy:+.2f} kcal/mol")

print("\nDone.")
