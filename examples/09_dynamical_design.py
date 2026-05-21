"""
Example 09: Closed-loop dynamical sequence design (mantis-feedback driven)

Drives sequence optimization from a *kinetic* cost — an ODE simulation —
instead of a static equilibrium defect.  Every score evaluation:

    1.  Take the candidate sequence
    2.  Rebuild a strider CircuitBridge (recomputes ΔΔG per reaction)
    3.  Convert ΔΔG → rates via Zhang & Winfree (2009)
    4.  Hand the rate dict to mantis as a CRNetwork
    5.  Run an ODE simulation (deterministic Radau)
    6.  Score the trajectory and feed the scalar back to the SA optimizer

The closed loop is honest: the rate constants reflect the *real* binding
affinity of the latest sequence to its partners, not a frozen rate table.

── Demonstrated objective ──────────────────────────────────────────────────
This example showcases `DesignObjective.kinetic_trajectory`, the canonical
use case from outperform_nupack.md item 1:  "Optimize sequences to match a
target step-response."  We specify a desired [AB](t) curve and let the
optimizer mutate the toehold of A until the simulated trajectory matches.

Other dynamical factories — `maximize_kcat`, `minimize_leak`,
`bistable_threshold`, `from_simulation` — follow the same pattern (closure
over `bridge.to_crnetwork`, ODE inside the score function); see the unit
tests in `tests/test_design_dynamical.py` for self-contained examples of
each.

── Circuit ─────────────────────────────────────────────────────────────────
A single bimolecular hybridization step:

    A + B  <->  AB

Strand A is just the 7-nt designed toehold (no flanking tail), so ΔΔG of
A·B is fully sequence-dependent.  Across all 7-mers, ΔΔG against the fixed
B partner spans ≈ −1 to −8 kcal/mol — a ≳10⁴× spread in equilibrium
constant Keq.  With [B] = 10 µM the well-matched sequences saturate
[AB] → A₀ = 10 nM, while mismatched sequences plateau orders of magnitude
below that.  The optimizer searches the 4⁷ = 16,384 candidates for one
whose trajectory hugs the target curve.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np

from strider import (
    CircuitBridge,
    DesignObjective,
    DomainSpec,
    HardConstraint,
    SequenceDesigner,
    ThermoEngine,
)

_here = pathlib.Path(__file__).parent

# ── Engine + fixed strand ──────────────────────────────────────────────────
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

# Fixed partner.  Strand A is *just* the 7-nt toehold — no flanking tail —
# so the ΔΔG of A·B is dominated by the toehold sequence (no extra tail
# region to fix the affinity at a high baseline).  Different 7-mers give
# ΔΔG ranging from ≈ −1.2 kcal/mol (poor match) to ≈ −7.8 kcal/mol
# (reverse complement of B's 5' end) — a ~6 kcal/mol spread that maps onto
# a ~10⁴× spread in Kd, and therefore a clearly distinguishable [AB]
# trajectory across sequences.
B_SEQ = "GCAGTGAGACGAGCTGCT"     # 18 nt fixed partner


def make_factory(domain_name: str = "toehold_A"):
    """Closure: ``(seqs) -> mantis.CRNetwork``."""
    reactions = ["A + B <-> AB"]

    def factory(seqs: dict[str, str]):
        toehold = seqs[domain_name]
        bridge = CircuitBridge(
            reactions=reactions,
            sequences={"A": toehold, "B": B_SEQ},
            engine=engine,
            toehold_map={"A + B <-> AB": len(toehold)},
        )
        return bridge.to_crnetwork()

    return factory


factory = make_factory()

# ── Target step-response curve ─────────────────────────────────────────────
# Initial conditions: 10 nM A + 10 µM B (B in 1000× excess so [B] stays
# approximately constant; the apparent first-order rate is k_obs = kf·[B] +
# kr, and the equilibrium [AB]/A₀ = Keq·[B] / (1 + Keq·[B])).  A strong
# toehold (Keq · [B] ≫ 1) saturates [AB] → A₀; a weak toehold doesn't.
#
# Note on time constant: with kf ≈ 4 × 10⁶ M⁻¹s⁻¹ (Zhang-Winfree 7-nt) and
# [B] = 10 µM, the observed first-order rate kf·[B] = 40 s⁻¹ — the system
# equilibrates in ≈ 0.1 s regardless of sequence.  This is a physical
# limit of the rate model (kf depends on toehold *length*, not sequence —
# only kr varies with ΔΔG).  So the design lever is the **plateau height**,
# not the time constant: a target with a short rise time (τ = 0.3 s)
# reflects what's actually achievable, and the optimizer matches the
# plateau, not the shape.
IC = {"A": 1e-8, "B": 1e-5, "AB": 0.0}
t_window = (0.0, 10.0)
t_eval = np.linspace(*t_window, 121)

A0 = 1e-8           # M (saturating plateau target)
tau_target = 0.3    # seconds (matches the fastest achievable kinetics)
target_AB = A0 * (1.0 - np.exp(-t_eval / tau_target))
target_curve = {"AB": target_AB}

# ── Composed objective ─────────────────────────────────────────────────────
objective = (
    # Primary: match the target [AB](t) step response.
    1.0 * DesignObjective.kinetic_trajectory(
        factory, IC, target_curve, t_eval, label="trajectory_MSE",
    )
    # Mild static regularizer so the toehold stays synthesizable.
    + 0.05 * DesignObjective.gc_content("toehold_A", target_gc=0.55)
)

# ── Baseline: deliberately poor (low-affinity, GC-imbalanced) toehold ─────
baseline_seqs = {"toehold_A": "ATATATA"}    # weak hybridization, low GC
baseline_breakdown = objective.evaluate_breakdown(baseline_seqs)
baseline_score = sum(baseline_breakdown.values())

print("── Baseline (deliberately weak toehold) ──────────────────")
print(f"  toehold_A = {baseline_seqs['toehold_A']}")
for label, val in baseline_breakdown.items():
    print(f"    {label:<42}  {val:+.4f}")
print(f"  total = {baseline_score:+.4f}")

# ── Optimize ───────────────────────────────────────────────────────────────
print("\n── Optimizing (closed-loop mantis feedback) ──────────────")
print("  Each SA step rebuilds the CRN and reruns the ODE against the target curve.")
designer = SequenceDesigner(engine=engine, seed=3)
result = designer.design(
    domains={"toehold_A": DomainSpec(length=7, material="dna")},
    objective=objective,
    hard_constraints=[
        HardConstraint.gc_content(min_gc=0.4, max_gc=0.8),
        HardConstraint.max_run(max_run_length=3),
    ],
    n_trials=3,
    max_iterations=60,
    verbose=False,
)

best_toehold = result.sequences["toehold_A"]
best_breakdown = result.objective_breakdown
print(f"\n  best toehold_A = {best_toehold}")
for label, val in best_breakdown.items():
    print(f"    {label:<42}  {val:+.4f}")
print(f"  total = {result.objective_value:+.4f}")
print(f"  improvement = {baseline_score - result.objective_value:+.4f}  (positive = better)")

# ── Visualize: target vs. baseline vs. optimized ──────────────────────────
sim_base = factory(baseline_seqs).simulate(IC, t_span=t_window, t_eval=t_eval)
sim_best = factory(result.sequences).simulate(IC, t_span=t_window, t_eval=t_eval)

# Report each curve's *own* equilibrium plateau (= last-sample value).
# The optimizer's job is to drive this plateau toward the target's plateau
# (= A₀ = 10 nM), since the time-constant is essentially fixed by the
# Zhang-Winfree kf model.
def plateau(arr: np.ndarray) -> float:
    return float(arr[-1])

plateau_target = plateau(target_AB) * 1e9
plateau_base   = plateau(sim_base.concentrations["AB"]) * 1e9
plateau_best   = plateau(sim_best.concentrations["AB"]) * 1e9

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    "Closed-loop dynamical design: kinetic-trajectory matching",
    fontsize=12, fontweight="bold",
)

# Left panel: full window, target + baseline + optimized
ax = axes[0]
ax.plot(t_eval, target_AB * 1e9, ":", color="#222", lw=2.0,
        label=f"target  (plateau {plateau_target:.2f} nM)")
ax.plot(t_eval, sim_base.concentrations["AB"] * 1e9, "--",
        color="#aaaaaa", lw=2.0,
        label=f"baseline {baseline_seqs['toehold_A']}  (plateau {plateau_base:.2f} nM)")
ax.plot(t_eval, sim_best.concentrations["AB"] * 1e9, "-",
        color="#22882f", lw=2.5,
        label=f"optimized {best_toehold}  (plateau {plateau_best:.2f} nM)")
ax.axhline(plateau_target, color="#222", linestyle=":", lw=0.5, alpha=0.4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("[AB] (nM)")
ax.set_title(
    f"Trajectory match — target τ = {tau_target:.1f} s, "
    f"plateau = {plateau_target:.1f} nM"
)
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.3)

# Right panel: per-trial SA convergence — each bar is the best objective
# score reached on that trial, with the global best highlighted.
ax = axes[1]
trial_scores = result.trial_scores
n_trials = len(trial_scores)
colors = ["#22882f" if s == result.objective_value else "#7799cc"
          for s in trial_scores]
bars = ax.bar(range(1, n_trials + 1), trial_scores, color=colors,
              edgecolor="#222", linewidth=0.5)
ax.axhline(baseline_score, color="#aaaaaa", linestyle="--", lw=1.5,
           label=f"baseline ({baseline_seqs['toehold_A']}) = {baseline_score:.3f}")
ax.axhline(result.objective_value, color="#22882f", linestyle=":", lw=1.5,
           label=f"best ({best_toehold}) = {result.objective_value:.3f}")
for i, s in enumerate(trial_scores):
    ax.text(i + 1, s + max(trial_scores) * 0.02, f"{s:.3f}",
            ha="center", fontsize=8)
ax.set_xticks(range(1, n_trials + 1))
ax.set_xlabel("SA trial")
ax.set_ylabel("Objective (lower = better)")
ax.set_title("Closed-loop SA convergence\n(each bar = best of one annealing run)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = _here / "dynamical_design.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out.name}")
print("\nDone.")
