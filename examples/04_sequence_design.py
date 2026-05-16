"""
Example 04: Sequence design with composable objectives

Covers:
- DesignObjective factory methods and composition
- HardConstraint (no repeats, GC content, max run)
- SequenceDesigner (simulated annealing)
- DomainSpec: fixed vs free domains
- MutationAnalyzer: single-nt scan and robustness score
- Visualizing the design result

Scoring convention throughout:
  quality  ∈ [0, 1]   higher = better   (1.0 = perfect match to all objectives)
  penalty  ∈ [0, ∞)   lower  = better   (internal minimisation target)
  quality  = 1 / (1 + penalty)
"""

import math
import pathlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np
from strider import ThermoEngine
from strider.design.objective import DesignObjective
from strider.design.constraints import HardConstraint
from strider.design.optimizer import SequenceDesigner, DomainSpec
from strider.design.mutation import MutationAnalyzer

_here = pathlib.Path(__file__).parent

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


def quality(penalty: float) -> float:
    """Convert a penalty score (0 = perfect) to a quality score (1.0 = perfect)."""
    return 1.0 / (1.0 + penalty)


def gc_pct(seq: str) -> float:
    return sum(c in "GC" for c in seq) / len(seq) if seq else 0.0


def quality_bar(q: float, width: int = 20) -> str:
    filled = round(q * width)
    return "█" * filled + "░" * (width - filled)


# ── 1. Composable design objectives ─────────────────────────────────────────
print("── Composable objectives ────────────────────────────────────")
print("  quality = 1/(1+penalty)  |  1.00 = perfectly satisfies objective")
print()

obj_gc       = DesignObjective.gc_content("D", target_gc=0.5, weight=1.0, label="gc")
obj_gc_soft  = DesignObjective.gc_content("D", target_gc=0.5, weight=0.5, label="gc_soft")
obj_combined = obj_gc + obj_gc_soft

test_cases = [
    ("ACGTACGTACGT", "50% GC — target"),
    ("GCGCGCGCGCGC", "100% GC — too G/C-rich"),
    ("AAAAAAAAAAAA", "0% GC — all AT"),
    ("GGGCCCAAATTT", "50% GC, uneven distribution"),
]

print(f"  {'Sequence':<20} {'GC%':<8} {'Quality':<12} {'Bar'}")
for seq, desc in test_cases:
    seqs = {"D": seq}
    pen = obj_combined(seqs)
    q   = quality(pen)
    print(f"  {seq:<20} {gc_pct(seq):>5.0%}    {q:.3f}       [{quality_bar(q)}]  {desc}")

# Scalar multiplication raises weight (lowers quality faster for mismatches)
print("\n  Effect of 3× weight on all-A sequence:")
obj_3x = 3.0 * obj_gc
pen_1x = obj_gc({"D": "AAAAAAAAAAAA"})
pen_3x = obj_3x({"D": "AAAAAAAAAAAA"})
print(f"    1× weight → quality = {quality(pen_1x):.3f}")
print(f"    3× weight → quality = {quality(pen_3x):.3f}  (penalised more strongly)")

# Custom callable objective
def at_streak_count(seqs):
    seq = seqs.get("D", "")
    return float(sum(1 for i in range(len(seq) - 1) if seq[i:i+2] in ("AT", "TA")))

obj_at   = DesignObjective.from_callable(at_streak_count, label="at_streaks", weight=0.3)
obj_full = obj_gc + obj_at

print("\n  Combined GC + AT-streak objective:")
for seq, desc in [("ACGTACGTACGT", "50% GC, balanced"), ("ATATATATAT", "alternating AT")]:
    seqs = {"D": seq}
    streaks = int(at_streak_count(seqs))
    pen = obj_full(seqs)
    q   = quality(pen)
    print(f"    {seq:<20} GC={gc_pct(seq):.0%}  streaks={streaks}  "
          f"quality={q:.3f} [{quality_bar(q, 15)}]  {desc}")

# ── 2. Hard constraints ──────────────────────────────────────────────────────
print("\n── Hard constraints (pass/fail — must satisfy all to be used) ─")
c_no_repeats = HardConstraint.no_repeats(["AAAA", "CCCC", "GGGG", "TTTT"])
c_gc         = HardConstraint.gc_content(min_gc=0.35, max_gc=0.65)
c_run        = HardConstraint.max_run(max_run_length=3)

print(f"  {'OK':<4} {'Sequence':<20} {'no_repeat':<12} {'35–65% GC':<12} {'max run ≤3'}")
for seq, desc in [
    ("ACGTACGTACGT", "balanced"),
    ("AAAAACGTACGT", "4-A run"),
    ("AAAA",         "all-A"),
    ("GCGCGCGCGCGC", "100% GC"),
    ("CATGCATGCATG", "good design"),
]:
    no_rep = c_no_repeats.check("D", seq)
    gc_ok  = c_gc.check("D", seq)
    run_ok = c_run.check("D", seq)
    icon   = "✓" if (no_rep and gc_ok and run_ok) else "✗"
    print(f"  {icon:<4} {seq:<20} {str(no_rep):<12} {str(gc_ok):<12} {str(run_ok)}  ({desc})")

# ── 3. Sequence design: 9-nt toehold with GC target ─────────────────────────
print("\n── Design run: 9-nt toehold, target GC = 44% ───────────────")
print("  Running 5 SA trials — showing quality per trial (↑ better):\n")

obj_toehold = DesignObjective.gc_content("T1", target_gc=0.44, weight=1.0) + obj_at

designer = SequenceDesigner(engine=engine, seed=42)
result = designer.design(
    domains={"T1": DomainSpec(length=9)},
    objective=obj_toehold,
    hard_constraints=[c_no_repeats, c_gc, c_run],
    n_trials=5,
    max_iterations=300,
    verbose=False,   # we display quality ourselves below
)

for i, trial_pen in enumerate(result.trial_scores, 1):
    q   = quality(trial_pen)
    bar = quality_bar(q, 25)
    star = " ← best" if trial_pen == result.objective_value else ""
    print(f"  Trial {i}: [{bar}] {q:.3f}{star}")

best_seq = result.sequences["T1"]
best_gc  = gc_pct(best_seq)
best_q   = quality(result.objective_value)
streaks  = int(at_streak_count({"D": best_seq}))

print(f"\n  Best sequence : {best_seq}")
print(f"  GC content    : {best_gc:.0%}  (target 44%)")
print(f"  AT streaks    : {streaks}")
print(f"  Overall quality: {best_q:.4f}  [{quality_bar(best_q)}]")
print(f"  Converged     : {result.converged}")

# Component quality breakdown
breakdown = result.objective_breakdown
print("\n  Component breakdown:")
for name, comp_pen in breakdown.items():
    comp_q = quality(comp_pen)
    print(f"    {name:<38} quality = {comp_q:.4f} [{quality_bar(comp_q, 15)}]")

# ── 4. Fixed + free domain co-design ────────────────────────────────────────
print("\n── Co-design: fix anchor sequence, optimise 12-nt linker ────")
ANCHOR = "GCATGC"   # generic 6-nt GC anchor — kept fixed

result2 = designer.design(
    domains={
        "anchor": DomainSpec(sequence=ANCHOR),   # fixed
        "linker": DomainSpec(length=12),           # free
    },
    objective=DesignObjective.gc_content("linker", target_gc=0.5),
    hard_constraints=[c_no_repeats, c_gc],
    n_trials=3,
    max_iterations=200,
    verbose=False,
)
linker_gc = gc_pct(result2.sequences["linker"])
linker_q  = quality(result2.objective_value)
print(f"  anchor (fixed):   {result2.sequences['anchor']}  (GC = {gc_pct(ANCHOR):.0%})")
print(f"  linker (designed): {result2.sequences['linker']}  "
      f"(GC = {linker_gc:.0%}, quality = {linker_q:.3f})")

# ── 5. Mutation analysis ─────────────────────────────────────────────────────
print("\n── Mutation sensitivity of designed toehold ─────────────────")
analyzer = MutationAnalyzer(engine)
profile  = analyzer.single_nt_scan(best_seq)
robustness = analyzer.robustness_score(best_seq, ddg_tolerance=1.0)
critical   = profile.critical_positions(threshold=0.3)

print(f"  Sequence:  {best_seq}")
print(f"  Robustness: {robustness:.3f}  [{quality_bar(robustness)}]  "
      f"(1.0 = all mutations are neutral)")
print(f"  Critical positions (Δscore > 0.3): {critical if critical else 'none'}")

# Per-position disruption summary
print("\n  Per-position mutation impact:")
bases = "ACGT"
orig  = list(best_seq)
print(f"  {'Pos':<5} {'Base':<6} {'Max Δscore':<14} {'Impact'}")
for i in range(len(best_seq)):
    max_delta = float(profile.delta_score[i].max())
    impact    = "HIGH" if max_delta > 0.5 else ("MED" if max_delta > 0.1 else "low")
    bar       = quality_bar(max(0.0, 1.0 - max_delta), 10)
    print(f"  {i:<5} {orig[i]:<6} {max_delta:<14.3f} [{bar}] {impact}")

# ── 6. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("strider — Sequence Design & Mutation Analysis", fontsize=13, fontweight="bold")

# Trial quality bars (higher = better)
ax = axes[0]
trial_qualities = [quality(p) for p in result.trial_scores]
colors = ["#44aa44" if q == max(trial_qualities) else "#4488dd" for q in trial_qualities]
ax.bar(range(1, len(trial_qualities) + 1), trial_qualities, color=colors)
ax.axhline(max(trial_qualities), color="#22882f", linestyle="--",
           label=f"Best = {max(trial_qualities):.4f}")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Trial")
ax.set_ylabel("Quality score (higher = better)")
ax.set_title("Design trial quality\n(1.0 = perfectly satisfies objective)")
ax.legend()
ax.grid(alpha=0.3, axis="y")

# Mutation stability heatmap (inverted: green = stays good, red = disrupted)
ax = axes[1]
n_pos = profile.delta_score.shape[0]
# Convert Δscore → stability: high stability (low disruption) = green
stability = np.clip(1.0 - profile.delta_score, 0, 1)
im = ax.imshow(stability.T, aspect="auto", cmap="RdYlGn",
               vmin=0.0, vmax=1.0)
ax.set_xticks(range(n_pos))
ax.set_xticklabels(
    [f"{orig[i]}{i}" for i in range(n_pos)], fontsize=8, rotation=45, ha="right"
)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["alt 1", "alt 2", "alt 3"])
ax.set_title("Mutation stability\n(green = mutation neutral, red = disruptive)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Stability (1.0 = neutral, 0.0 = disruptive)")

plt.tight_layout()
fig.savefig(_here / "sequence_design.png", dpi=150, bbox_inches="tight")
print("\nSaved: sequence_design.png")
print("\nDone.")
