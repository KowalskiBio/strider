"""
Example 02: Hairpin folding — structure prediction and visualization

Covers:
- MFE folding with dot-bracket output
- Partition function and base-pair probabilities
- Arc diagram and mountain plot visualization
- H-type pseudoknot detection
- Comparing natural vs designed hairpin structures
"""

import numpy as np
import matplotlib.pyplot as plt
from strider import ThermoEngine
from strider.structure.dot_bracket import parse_pairs, stem_regions, unpaired_positions
from strider.structure.mfe import fold_mfe
from strider.structure.pseudoknot import fold_pseudoknot
from strider.viz.arc import arc_diagram
from strider.viz.mountain_plot import mountain_plot

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)
# Native engine used for pair-probability calculations: the NUPACK backend
# returns ensemble ΔG accurately but our wrapper does not call nupack.pairs().
engine_native = ThermoEngine(material="dna", celsius=37.0, sodium=0.137,
                              magnesium=0.01, backend="native")

# ── 1. MFE folding of H1 and H2 ─────────────────────────────────────────────
H1 = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2 = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"

print("── MFE Folding ──────────────────────────────────────────────")
for name, seq in [("H1", H1), ("H2", H2)]:
    structure, energy, pairs = fold_mfe(seq, celsius=37.0, material="dna")
    stems   = stem_regions(structure)
    unpaired = unpaired_positions(structure)
    print(f"\n  {name} ({len(seq)} nt)")
    print(f"    Sequence:  {seq}")
    print(f"    Structure: {structure}")
    print(f"    ΔG_MFE  = {energy:+.2f} kcal/mol")
    print(f"    Stems   : {stems}")
    print(f"    Unpaired: {len(unpaired)} positions")

# ── 2. Base-pair probabilities from partition function ───────────────────────
# Using native backend: it runs the McCaskill DP which produces pair_probs.
# (The NUPACK backend gives more accurate ΔG but our wrapper omits nupack.pairs().)
print("\n── Base-pair probability profiles (native DP) ───────────────")
for name, seq in [("H1", H1), ("H2", H2)]:
    result_nupack = engine.pfunc(seq)          # accurate ΔG
    result_native = engine_native.pfunc(seq)   # pair probabilities
    probs = result_native.pair_probs
    p_paired = probs.sum(axis=1)
    toehold_p = p_paired[:6].mean() if name == "H1" else None
    print(f"\n  {name}: ΔG (nupack) = {result_nupack.free_energy:+.2f} kcal/mol  "
          f"ΔG (native) = {result_native.free_energy:+.2f}")
    print(f"    Mean pairing prob (all):      {p_paired.mean():.3f}")
    if toehold_p is not None:
        print(f"    Mean pairing prob (toehold):  {toehold_p:.3f}  ← low = accessible")

# ── 3. Pseudoknot search ────────────────────────────────────────────────────
# Synthetic sequence with a known H-type pseudoknot for demonstration
print("\n── Pseudoknot detection ─────────────────────────────────────")
# Simple H-type: stem1 = AAAA:TTTT, stem2 = CCCC:GGGG with crossing
PK_SEQ = "AAAALLLCCCCLLTTTTLLLGGGG".replace("L", "A")  # placeholder loop
PK_SEQ = "AAAACCCCAAAATTTTGGGG"  # simplified test sequence
struct_pk, energy_pk, pairs_pk = fold_pseudoknot(PK_SEQ, celsius=37.0)
print(f"  Sequence:  {PK_SEQ}")
print(f"  Structure: {struct_pk}  ([] = pseudoknot pairs)")
print(f"  Energy:    {energy_pk:+.2f} kcal/mol")
if "[" in struct_pk:
    print("  → H-type pseudoknot detected")
else:
    print("  → No pseudoknot found (simple stem-loop)")

# ── 4. Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("strider — Hairpin Folding Analysis", fontsize=14, fontweight="bold")

# Arc diagrams (native engine provides pair probabilities)
arc_diagram(H1, engine=engine_native, ax=axes[0][0], title="H1 arc diagram (pair probs)")
arc_diagram(H2, engine=engine_native, ax=axes[0][1], title="H2 arc diagram (pair probs)")

# Mountain plots
mountain_plot(H1, engine=engine_native, ax=axes[1][0], title="H1 mountain plot")
mountain_plot(H2, engine=engine_native, ax=axes[1][1], title="H2 mountain plot")

plt.tight_layout()
plt.savefig("hairpin_folding.png", dpi=120, bbox_inches="tight")
print("\nSaved: hairpin_folding.png")

# ── 5. Temperature scan: how structure changes from 20 to 70°C ───────────────
print("\n── H1 stability across temperatures ────────────────────────")
print(f"  {'T (°C)':<10} {'ΔG (kcal/mol)':<18} {'Toehold access.'}")
for celsius in [20, 30, 37, 45, 55, 65]:
    eng_t = ThermoEngine(material="dna", celsius=celsius, sodium=0.137, magnesium=0.01)
    eng_t_nat = ThermoEngine(material="dna", celsius=celsius, sodium=0.137,
                              magnesium=0.01, backend="native")
    pf  = eng_t.pfunc(H1)
    acc = eng_t_nat.toehold_accessibility(H1, list(range(6)))
    print(f"  {celsius:<10} {pf.free_energy:<18.2f} {acc:.3f}")

print("\nDone.")
