"""
Example 05: Leakage enumeration and off-target screening

Covers:
- LeakageEnumerator: find spurious pathways in a CHA circuit
- LeakageReport: filter, rank, and export to mantis strings
- OffTargetScreener: k-mer index + ΔΔG ranking against a sequence database
- Specificity: ΔΔG(target) vs ΔΔG(off-targets) as a selectivity metric
- Visualization: leakage pathway bar chart, specificity landscape
"""

import matplotlib.pyplot as plt
import numpy as np
from strider import ThermoEngine
from strider.kinetics.leakage import LeakageEnumerator
from strider.screen.offtarget import OffTargetScreener

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

MIR21 = "TAGCTTATCAGACTGATGTTGA"
H1    = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2    = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
CP    = "AAAAA"

# ── 1. Leakage enumeration ───────────────────────────────────────────────────
print("── LeakageEnumerator ────────────────────────────────────────")

enumerator = LeakageEnumerator(
    engine,
    ddg_threshold=-3.0,   # only flag pathways more favorable than -3 kcal/mol
    max_complex_size=3,
    max_pathways=50,
)

intended = [
    "miR21 + H1 <-> miR21_H1",
    "miR21_H1 + H2 <-> H1H2 + miR21",
    "H1H2 + CP <-> H1H2_CP",
]

report = enumerator.enumerate(
    strands={"H1": H1, "H2": H2, "CP": CP, "miR21": MIR21},
    intended_reactions=intended,
)

print(f"\n  Total spurious pathways found:  {len(report.reactions)}")
print(f"  Worst ΔΔG:  {report.worst_ddg:.2f} kcal/mol")
print(f"  Mean  ΔΔG:  {np.mean([r.ddg for r in report.reactions]):.2f} kcal/mol"
      if report.reactions else "  (none found)")

print(f"\n  Top 5 most favorable spurious reactions:")
for rxn in sorted(report.reactions, key=lambda r: r.ddg)[:5]:
    print(f"    {rxn.ddg:+.2f} kcal/mol  [{rxn.pathway_type}]  "
          f"{' + '.join(rxn.reactant_names)} → complex")

# Export to mantis-compatible strings
mantis_strings = report.to_mantis_strings()
print(f"\n  mantis-compatible reaction strings ({len(mantis_strings)} total):")
for s in mantis_strings[:4]:
    print(f"    {s}")

# Filter to only very unfavorable leakage (dangerous ones)
dangerous = report.filter(ddg_threshold=-5.0)
print(f"\n  Pathways more favorable than -5 kcal/mol: {len(dangerous.reactions)}")

# ── 2. Leakage suppression by design — compare two stem stabilities ──────────
print("\n── Leakage suppression: weak vs strong stem ─────────────────")

# Weak stem: 8-nt loop region (easier breathing)
H1_weak = "TCAACATCAGTCTGATAAAAAATCAGACTGA"   # reduced stem — shorter
# Strong stem: original H1
H1_strong = H1

for label, seq in [("H1 weak stem", H1_weak), ("H1 strong stem", H1_strong)]:
    g = engine.pfunc(seq).free_energy
    from strider.kinetics.tmsd import leakage_kf
    k_leak = leakage_kf(abs(g), kf_max=1e6, celsius=37.0)
    print(f"  {label}: G = {g:+.2f} kcal/mol  →  k_leak = {k_leak:.2e} M⁻¹s⁻¹")

# ── 3. Off-target screening ───────────────────────────────────────────────────
print("\n── OffTargetScreener ────────────────────────────────────────")

# Simulate a small miRNA family database (miR-21 and related sequences)
MIR_FAMILY = {
    "miR-21-5p":  "TAGCTTATCAGACTGATGTTGA",   # our target
    "miR-21-3p":  "CAACACCAGTCGATGGGCTGT",    # seed-family member
    "miR-155-5p": "TTAATGCTAATCGTGATAGGGGT",  # different family
    "miR-210-3p": "CTGTGCGTGTGACAGCGGCTGA",
    "miR-let-7a": "TGAGGTAGTAGGTTGTATAGTT",
    "miR-17-5p":  "CAAAGUGCTTACAGTGCAGGTAG".replace("U", "T"),
    "miR-92a":    "TATTGCACTTGTCCCGGCCTGT",
    "miR-141":    "TAACACUGUCUGGUAAAGAUGG".replace("U", "T"),
}

screener = OffTargetScreener(engine, kmer_k=6)
screener.add_sequences(MIR_FAMILY)

print(f"  Database: {len(MIR_FAMILY)} sequences loaded")

# Screen H1 against the whole family
screening_report = screener.screen(H1, n_top=8, ddg_threshold=-2.0)
print(f"\n  H1 off-target screen results (top {len(screening_report.hits)} hits):")
print(f"  {'Sequence':<14} {'ΔΔG (kcal/mol)':<20} {'k-mer hits'}")
for hit in screening_report.hits:
    print(f"  {hit.name:<14} {hit.ddg:<20.2f} {hit.k_score}")

# Specificity: compare ΔΔG(target) vs ΔΔG(off-targets)
specificity = screener.specificity_vs(
    H1,
    family_members={k: v for k, v in MIR_FAMILY.items() if k != "miR-21-5p"},
    target=MIR_FAMILY["miR-21-5p"],
)
print(f"\n  Specificity scores (target / off-target ΔΔG ratio):")
for name, score in sorted(specificity.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    {name:<16} selectivity = {score:.3f}")

# ── 4. Export formats ─────────────────────────────────────────────────────────
print("\n── Export: H1 in multiple formats ───────────────────────────")
from strider.export.formats import to_vienna, to_fasta, to_ct
from strider.structure.mfe import fold_mfe

struct, energy, _ = fold_mfe(H1, celsius=37.0)
print("  Vienna format:")
print("  " + to_vienna(H1, struct, name="H1"))
print("  FASTA format:")
print("  " + to_fasta(H1, name="H1", description="CHA hairpin 1, miR-21 biosensor"))
print("  CT (first 3 lines):")
ct = to_ct(H1, struct, name="H1", energy=energy)
for line in ct.split("\n")[:3]:
    print(f"  {line}")

# ── 5. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("strider — Leakage & Off-target Screening", fontsize=13, fontweight="bold")

# Leakage ΔΔG histogram
ax = axes[0]
if report.reactions:
    ddgs = [r.ddg for r in report.reactions]
    ax.hist(ddgs, bins=15, color="#dd7744", edgecolor="white")
    ax.axvline(-5.0, color="#dd3333", linestyle="--", label="Dangerous threshold")
    ax.axvline(-3.0, color="#ff9933", linestyle="--", label="Filter threshold")
    ax.set_xlabel("ΔΔG (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Spurious pathway ΔΔG distribution")
    ax.legend()
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "No leakage pathways\nfound above threshold",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Spurious pathway ΔΔG distribution")

# Off-target ΔΔG comparison
ax = axes[1]
hits = screening_report.hits
if hits:
    names = [h.name for h in hits]
    ddgs  = [h.ddg  for h in hits]
    colors = ["#dd3333" if n == "miR-21-5p" else "#4488dd" for n in names]
    bars = ax.barh(names, ddgs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔΔG (kcal/mol)")
    ax.set_title("H1 off-target binding ΔΔG\n(red = target miR-21)")
    ax.grid(alpha=0.3, axis="x")
else:
    ax.text(0.5, 0.5, "No hits above threshold", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig("leakage_screening.png", dpi=120, bbox_inches="tight")
print("\nSaved: leakage_screening.png")
print("\nDone.")
