"""
Example 05: Leakage enumeration and off-target screening

Covers:
- LeakageEnumerator: find spurious pathways in a strand displacement circuit
- LeakageReport: filter, rank, and export to mantis strings
- OffTargetScreener: k-mer index + ΔΔG ranking against a sequence database
- Specificity: ΔΔG(target) vs ΔΔG(off-targets) as a selectivity metric
- Visualization: leakage ΔΔG distribution, off-target binding landscape

Circuit: a 3-strand catalytic displacement (Target opens Probe, Fuel drives recycling).
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np
from strider import ThermoEngine
from strider.kinetics.leakage import LeakageEnumerator
from strider.screen.offtarget import OffTargetScreener

_here = pathlib.Path(__file__).parent

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

# Generic 3-strand displacement circuit (non-repetitive sequences)
PROBE  = "CGCAGTCGATCAGTACGCTG"   # 20 nt probe recognition strand
TARGET = "CAGCGTACTGATCGACTGCG"   # 20 nt target (reverse complement of PROBE)
FUEL   = "TAGCGTACTCGATCGCATAG"   # 20 nt fuel strand (drives recycling)
OUTPUT = "GCTACGATCGACTGCATCGA"   # 20 nt output indicator strand

# ── 1. Leakage enumeration ───────────────────────────────────────────────────
print("── LeakageEnumerator ────────────────────────────────────────")

enumerator = LeakageEnumerator(
    engine,
    ddg_threshold=-3.0,   # only flag pathways more favorable than -3 kcal/mol
    max_complex_size=3,
    max_pathways=50,
)

intended = [
    "Target + Probe <-> Target_Probe",
    "Target_Probe + Fuel <-> Probe_Fuel + Target",
    "Probe_Fuel + Output <-> Fuel_Out + Probe",
]

report = enumerator.enumerate(
    strands={"Probe": PROBE, "Target": TARGET, "Fuel": FUEL, "Output": OUTPUT},
    intended_reactions=intended,
)

print(f"\n  Total spurious pathways found:  {len(report.reactions)}")
print(f"  Worst ΔΔG:  {report.worst_ddg:.2f} kcal/mol")
if report.reactions:
    print(f"  Mean  ΔΔG:  {np.mean([r.ddg for r in report.reactions]):.2f} kcal/mol")

print(f"\n  Top 5 most favorable spurious reactions:")
for rxn in sorted(report.reactions, key=lambda r: r.ddg)[:5]:
    print(f"    {rxn.ddg:+.2f} kcal/mol  [{rxn.pathway_type}]  "
          f"{' + '.join(rxn.reactant_names)} → complex")

mantis_strings = report.to_mantis_strings()
print(f"\n  mantis-compatible reaction strings ({len(mantis_strings)} total):")
for s in mantis_strings[:4]:
    print(f"    {s}")

dangerous = report.filter(ddg_threshold=-5.0)
print(f"\n  Pathways more favorable than -5 kcal/mol: {len(dangerous.reactions)}")

# ── 2. Leakage suppression: weak vs strong probe stem ───────────────────────
print("\n── Leakage suppression: weak vs strong stem ─────────────────")

PROBE_WEAK   = "ATATTTTTTTTTTTTATAT"   # 4-nt AT stem (lower stability, more breathing)
PROBE_STRONG = "CGCGTTTTTTTTTTTTCGCG"  # 4-nt GC stem (higher stability)

for label, seq in [("Probe weak  (AT stem)", PROBE_WEAK),
                   ("Probe strong (GC stem)", PROBE_STRONG)]:
    g = engine.pfunc(seq).free_energy
    from strider.kinetics.tmsd import leakage_kf
    k_leak = leakage_kf(abs(g), kf_max=1e6, celsius=37.0)
    print(f"  {label}: G = {g:+.2f} kcal/mol  →  k_leak = {k_leak:.2e} M⁻¹s⁻¹")

# ── 3. Off-target screening ───────────────────────────────────────────────────
print("\n── OffTargetScreener ────────────────────────────────────────")

# Cross-reactivity database: exact target (RC of PROBE), single/triple mismatches,
# and unrelated sequences — designed to produce a clear energy gradient.
PROBE_DB = {
    "exact_target":  "CAGCGTACTGATCGACTGCG",  # RC of PROBE — full complement
    "mismatch_1nt":  "CAGCGTTCTGATCGACTGCG",  # 1 substitution (A→T at pos 7)
    "mismatch_3nt":  "CAGCGTATCGATCGACTACG",  # 3 substitutions
    "off_target_A":  "GTAGCATCGTAGCATCGATA",  # unrelated, low complementarity
    "off_target_B":  "GTCAGTCAGTCAGTCAGTCA",  # GTC-repeat, different composition
    "off_target_C":  "GCGCGCGCGCGCGCGCGCGC",  # GC-only run
    "off_target_D":  "TATATATATATATATATATAT",  # AT-repeat
    "off_target_E":  "GCTAGCTAGCTAGCTAGCTA",  # TAG-repeat
}

screener = OffTargetScreener(engine, kmer_k=6)
screener.add_sequences(PROBE_DB)

print(f"  Database: {len(PROBE_DB)} sequences loaded")

screening_report = screener.screen(PROBE, n_top=8, ddg_threshold=-2.0)
print(f"\n  PROBE off-target screen results (top {len(screening_report.hits)} hits):")
print(f"  {'Sequence':<16} {'ΔΔG (kcal/mol)':<20} {'k-mer hits'}")
for hit in screening_report.hits:
    print(f"  {hit.name:<16} {hit.ddg:<20.2f} {hit.k_score}")

specificity = screener.specificity_vs(
    PROBE,
    family_members={k: v for k, v in PROBE_DB.items() if k != "exact_target"},
    target=PROBE_DB["exact_target"],
)
print(f"\n  Specificity scores (target / off-target ΔΔG ratio):")
for name, score in sorted(specificity.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    {name:<20} selectivity = {score:.3f}")

# ── 4. Export formats ─────────────────────────────────────────────────────────
print("\n── Export: PROBE in multiple formats ────────────────────────")
from strider.export.formats import to_vienna, to_fasta, to_ct
from strider.structure.mfe import fold_mfe

struct, energy, _ = fold_mfe(PROBE, celsius=37.0)
print("  Vienna format:")
print("  " + to_vienna(PROBE, struct, name="Probe"))
print("  FASTA format:")
print("  " + to_fasta(PROBE, name="Probe",
                       description="Strand displacement probe, generic circuit"))
print("  CT (first 3 lines):")
ct = to_ct(PROBE, struct, name="Probe", energy=energy)
for line in ct.split("\n")[:3]:
    print(f"  {line}")

# ── 5. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("strider — Leakage & Off-target Screening", fontsize=13)

# Leakage ΔΔG histogram
ax = axes[0]
if report.reactions:
    ddgs = [r.ddg for r in report.reactions]
    ax.hist(ddgs, bins=15, color="#F58518", edgecolor="white")
    ax.axvline(-5.0, color="#E45756", linestyle="--", label="Dangerous threshold")
    ax.axvline(-3.0, color="#FF9800", linestyle="--", label="Filter threshold")
    ax.set_xlabel("ΔΔG (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Spurious pathway ΔΔG distribution")
    ax.legend(framealpha=0.85)
    ax.grid(True, alpha=0.25)
else:
    ax.text(0.5, 0.5, "No leakage pathways\nfound above threshold",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Spurious pathway ΔΔG distribution")

# Off-target ΔΔG comparison
ax = axes[1]
hits = screening_report.hits
if hits:
    names  = [h.name for h in hits]
    ddgs   = [h.ddg  for h in hits]
    colors = ["#E45756" if n == "exact_target" else "#4C78A8" for n in names]
    ax.barh(names, ddgs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔΔG (kcal/mol)")
    ax.set_title("PROBE off-target binding ΔΔG\n(red = exact target)")
    ax.grid(True, alpha=0.25, axis="x")
else:
    ax.text(0.5, 0.5, "No hits above threshold", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)

plt.tight_layout()
fig.savefig(_here / "leakage_screening.png", dpi=150, bbox_inches="tight")
print("\nSaved: leakage_screening.png")
print("\nDone.")
