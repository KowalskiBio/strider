"""
Example 01: DNA thermodynamics basics

Covers:
- ThermoEngine setup and backend selection
- Single-strand ensemble free energy (pfunc)
- Bimolecular duplex ΔG (ddg)
- Melting temperature
- Salt correction comparison
- Stem-loop probe accessibility
"""

from strider import ThermoEngine

# ── Engine at physiological conditions ──────────────────────────────────────
engine = ThermoEngine(
    material="dna",
    celsius=37.0,
    sodium=0.137,    # 137 mM NaCl (physiological)
    magnesium=0.01,  # 10 mM MgCl2
)
print(f"Backend: {engine.backend_name}")
print(f"Engine:  {engine}\n")

# Generic probe pair and stem-loop probe (no specific biological target)
STRAND_A = "GCATGCATGCATGCATGCAT"   # 20 nt, 50% GC, linear
STRAND_B = "ATGCATGCATGCATGCATGC"   # 20 nt, reverse complement of STRAND_A
PROBE_SL = "CGCGTTTTTTTTTTTTCGCG"   # 20 nt, 4-nt GC stem + 12-T loop (hairpin probe)

# ── 1. Single-strand ensemble free energies ──────────────────────────────────
# STRAND_A and STRAND_B are linear — minimal self-structure expected.
# PROBE_SL is a hairpin — substantial negative ΔG from stem formation.
print("── Single-strand ensemble free energies ─────────────────────")
for name, seq in [("Strand A (20 nt)", STRAND_A),
                  ("Strand B (20 nt)", STRAND_B),
                  ("Stem-loop probe (20 nt)", PROBE_SL)]:
    result = engine.pfunc(seq)
    print(f"  {name:<26}  ΔG = {result.free_energy:+.2f} kcal/mol  "
          f"(Q = {result.partition_function:.2e})")

# ── 2. MFE structure ─────────────────────────────────────────────────────────
print("\n── MFE structures ───────────────────────────────────────────")
for name, seq in [("Strand A", STRAND_A), ("Probe SL", PROBE_SL)]:
    mfe = engine.mfe(seq)
    print(f"  {name}: {mfe.structure}  ({mfe.energy:+.2f} kcal/mol)")

# ── 3. Bimolecular binding ΔΔG ──────────────────────────────────────────────
# ΔΔG = G(complex) − G(A) − G(B)
# A and B are reverse complements — strong duplex (large negative ΔΔG) expected.
# Probe SL vs Strand A: little complementarity — near-zero ΔΔG expected.
print("\n── Bimolecular binding (nick-aware partition function) ───────")
ddg_ab = engine.ddg([STRAND_A, STRAND_B], [[STRAND_A, STRAND_B]])
print(f"  A + B → A·B  (full duplex)     ΔΔG = {ddg_ab:+.2f} kcal/mol")

ddg_probe_a = engine.ddg([PROBE_SL, STRAND_A], [[PROBE_SL, STRAND_A]])
print(f"  Probe + A → Probe·A  (off-tgt) ΔΔG = {ddg_probe_a:+.2f} kcal/mol  ← expect weak")

# ── 4. Melting temperature ───────────────────────────────────────────────────
print("\n── Melting temperatures ─────────────────────────────────────")
for name, seq in [("A vs B (full duplex, 20 nt)", STRAND_A),
                  ("Probe SL stem (4 nt)", PROBE_SL[:4])]:
    tm = engine.melting_temperature(seq, strand_conc_M=250e-9)
    print(f"  {name:<32}  Tm = {tm:.1f} °C")

# ── 5. Salt dependence of melting temperature ────────────────────────────────
# Lower salt reduces electrostatic shielding → duplex melts at lower temperature.
# The Owczarzy 2004 correction shifts Tm by several degrees across the Na+ range.
print("\n── Salt dependence: Tm at various [Na+] ─────────────────────")
from strider.thermo.nn_dna import melting_temperature
for na_mM in [10, 50, 137, 500, 1000]:
    tm_a  = melting_temperature(STRAND_A, strand_conc_M=250e-9, sodium_M=na_mM/1000)
    tm_sl = melting_temperature(PROBE_SL, strand_conc_M=250e-9, sodium_M=na_mM/1000)
    print(f"  [{na_mM:4d} mM Na+]  Tm(Strand A) = {tm_a:.1f}°C   Tm(Probe SL) = {tm_sl:.1f}°C")

# ── 6. Stem-loop probe accessibility ─────────────────────────────────────────
# How often is the probe's stem (first 4 nt) vs loop (middle 12 nt) unpaired?
# Stem should be mostly paired (low accessibility); loop mostly open (high).
print("\n── Probe SL accessibility (fraction unpaired per region) ────")
engine_native = ThermoEngine(material="dna", celsius=37.0, sodium=0.137,
                              magnesium=0.01, backend="native")
acc_stem = engine_native.toehold_accessibility(PROBE_SL, toehold_positions=list(range(4)))
acc_loop = engine_native.toehold_accessibility(PROBE_SL, toehold_positions=list(range(4, 16)))
print(f"  Probe SL stem [0:4]   accessibility = {acc_stem:.3f}  ({acc_stem*100:.1f}%)  ← stem, expect low")
print(f"  Probe SL loop [4:16]  accessibility = {acc_loop:.3f}  ({acc_loop*100:.1f}%)  ← loop, expect high")

linear = "T" * 20
acc_lin = engine_native.toehold_accessibility(linear, toehold_positions=list(range(4)))
print(f"  Linear reference      accessibility = {acc_lin:.3f}  ({acc_lin*100:.1f}%)  ← no structure")

print("\nDone.")
