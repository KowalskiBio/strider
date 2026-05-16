"""
Example 01: DNA thermodynamics basics

Covers:
- ThermoEngine setup and backend selection
- Single-strand ensemble free energy (pfunc)
- Bimolecular duplex ΔG (duplex_dg / ddg)
- Melting temperature
- Salt correction comparison
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

# ── 1. Single-strand ensemble free energy ────────────────────────────────────
# miR-21 is a 22-nt microRNA; at 37°C it has minimal structure.
# H1 is a 38-nt hairpin designed to open specifically on miR-21 binding.
MIR21 = "TAGCTTATCAGACTGATGTTGA"
H1    = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"

print("── Single-strand ensemble free energies ─────────────────────")
for name, seq in [("miR-21 (22 nt)", MIR21), ("H1 hairpin (38 nt)", H1)]:
    result = engine.pfunc(seq)
    print(f"  {name:<22}  ΔG = {result.free_energy:+.2f} kcal/mol  "
          f"(Q = {result.partition_function:.2e})")

# ── 2. MFE structure ─────────────────────────────────────────────────────────
print("\n── MFE structures ───────────────────────────────────────────")
for name, seq in [("miR-21", MIR21), ("H1", H1)]:
    mfe = engine.mfe(seq)
    print(f"  {name}: {mfe.structure}  ({mfe.energy:+.2f} kcal/mol)")

# ── 3. Bimolecular binding ΔΔG ──────────────────────────────────────────────
# ΔΔG = G(complex) − G(miR21) − G(H1)
# The nick-aware McCaskill DP correctly accounts for each strand's
# intramolecular structure competing with intermolecular binding.
print("\n── Bimolecular binding (nick-aware partition function) ───────")
ddg_r1 = engine.ddg([MIR21, H1], [[MIR21, H1]])
print(f"  miR21 + H1 → miR21·H1   ΔΔG = {ddg_r1:+.2f} kcal/mol")

H2 = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
CP = "AAAAA"
ddg_spont = engine.ddg([H1, H2], [[H1, H2]])
ddg_r3    = engine.ddg([[H1, H2], CP], [[H1, H2, CP]])
print(f"  H1 + H2   → H1·H2        ΔΔG = {ddg_spont:+.2f} kcal/mol  (spontaneous leakage)")
print(f"  H1·H2 + CP → H1·H2·CP    ΔΔG = {ddg_r3:+.2f} kcal/mol  (detection)")

# ── 4. Melting temperature ───────────────────────────────────────────────────
print("\n── Melting temperatures ─────────────────────────────────────")
for name, seq in [("miR-21 (vs complement)", MIR21), ("H1 toehold (6 nt)", H1[:6])]:
    tm = engine.melting_temperature(seq, strand_conc_M=250e-9)
    print(f"  {name:<28}  Tm = {tm:.1f} °C")

# ── 5. Salt dependence of melting temperature ────────────────────────────────
# Lower salt reduces electrostatic shielding → duplex melts at lower temperature.
# The Owczarzy 2004 correction shifts Tm by several degrees across the Na+ range.
print("\n── Salt dependence: Tm(H1) and Tm(miR-21) at various [Na+] ─")
from strider.thermo.nn_dna import melting_temperature
for na_mM in [10, 50, 137, 500, 1000]:
    tm_h1  = melting_temperature(H1,    strand_conc_M=250e-9, sodium_M=na_mM/1000)
    tm_mir = melting_temperature(MIR21, strand_conc_M=250e-9, sodium_M=na_mM/1000)
    print(f"  [{na_mM:4d} mM Na+]  Tm(H1) = {tm_h1:.1f}°C   Tm(miR-21) = {tm_mir:.1f}°C")

# ── 6. Toehold accessibility ─────────────────────────────────────────────────
# How often is the toehold (first 6 nt of H1) unpaired in the ensemble?
print("\n── Toehold accessibility (fraction unpaired) ────────────────")
acc = engine.toehold_accessibility(H1, toehold_positions=list(range(6)))
print(f"  H1 toehold [0:6]  accessibility = {acc:.3f}  ({acc*100:.1f}%)")

# A 30-nt linear strand (no hairpin) should be nearly fully accessible
linear = "A" * 30
acc_lin = engine.toehold_accessibility(linear, toehold_positions=list(range(6)))
print(f"  Linear AAAAAA..  accessibility = {acc_lin:.3f}  ({acc_lin*100:.1f}%)")

print("\nDone.")
