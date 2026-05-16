"""
Example 03: TMSD kinetics — toehold-mediated strand displacement

Covers:
- Empirical toehold kf from Zhang & Winfree 2009
- Arrhenius temperature correction
- Detailed balance: kr from ΔΔG and kf
- TMSDKineticModel: reaction-level rate sets
- Leakage rate modelling via Boltzmann suppression
- Plots: kf vs toehold length, temperature dependence
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from strider import ThermoEngine
from strider.kinetics.tmsd import (
    toehold_kf, leakage_kf, rates_from_ddg, TMSDKineticModel,
)
from strider.kinetics.arrhenius import arrhenius, detailed_balance_kr, k_eq_from_ddg

# ── 1. Zhang & Winfree 2009 toehold kf lookup ───────────────────────────────
print("── Toehold kf (25°C, DNA, from Zhang & Winfree 2009) ────────")
print(f"  {'Toehold (nt)':<16} {'kf (M⁻¹s⁻¹)':<16} {'log10(kf)'}")
for nt in range(0, 13):
    kf = toehold_kf(nt, material="dna", celsius=25.0)
    print(f"  {nt:<16} {kf:<16.2e} {math.log10(kf):.2f}")

# ── 2. Temperature dependence via Arrhenius ──────────────────────────────────
print("\n── kf vs temperature (8-nt toehold, Ea = 20 kcal/mol) ──────")
kf_ref = toehold_kf(8, material="dna", celsius=25.0)
for celsius in [4, 20, 25, 37, 45, 55]:
    kf = toehold_kf(8, material="dna", celsius=celsius)
    print(f"  {celsius:2d}°C  kf = {kf:.2e} M⁻¹s⁻¹  ({kf/kf_ref:.2f}× rel. 25°C)")

# ── 3. Detailed balance: kr from kf and ΔΔG ──────────────────────────────────
# For a favorable reaction (ΔΔG < 0), kr << kf.
# For the CHA initiation step R1, ΔΔG ≈ -11.5 kcal/mol at 37°C.
print("\n── Detailed balance for R1 (miR21 + H1 → miR21·H1) ─────────")
ddg_R1 = -11.5   # kcal/mol (from nick-aware DP or NUPACK)
kf_R1  = toehold_kf(6, material="dna", celsius=37.0)
kf_val, kr_val = rates_from_ddg(ddg_R1, kf_R1, celsius=37.0)
k_eq = k_eq_from_ddg(ddg_R1, celsius=37.0)
print(f"  ΔΔG(R1)  = {ddg_R1:+.1f} kcal/mol")
print(f"  kf(R1)   = {kf_val:.2e} M⁻¹s⁻¹  (6-nt toehold, 37°C)")
print(f"  kr(R1)   = {kr_val:.2e} s⁻¹      (from detailed balance)")
print(f"  Keq(R1)  = {k_eq:.2e} M⁻¹       (= kf/kr)")
print(f"  Verify:  kr = kf * exp(ΔΔG/RT) = {kf_val * math.exp(ddg_R1 / (1.987e-3 * 310.15)):.2e}")

# ── 4. Leakage rate: Boltzmann-suppressed by hairpin stability ────────────────
# Spontaneous hairpin opening (breathing) is exponentially suppressed by
# the stem stability. G(H1) ≈ -7.2 kcal/mol → k_leak ≈ kf_max * exp(-7.2/RT)
print("\n── Leakage rate vs hairpin stem stability ───────────────────")
print(f"  {'|ΔG_stem| (kcal/mol)':<24} {'k_leak (M⁻¹s⁻¹)'}")
for g_stem in [0, 2, 4, 6, 7.2, 8, 10, 12]:
    k_leak = leakage_kf(g_stem, kf_max=1e6, celsius=37.0)
    print(f"  {g_stem:<24.1f} {k_leak:.2e}")

# ── 5. Full CHA kinetic model via TMSDKineticModel ───────────────────────────
print("\n── TMSDKineticModel: full CHA rate set ─────────────────────")
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)
kinetic_model = TMSDKineticModel(engine, celsius=37.0)

H1    = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2    = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
MIR21 = "TAGCTTATCAGACTGATGTTGA"
CP    = "AAAAA"

reactions = [
    "miR21 + H1 <-> miR21_H1",
    "miR21_H1 + H2 <-> H1H2 + miR21",
    "H1H2 + CP <-> H1H2_CP",
    "H1 + H2 <-> H1H2",
]
sequences = {"miR21": MIR21, "H1": H1, "H2": H2, "CP": CP}
toehold_map = {
    "miR21 + H1 <-> miR21_H1":     6,
    "miR21_H1 + H2 <-> H1H2 + miR21": 11,
    "H1H2 + CP <-> H1H2_CP":       5,
    "H1 + H2 <-> H1H2":            0,
}
rates = kinetic_model.circuit_rates(reactions, sequences, toehold_map)
print(f"  {'Reaction':<42} {'Rate'}")
for key, val in rates.items():
    print(f"  {key:<42} {val:.2e}")

# ── 6. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("strider — TMSD Kinetics", fontsize=13, fontweight="bold")

# kf vs toehold length at 37°C
toeholds = list(range(0, 13))
kfs_37   = [toehold_kf(n, celsius=37.0) for n in toeholds]
kfs_25   = [toehold_kf(n, celsius=25.0) for n in toeholds]
ax = axes[0]
ax.semilogy(toeholds, kfs_25, "o-", label="25°C", color="#4466dd")
ax.semilogy(toeholds, kfs_37, "s-", label="37°C", color="#dd4444")
ax.axvline(6, color="gray", linestyle="--", alpha=0.5, label="D1 toehold")
ax.set_xlabel("Toehold length (nt)")
ax.set_ylabel("kf (M⁻¹s⁻¹)")
ax.set_title("kf vs toehold length")
ax.legend()
ax.grid(alpha=0.3)

# kf vs temperature for 6-nt toehold
temps = np.linspace(10, 65, 100)
kfs_t = [toehold_kf(6, celsius=t) for t in temps]
ax = axes[1]
ax.semilogy(temps, kfs_t, "-", color="#44aa44", linewidth=2)
ax.axvline(37, color="gray", linestyle="--", alpha=0.5, label="37°C (body)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("kf (M⁻¹s⁻¹)")
ax.set_title("Temperature dependence (6-nt toehold)")
ax.legend()
ax.grid(alpha=0.3)

# Leakage rate vs stem stability
g_stems = np.linspace(0, 15, 100)
k_leaks = [leakage_kf(g, kf_max=1e6, celsius=37.0) for g in g_stems]
ax = axes[2]
ax.semilogy(g_stems, k_leaks, "-", color="#aa4488", linewidth=2)
ax.axvline(7.2, color="gray", linestyle="--", alpha=0.5, label="|G(H1)| ≈ 7.2")
ax.axhline(1e3, color="#ff8800", linestyle=":", alpha=0.7, label="1000 M⁻¹s⁻¹ threshold")
ax.set_xlabel("|ΔG_stem| (kcal/mol)")
ax.set_ylabel("k_leak (M⁻¹s⁻¹)")
ax.set_title("Leakage rate vs stem stability")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tmsd_kinetics.png", dpi=120, bbox_inches="tight")
print("\nSaved: tmsd_kinetics.png")
print("\nDone.")
