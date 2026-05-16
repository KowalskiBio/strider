"""
Example 03: TMSD kinetics — toehold-mediated strand displacement

Covers:
- Empirical toehold kf from Zhang & Winfree 2009
- Arrhenius temperature correction
- Detailed balance: kr from ΔΔG and kf
- TMSDKineticModel: rate sets for a generic 3-strand displacement circuit
- Leakage rate modelling via Boltzmann suppression
- Plots: kf vs toehold length, temperature dependence, leakage vs stem stability

Circuit: Signal opens Gate hairpin → Reporter binds released tail; Signal recycled.
"""

import math
import pathlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np
from strider import ThermoEngine
from strider.kinetics.tmsd import (
    toehold_kf, leakage_kf, rates_from_ddg, TMSDKineticModel,
)
from strider.kinetics.arrhenius import arrhenius, detailed_balance_kr, k_eq_from_ddg

_here = pathlib.Path(__file__).parent

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
# For a favorable displacement step (ΔΔG < 0), kr << kf.
# Here: generic 7-nt toehold step at ΔΔG = -9.0 kcal/mol (illustrative).
print("\n── Detailed balance for a generic displacement step ─────────")
ddg_disp = -9.0   # kcal/mol, typical favorable strand displacement
kf_disp  = toehold_kf(7, material="dna", celsius=37.0)
kf_val, kr_val = rates_from_ddg(ddg_disp, kf_disp, celsius=37.0)
k_eq = k_eq_from_ddg(ddg_disp, celsius=37.0)
print(f"  ΔΔG(disp)  = {ddg_disp:+.1f} kcal/mol")
print(f"  kf(disp)   = {kf_val:.2e} M⁻¹s⁻¹  (7-nt toehold, 37°C)")
print(f"  kr(disp)   = {kr_val:.2e} s⁻¹      (from detailed balance)")
print(f"  Keq(disp)  = {k_eq:.2e} M⁻¹       (= kf/kr)")
print(f"  Verify:  kr = kf * exp(ΔΔG/RT) = "
      f"{kf_val * math.exp(ddg_disp / (1.987e-3 * 310.15)):.2e}")

# ── 4. Leakage rate: Boltzmann-suppressed by hairpin stability ────────────────
# Spontaneous hairpin opening (breathing) is exponentially suppressed by
# the stem stability.
print("\n── Leakage rate vs hairpin stem stability ───────────────────")
print(f"  {'|ΔG_stem| (kcal/mol)':<24} {'k_leak (M⁻¹s⁻¹)'}")
for g_stem in [0, 2, 4, 6, 8.0, 8, 10, 12]:
    k_leak = leakage_kf(g_stem, kf_max=1e6, celsius=37.0)
    print(f"  {g_stem:<24.1f} {k_leak:.2e}")

# ── 5. Generic 3-strand displacement: TMSDKineticModel ───────────────────────
# Signal strand opens Gate hairpin (7-nt toehold); Reporter binds the freed tail.
# Signal is released in step 2 — a catalytic-like recycling motif.
print("\n── TMSDKineticModel: generic 3-strand displacement ─────────")
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)
kinetic_model = TMSDKineticModel(engine, celsius=37.0)

SIGNAL   = "GCATCGATCGATCGATCGCA"         # 20 nt trigger strand
GATE     = "TGCATCGATCGATCGATCGCATGCAT"   # 26 nt hairpin gate
REPORTER = "ATGCATGCATGCATGCATGC"         # 20 nt reporter strand

reactions = [
    "Signal + Gate <-> Signal_Gate",
    "Signal_Gate + Reporter <-> Gate_Rep + Signal",
    "Gate + Reporter <-> Gate_Rep",
]
sequences  = {"Signal": SIGNAL, "Gate": GATE, "Reporter": REPORTER}
toehold_map = {
    "Signal + Gate <-> Signal_Gate": 7,
    "Signal_Gate + Reporter <-> Gate_Rep + Signal": 9,
    "Gate + Reporter <-> Gate_Rep": 0,
}
rates = kinetic_model.circuit_rates(reactions, sequences, toehold_map)
print(f"  {'Reaction':<50} {'Rate'}")
for key, val in rates.items():
    print(f"  {key:<50} {val:.2e}")

# ── 6. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("strider — TMSD Kinetics", fontsize=13)

# kf vs toehold length at 25°C and 37°C
toeholds = list(range(0, 13))
kfs_37   = [toehold_kf(n, celsius=37.0) for n in toeholds]
kfs_25   = [toehold_kf(n, celsius=25.0) for n in toeholds]
ax = axes[0]
ax.semilogy(toeholds, kfs_25, "o-", label="25°C", color="#4C78A8")
ax.semilogy(toeholds, kfs_37, "s-", label="37°C", color="#E45756")
ax.axvline(7, color="gray", linestyle="--", alpha=0.5, label="7-nt (example)")
ax.set_xlabel("Toehold length (nt)")
ax.set_ylabel("$k_f$ (M$^{-1}$s$^{-1}$)")
ax.set_title("$k_f$ vs toehold length")
ax.legend(framealpha=0.85)
ax.grid(True, alpha=0.25)

# kf vs temperature for 7-nt toehold
temps = np.linspace(10, 65, 100)
kfs_t = [toehold_kf(7, celsius=t) for t in temps]
ax = axes[1]
ax.semilogy(temps, kfs_t, "-", color="#54A24B", linewidth=2)
ax.axvline(37, color="gray", linestyle="--", alpha=0.5, label="37°C (body)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("$k_f$ (M$^{-1}$s$^{-1}$)")
ax.set_title("Temperature dependence (7-nt toehold)")
ax.legend(framealpha=0.85)
ax.grid(True, alpha=0.25)

# Leakage rate vs stem stability
g_stems = np.linspace(0, 15, 100)
k_leaks = [leakage_kf(g, kf_max=1e6, celsius=37.0) for g in g_stems]
ax = axes[2]
ax.semilogy(g_stems, k_leaks, "-", color="#B279A2", linewidth=2)
ax.axvline(8.0, color="gray", linestyle="--", alpha=0.5, label="|ΔG| = 8 kcal/mol")
ax.axhline(1e3, color="#FF9800", linestyle=":", alpha=0.7, label="10$^3$ M$^{-1}$s$^{-1}$ threshold")
ax.set_xlabel("|ΔG_stem| (kcal/mol)")
ax.set_ylabel("$k_{leak}$ (M$^{-1}$s$^{-1}$)")
ax.set_title("Leakage rate vs stem stability")
ax.legend(framealpha=0.85)
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(_here / "tmsd_kinetics.png", dpi=150, bbox_inches="tight")
print("\nSaved: tmsd_kinetics.png")
print("\nDone.")
