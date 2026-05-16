"""
Example 06: Parameter sweep with caching — toehold, temperature, concentration

Covers:
- ParameterSweep: toehold_sweep, temperature_sweep, grid_sweep
- DiskCache: persistent sqlite3 memoisation (skip recomputation on re-runs)
- SweepResult: to_dataframe, optimum, plot
- 2D grid sweep: toehold length × temperature
- Dose-response curve: signal fraction vs [Target] concentration
"""

import math
import os
import time
import pathlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
from strider.thermo.engine import ThermoEngine
from strider.sweep.cache import DiskCache
from strider.sweep.batch import ParameterSweep
from strider.kinetics.tmsd import toehold_kf

_here = pathlib.Path(__file__).parent

# ── Setup ────────────────────────────────────────────────────────────────────
engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

# Generic probe/target system (no specific biological target)
TARGET = "GCATCGATCGATCGATCGCA"    # 20 nt analyte strand
PROBE  = "CGCGATCGATCGATCGATCGCG"  # 22 nt probe hairpin
BEACON = "TGCATCGATCGATCGATCGCAT"  # 22 nt beacon strand
LINKER = "AAAAA"                    # 5-nt poly-A linker

# ── 1. DiskCache: persistent memoisation ────────────────────────────────────
print("── DiskCache persistent memoisation ─────────────────────────")
CACHE_PATH = "/tmp/strider_sweep_demo.db"
cache = DiskCache(CACHE_PATH)
print(f"  Cache entries (start): {cache.stats()['entries']}")

# First call — cache miss
t0 = time.perf_counter()
key = DiskCache.make_key("pfunc", "dna", 37.0, PROBE)
if cache.get(key) is None:
    val = engine.pfunc(PROBE)
    cache.set(key, val)
    print(f"  Cache MISS — computed in {(time.perf_counter()-t0)*1000:.1f} ms  "
          f"(G = {val.free_energy:.2f} kcal/mol)")
else:
    print(f"  Cache HIT  — instant (G = {cache.get(key).free_energy:.2f})")

# Second call — cache hit
t0 = time.perf_counter()
hit = cache.get(key)
print(f"  Cache HIT  — fetched in {(time.perf_counter()-t0)*1000:.2f} ms")
print(f"  Cache stats: {cache.stats()}")

# ── 2. Toehold length sweep ──────────────────────────────────────────────────
print("\n── Toehold sweep: kf and ΔΔG vs toehold length ─────────────")

sweep = ParameterSweep(engine, cache=None, n_workers=1)
toehold_result = sweep.toehold_sweep(
    hairpin_seq=PROBE,
    toehold_lengths=list(range(3, 13)),
    target_strand=TARGET,
)

toehold_lengths = list(toehold_result.axes["toehold_length"])
kf_values       = list(toehold_result.values)
ddg_values      = []
for nt in toehold_lengths:
    toehold_probe  = PROBE[:nt]
    toehold_target = TARGET[-nt:]
    ddg = engine.ddg([toehold_probe, toehold_target], [[toehold_probe, toehold_target]])
    ddg_values.append(ddg)

print(f"  {'Toehold (nt)':<14} {'kf (M⁻¹s⁻¹)':<16} {'ΔΔG (kcal/mol)'}")
for nt, kf, ddg in zip(toehold_lengths, kf_values, ddg_values):
    print(f"  {nt:<14} {kf:<16.2e} {ddg:.2f}")

best_nt = toehold_lengths[int(np.argmax(kf_values))]
print(f"\n  Fastest toehold: {best_nt} nt  (kf = {max(kf_values):.2e} M⁻¹s⁻¹)")

# ── 3. Temperature sweep ──────────────────────────────────────────────────────
print("\n── Temperature sweep: ΔG(Probe), ΔG(Beacon) vs temperature ─")

temps = list(range(20, 65, 5))
temp_result = sweep.temperature_sweep(
    sequences={"Probe": PROBE, "Beacon": BEACON},
    temperatures=temps,
)
strand_names = temp_result.metadata["strand_names"]

print(f"  {'T (°C)':<10} {'ΔG(Probe)':<14} {'ΔG(Beacon)':<14} {'ΔΔG(Probe+Beacon)'}")
for j, T in enumerate(temps):
    g_probe  = temp_result.values[0, j]
    g_beacon = temp_result.values[1, j]
    print(f"  {T:<10} {g_probe:<14.2f} {g_beacon:<14.2f} {g_probe+g_beacon:.2f}")

# ── 4. 2D grid sweep: kf(toehold, temperature) ──────────────────────────────
print("\n── 2D grid: kf(toehold length × temperature) ───────────────")

def kf_at(params):
    return toehold_kf(int(params["toehold"]), material="dna", celsius=params["celsius"])

grid_result = sweep.grid_sweep(
    axes={
        "toehold": list(range(4, 11)),
        "celsius": [25, 30, 37, 45, 55],
    },
    fn=kf_at,
)

inv_grid = sweep.grid_sweep(
    axes={"toehold": list(range(4, 11)), "celsius": [25, 30, 37, 45, 55]},
    fn=lambda p: 1.0 / kf_at(p),
)
opt_params = inv_grid.optimum()
opt_kf = kf_at(opt_params)
print(f"  Grid shape: {grid_result.values.shape}  "
      f"(toehold 4–10 nt × temp 25–55°C)")
print(f"  Maximum kf = {opt_kf:.2e} M⁻¹s⁻¹  at  "
      f"toehold={int(opt_params['toehold'])} nt, T={opt_params['celsius']}°C")

# ── 5. Dose-response: signal fraction vs [Target] ─────────────────────────────
print("\n── Dose-response: predicted signal vs [Target] ──────────────")
# First-order approximation: fraction opened ≈ 1 − exp(−kf·[Target]·t)
# Valid when [Target] << [Probe]. For a proper ODE use the mantis bridge (example 07).
kf_probe       = toehold_kf(7, material="dna", celsius=37.0)
t_incubation   = 3600  # seconds (1 h)
probe_conc_nM  = 100.0

print(f"  kf(7-nt toehold) = {kf_probe:.2e} M⁻¹s⁻¹")
print(f"  Incubation = {t_incubation//60} min,  [Probe] = {probe_conc_nM:.0f} nM")
print(f"\n  {'[Target] (nM)':<18} {'Signal fraction'}")
target_nM_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
signal_fracs = []
for c_nM in target_nM_vals:
    frac = min(1.0 - math.exp(-kf_probe * c_nM * 1e-9 * t_incubation), 1.0)
    signal_fracs.append(frac)
    print(f"  {c_nM:<18.2f} {frac:.4f}  ({frac*100:.1f}%)")

# ── 6. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("strider — Parameter Sweep & Dose-Response", fontsize=13)

# Toehold sweep: ΔΔG and kf (dual-axis)
ax = axes[0][0]
color_kf, color_ddg = "#E45756", "#4C78A8"
ax.semilogy(toehold_lengths, kf_values, "s-", color=color_kf, label="kf (37°C)")
ax.set_ylabel("$k_f$ (M$^{-1}$s$^{-1}$)", color=color_kf)
ax.tick_params(axis="y", labelcolor=color_kf)
ax_ddg = ax.twinx()
ax_ddg.plot(toehold_lengths, ddg_values, "o--", color=color_ddg, label="ΔΔG")
ax_ddg.set_ylabel("ΔΔG (kcal/mol)", color=color_ddg)
ax_ddg.tick_params(axis="y", labelcolor=color_ddg)
ax.axvline(7, color="gray", linestyle=":", alpha=0.6, label="7-nt (example)")
ax.set_xlabel("Toehold length (nt)")
ax.set_title("Toehold sweep")
lines = [plt.Line2D([0], [0], color=color_kf, marker="s"),
         plt.Line2D([0], [0], color=color_ddg, linestyle="--", marker="o")]
ax.legend(lines, ["$k_f$", "ΔΔG"], loc="lower right", framealpha=0.85)
ax.grid(True, alpha=0.25)

# Temperature sweep: G(Probe) and G(Beacon)
ax = axes[0][1]
ax.plot(temps, temp_result.values[0], "o-", color="#4C78A8", label="G(Probe)")
ax.plot(temps, temp_result.values[1], "s-", color="#B279A2", label="G(Beacon)")
ax.axvline(37, color="gray", linestyle=":", alpha=0.6, label="37°C (body)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("ΔG (kcal/mol)")
ax.set_title("Temperature sweep: probe stability")
ax.legend(framealpha=0.85)
ax.grid(True, alpha=0.25)

# 2D heatmap: log10(kf)
ax = axes[1][0]
toehold_grid = [int(x) for x in grid_result.axes["toehold"]]
celsius_grid = list(grid_result.axes["celsius"])
im = ax.imshow(
    np.log10(grid_result.values).T,
    aspect="auto",
    origin="lower",
    cmap="plasma",
    extent=[toehold_grid[0] - 0.5, toehold_grid[-1] + 0.5,
            celsius_grid[0] - 2.5,  celsius_grid[-1] + 2.5],
)
ax.set_xlabel("Toehold length (nt)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("log$_{10}$($k_f$) heat map")
plt.colorbar(im, ax=ax, label="log$_{10}$($k_f$ [M$^{-1}$s$^{-1}$])")

# Dose-response
ax = axes[1][1]
ax.semilogx([c * 1e-9 for c in target_nM_vals], signal_fracs,
            "o-", color="#54A24B", linewidth=2, markersize=7)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% signal")
ax.set_xlabel("[Target] (M)")
ax.set_ylabel("Predicted signal fraction")
ax.set_title(f"Dose-response (t = {t_incubation//60} min, 7-nt toehold)")
ax.legend(framealpha=0.85)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(_here / "parameter_sweep.png", dpi=150, bbox_inches="tight")
print("\nSaved: parameter_sweep.png")

if os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)

print("\nDone.")
