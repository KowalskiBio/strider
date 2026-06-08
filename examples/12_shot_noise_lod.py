"""
Example 12: Shot-noise-limited detection — why the real LOD is a counting limit

The deterministic surface transducer (example 11, `SurfaceModel`) integrates a
diffusion-limited flux against a bulk concentration that *never depletes*.  At
the fM–aM concentrations where a limit of detection actually lives, that is a
fantasy: the sample aliquot holds only a handful of molecules, so capture is a
counting process and its Poisson shot noise — not the amplifier — sets the LOD.

`StochasticSurfaceModel` adds that physics: it caps the captured mean at the
molecule budget, applies the Currie (1968) detection-limit framework, and can
drive the capture as a mantis Gillespie SSA.  The headline is the gap between
the naive deterministic LOD and the honest shot-noise LOD.

NUPACK has no surface, no kinetics, and no stochastic mode — this is several
layers past it.
"""

import numpy as np

from strider.surface import SurfaceModel, SurfaceParams, StochasticSurfaceModel

p = SurfaceParams()
det = SurfaceModel(p)
sto = StochasticSurfaceModel(p)
V_NA = p.sample_volume_L * 6.02214076e23   # molecules per molar in the aliquot

times = np.linspace(0.0, p.incubation_s, 40)
make_trace = lambda c: (times, np.full_like(times, c))

# ── 1. The Currie counting thresholds ─────────────────────────────────────────
lv = sto.levels()
print("Currie detection thresholds (in captured-molecule counts):")
print(f"  read-out noise σ_read = {lv.sigma_read:.1f} counts")
print(f"  critical level  L_C   = {lv.critical_level:.1f} counts  (decide 'detected')")
print(f"  detection limit L_D   = {lv.detection_limit:.1f} counts  (reliably detected)")
print(f"  zero-background floor  = k² = {lv.k**2:.2f} counts  (pure counting limit)")

# ── 2. Deterministic vs shot-noise LOD ────────────────────────────────────────
triggers = np.array([1e-19, 3e-19, 1e-18, 3e-18, 1e-17, 3e-17, 1e-16, 3e-16, 1e-15])
det_lod = det.lod(make_trace, triggers)
sto_lod = sto.shot_noise_lod(make_trace, triggers)
print(f"\nDeterministic LOD (infinite reservoir) = {det_lod:.0e} M")
print(f"Shot-noise-limited LOD (counting)       = {sto_lod:.0e} M")
print(f"  → the deterministic LOD is ~{sto_lod/det_lod:.0g}× too optimistic.\n")

print(f"  {'[analyte] M':>12s} {'molecules':>10s} {'det. n_cap':>12s} {'true μ':>9s} {'P(detect)':>10s}")
for c in triggers:
    t, s = make_trace(c)
    n_total = c * V_NA
    det_n = det.transduce(t, s).n_captured           # infinite-reservoir overcount
    mu = sto.capture_mean(t, s)                       # capped at the molecule budget
    pdet = sto.detection_probability(t, s)
    print(f"  {c:12.0e} {n_total:10.0f} {det_n:12.0f} {mu:9.1f} {pdet:10.3f}")

print("\nNote the deterministic transducer 'captures' thousands of molecules at "
      "concentrations\nwhere only a handful exist — the shot-noise model caps "
      "capture at the molecule budget.")

# ── 3. mantis Gillespie SSA of the capture, near the LOD ──────────────────────
try:
    c = 5e-17
    t, s = make_trace(c)
    samp = sto.simulate_capture(t, s, n_trajectories=200, seed=0)
    print(f"\nmantis SSA capture at {c:.0e} M  (N_total ≈ {samp.n_total:.0f}, "
          f"p_capture = {samp.p_capture:.2f}):")
    print(f"  analytic Poisson mean μ = {samp.mean_signal:.1f}")
    print(f"  SSA empirical mean      = {samp.empirical_mean:.1f}")
    print(f"  SSA empirical variance  = {samp.empirical_var:.1f}   (Poisson ⇒ var ≈ μ)")
    print(f"  SSA detection rate      = {samp.detection_rate(lv):.3f}")
    print("  → the discrete SSA reproduces the Poisson shot noise of capture.")
except ImportError:
    print("\n(install mantis-delta to run the Gillespie SSA capture driver)")
