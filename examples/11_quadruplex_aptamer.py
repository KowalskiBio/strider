"""
Example 11: K⁺-gated G-quadruplex aptamer → electrochemical read-out

A G-quadruplex is something NUPACK structurally cannot represent: it hardcodes
pseudoknots off and models only Watson–Crick / wobble pairs, whereas a G4 is four
guanine tracts stacked into Hoogsteen tetrads around a column of monovalent
cations.  strider adds it as a *competing macrostate* on top of the McCaskill
ensemble, with the K⁺/Na⁺ dependence that makes it a sensor.

Here the recognition element is a tethered G4 aptamer whose folding is gated by
[K⁺].  When folded, it brings a redox label into electron-transfer range of the
electrode; the folded fraction therefore scales the faradaic signal.  We chain:

    sequence + [K⁺]  ──strider──▶  G4 folded fraction
                     ──surface──▶  captured redox signal → current

— the full sequences → thermodynamics → transducer path, end to end.
"""

import numpy as np

from strider import fold_quadruplex, quadruplex_ensemble
from strider.surface import SurfaceParams

# Human-telomere-style G4 aptamer (3 tetrads, 3-nt loops), Tm ≈ 57 °C in K⁺.
APTAMER = "AGGGTTAGGGTTAGGGTTAGGG"

# ── 1. Folding is K⁺-gated (and K⁺ ≫ Na⁺) ─────────────────────────────────────
print("Aptamer:", APTAMER)
print("\n[cation] dependence of G4 folding (37 °C):")
print(f"  {'condition':14s} {'dG37':>7s} {'Tm(°C)':>7s} {'folded':>7s} {'p(G4)':>7s}")
for label, k, na in [
    ("1 mM K+", 1e-3, 0.0),
    ("10 mM K+", 1e-2, 0.0),
    ("100 mM K+", 1e-1, 0.0),
    ("100 mM Na+", 0.0, 1e-1),
]:
    f = fold_quadruplex(APTAMER, celsius=37, potassium=k, sodium=na)
    e = quadruplex_ensemble(APTAMER, celsius=37, potassium=k, sodium=na)
    print(f"  {label:14s} {f.dG:7.2f} {f.tm_celsius:7.1f} "
          f"{f.folded_fraction:7.3f} {e.p_g4:7.3f}")

# ── 2. Couple the folded fraction to the electrode read-out ───────────────────
# Unlike a sandwich assay, the reporter is *already* tethered to the electrode —
# there is no diffusion-limited capture step.  What [K⁺] controls is the fraction
# of probes in the signal-ON (folded) conformation, which sets how many redox
# labels are in electron-transfer range.  So we drive the surface label model
# directly with N_on = (tethered probes) × (folded fraction).
p = SurfaceParams()
N_PROBES = 1.0e4                               # sparse redox-reporter monolayer
q_per_event = p.label.signal_per_event()       # faradaic charge per ON label
i_floor_nA = p.readout.current_floor_A() * 1e9

print(f"\nElectrochemical calibration vs [K⁺]  ({N_PROBES:.0e} tethered probes, "
      f"floor ≈ {i_floor_nA:.2f} nA):")
print(f"  {'[K+] (M)':>10s} {'folded':>7s} {'I_p (nA)':>9s}")
for k in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    frac = fold_quadruplex(APTAMER, celsius=37, potassium=k).folded_fraction
    q = N_PROBES * frac * q_per_event          # faradaic charge from ON probes
    i_p = q / p.readout.dpv_pulse_s
    print(f"  {k:10.0e} {frac:7.3f} {i_p * 1e9:9.3f}")

print("\nThe sensor turns on with K⁺ — the G4 folding equilibrium *is* the "
      "transduction mechanism, and it lives entirely outside what NUPACK can model.")
