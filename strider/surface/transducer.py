"""
Surface transducer model — the stage a bulk-solution CRN does not cover.

Turns a solution-phase concentration trace C(t) (typically a mantis
``SimulationResult`` species, e.g. a captured dimer) into a predicted
electrochemical signal and a limit of detection, through two physical stages:

  A. CAPTURE (during incubation).  The analyte diffuses to an electrode-bound
     probe and docks.  With a strong, dense probe the surface is effectively
     *absorbing* and capture is **diffusion-limited**: the flux equals the
     diffusion-limited rate to a disk of radius a, using the Shoup–Szabo
     expression that spans the transient (Cottrell) and steady-state regimes:

        dN/dt = 4·D·a·f(τ)·C_number(t),     τ = 4·D·t / a²
        f(τ)  = 0.7854 + 0.8862·τ^(-1/2) + 0.2146·exp(-0.7823·τ^(-1/2))

     A finite probe-site capacity N_max then caps the count (the ULOQ knee):
        dN/dt = flux·(1 − N/N_max)  ⇒  N = N_max·(1 − exp(−N_unsat/N_max)).

  B. LABEL + READ-OUT.  Each captured event contributes a charge
     ``label.signal_per_event()`` (see :mod:`strider.surface.labels`):

        Q   = N_cap · signal_per_event           (faradaic peak charge)
        I_p ≈ Q / t_pulse                        (surface-confined ≈ one pulse)

  LOD = lowest trigger whose I_p (and Q) clears the read-out floor.

This is the surface-tethered-biophysics layer NUPACK has no equivalent of:
NUPACK assumes a well-mixed 3-D bulk and stops at static equilibrium.

All SI units unless noted.  Concentrations passed in molar (mol/L).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from strider.surface.labels import (
    LabelModel, PrussianBlueLabel, ReadoutChain, N_A,
)

if TYPE_CHECKING:  # pragma: no cover
    # mantis is an optional consumer; only needed for transduce_result()
    from mantis import SimulationResult


def shoup_szabo_f(tau: "np.ndarray | float") -> "np.ndarray | float":
    """Disk diffusion flux factor; →∞ as τ→0 (Cottrell), →0.785 as τ→∞ (steady)."""
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    inv_sqrt = tau ** -0.5
    return 0.7854 + 0.8862 * inv_sqrt + 0.2146 * np.exp(-0.7823 * inv_sqrt)


@dataclass
class SurfaceParams:
    """Geometry, transport, and read-out configuration for a surface assay."""
    # ── capture (stage A) ────────────────────────────────────────────────────
    d_species_m2_s: float = 1.0e-10    # analyte diffusion coeff (~1e-6 cm²/s)
    electrode_radius_m: float = 1.5e-3 # working-electrode footprint (3 mm⌀)
    incubation_s: float = 5400.0       # 90 min
    sample_volume_L: float = 50e-6     # 50 µL drop
    probe_density_per_m2: float = 1.0e16  # accessible capture sites
                                       #   (~1e12 /cm², MCH-passivated thiol-DNA
                                       #    SAM) → finite capacity sets the ULOQ
    # ── label + read-out (stage B) ──────────────────────────────────────────
    label: LabelModel = field(default_factory=PrussianBlueLabel)
    readout: ReadoutChain = field(default_factory=ReadoutChain)

    def max_capture_sites(self) -> float:
        """Finite number of capture sites on the electrode (sets the ULOQ)."""
        area = np.pi * self.electrode_radius_m ** 2
        return self.probe_density_per_m2 * area


@dataclass
class TransduceResult:
    n_captured: float          # molecules docked at end of incubation
    capture_fraction: float    # N_cap / N_total_in_sample
    peak_charge_C: float       # faradaic peak charge Q
    peak_current_A: float      # estimated DPV peak current I_p
    detectable: bool           # clears both read-out floors


def captured_count(times: np.ndarray, species_M: np.ndarray,
                   p: SurfaceParams) -> float:
    """Captured molecules at the end of incubation.

    First the diffusion-limited absorbing-disk flux is integrated against C(t)
    to give the *unsaturated* count N_unsat (perfect infinite sink).  Then a
    finite probe-site capacity N_max is imposed (docking a site removes it):

        N = N_max·(1 − exp(−N_unsat/N_max)).

    At low analyte N_unsat ≪ N_max ⇒ N ≈ N_unsat (linear); as it rises N → N_max
    (surface saturation = the upper limit of the working range, the ULOQ knee)."""
    D, a = p.d_species_m2_s, p.electrode_radius_m
    t = np.asarray(times, dtype=float)
    c_num = np.asarray(species_M, dtype=float) * N_A * 1000.0   # mol/L → /m³
    tau = 4.0 * D * np.maximum(t, 0.0) / a**2
    flux = 4.0 * D * a * shoup_szabo_f(tau) * c_num            # molecules/s
    n_unsat = float(np.trapezoid(flux, t))
    n_max = p.max_capture_sites()
    if n_max <= 0:
        return n_unsat
    return n_max * (1.0 - np.exp(-n_unsat / n_max))


class SurfaceModel:
    """Capture → label → read-out → detectability for a tethered-probe assay.

    The model consumes a raw ``(times, species_M)`` concentration trace, so it
    works with the output of *any* mantis simulation (or a synthetic trace), not
    just CHA.  Use :meth:`transduce_result` to pass a mantis ``SimulationResult``
    directly.
    """

    def __init__(self, params: SurfaceParams | None = None) -> None:
        self.params = params or SurfaceParams()

    # ── core: raw-array primitive ────────────────────────────────────────────
    def transduce(self, times: np.ndarray, species_M: np.ndarray) -> TransduceResult:
        p = self.params
        n_cap = captured_count(times, species_M, p)
        c_final = float(np.asarray(species_M)[-1])
        n_total = c_final * N_A * 1000.0 * (p.sample_volume_L * 1e-3)  # in m³
        cap_frac = n_cap / n_total if n_total > 0 else 0.0
        q = n_cap * p.label.signal_per_event()
        i_p = q / p.readout.dpv_pulse_s
        detect = (i_p >= p.readout.current_floor_A()) and (q >= p.readout.charge_floor_C())
        return TransduceResult(n_captured=n_cap, capture_fraction=cap_frac,
                               peak_charge_C=q, peak_current_A=i_p,
                               detectable=detect)

    # ── convenience: mantis SimulationResult ─────────────────────────────────
    def transduce_result(self, sim: "SimulationResult", species: str,
                         times: np.ndarray | None = None) -> TransduceResult:
        """Transduce a mantis ``SimulationResult`` species trace.

        ``times`` is taken from ``sim.times`` when present; pass it explicitly if
        the result object exposes the grid under a different attribute.
        """
        conc = sim.concentrations[species]
        if times is None:
            times = getattr(sim, "times", None)
            if times is None:
                times = getattr(sim, "t", None)
            if times is None:
                raise ValueError(
                    "could not infer the time grid from the SimulationResult; "
                    "pass times=... explicitly")
        return self.transduce(np.asarray(times), np.asarray(conc))

    # ── LOD over a trigger ladder ────────────────────────────────────────────
    def lod(self,
            make_trace: Callable[[float], tuple[np.ndarray, np.ndarray]],
            triggers: np.ndarray) -> float | None:
        """Lowest trigger (ascending) whose transduced signal clears the floor.

        ``make_trace(c)`` returns a ``(times, species_M)`` trace for trigger
        concentration ``c``.  When the underlying cascade is linear in the
        trigger, ``make_trace`` is just a linear scaling of one reference trace
        (cheap); otherwise it can run a fresh simulation per point.
        Returns ``None`` if no trigger in ``triggers`` is detectable.
        """
        for c in np.asarray(triggers, dtype=float):
            times, species_M = make_trace(float(c))
            if self.transduce(times, species_M).detectable:
                return float(c)
        return None

    def max_capture_sites(self) -> float:
        return self.params.max_capture_sites()
