"""
Surface-tethered thermodynamic corrections (the warping NUPACK ignores).

Probe immobilization on an electrode/bead warps bulk duplex thermodynamics in
two ways this module models, *reusing* the existing salt machinery in
:mod:`strider.thermo.salt`:

  1. **Double-layer / local salt activity.**  A dense, charged probe monolayer
     accumulates counterions, so the *local* ionic strength a hybridizing strand
     sees differs from the bulk.  :func:`double_layer_local_salt` turns a probe
     surface density into a Gouy–Chapman surface potential ψ₀ (Grahame equation,
     1:1 electrolyte), then Boltzmann-enhances the counterion concentration at
     the surface.  Feeding that *local* salt into
     :func:`strider.thermo.salt.na_correction_dg` gives a per-strand ΔG offset —
     the screening contribution, which is genuinely per-strand and composes
     correctly into a multi-strand ΔΔG.

  2. **Configurational-entropy tether penalty.**  Pinning one end of a strand to
     a surface costs configurational entropy on hybridization.
     :func:`tether_dg` returns a (positive, destabilizing) ΔG from a tabulated
     per-spacer value plus an optional ideal-chain confinement term.  This is a
     *complex-level* term (it applies once to the formed, tethered duplex), so it
     is kept separate from the per-strand salt hook — add it to a capture ΔΔG
     explicitly rather than to every ``pfunc`` call.

:class:`SurfaceCorrection` bundles both and is callable ``(seq) -> float`` so the
per-strand salt offset plugs straight into the existing
``ThermoEngine(correction_model=...)`` hook.

NUPACK assumes a single bulk salt and free 3-D strands; it has no representation
of either effect.

All SI units unless noted; ΔG in kcal/mol, concentrations in molar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from strider.thermo.salt import na_correction_dg

# physical constants (SI)
_EPS0 = 8.8541878128e-12     # F/m
_EPS_WATER = 78.5           # relative permittivity of water (~25–37 °C)
_KB = 1.380649e-23          # J/K
_E = 1.602176634e-19        # C
_NA = 6.02214076e23         # /mol
_R_KCAL = 1.987204259e-3    # kcal/(mol·K)


# ─── configurational-entropy tether penalty ────────────────────────────────────

# Tabulated immobilization-spacer entropic penalties (kcal/mol, destabilizing).
# Documented-approximate, order-of-magnitude values for ranking designs; refine
# against melting/SPR data when available.
SPACER_TETHER_DG: dict[str, float] = {
    "none": 0.0,
    "c3": 0.8,          # C3 alkane spacer
    "c6": 1.0,          # C6 alkanethiol (classic gold–thiol)
    "thiol-c6": 1.0,
    "peg6": 1.5,        # hexa-ethylene-glycol
    "peg18": 2.5,       # longer PEG
}


def tether_dg(spacer: str | None = None, n_segments: int | None = None,
              celsius: float = 37.0) -> float:
    """Configurational-entropy penalty (kcal/mol, ≥0) for tethering one end.

    ``spacer`` keys into :data:`SPACER_TETHER_DG` (case-insensitive).  When
    ``n_segments`` is given an optional ideal-chain confinement term is added:
    localizing the duplex end near the wall costs ~(3/2)kT, relieved by a longer,
    more flexible linker (∝ 1/√n).  Both pieces are approximate heuristics.
    """
    val = 0.0
    if spacer:
        val += SPACER_TETHER_DG.get(spacer.lower(), 0.0)
    if n_segments and n_segments > 0:
        rt = _R_KCAL * (celsius + 273.15)
        val += 1.5 * rt / math.sqrt(n_segments)
    return val


# ─── electrostatic double layer → local salt activity ──────────────────────────

def debye_length_m(ionic_strength_M: float, celsius: float = 37.0) -> float:
    """Debye screening length (m) for a 1:1 electrolyte at the given salt."""
    if ionic_strength_M <= 0:
        return float("inf")
    T = celsius + 273.15
    n0 = ionic_strength_M * 1000.0 * _NA          # ions/m³
    kappa2 = 2.0 * n0 * _E**2 / (_EPS_WATER * _EPS0 * _KB * T)
    return 1.0 / math.sqrt(kappa2)


def double_layer_local_salt(bulk_na_M: float, probe_density_per_m2: float,
                            celsius: float = 37.0, z: int = 1,
                            charge_per_probe_e: float = -1.0) -> float:
    """Local counterion concentration (M) at a charged probe monolayer.

    The probe layer carries surface charge density σ = ρ_probe · q_probe · e.
    The Grahame equation (1:1 electrolyte) gives the surface potential

        σ = √(8 ε ε₀ k_B T n₀) · sinh(e ψ₀ / 2k_B T)
        ⇒ ψ₀ = (2k_B T / e) · asinh[ σ / √(8 ε ε₀ k_B T n₀) ],

    and the counterion concentration is Boltzmann-enhanced at the surface:

        [salt]_local = [salt]_bulk · exp(−z e ψ₀ / k_B T).

    For a negative monolayer (q_probe < 0 → ψ₀ < 0) this *raises* the local salt
    seen by a hybridizing strand (more screening).  Returns the bulk value
    unchanged when there is no charge or no salt.
    """
    if bulk_na_M <= 0 or probe_density_per_m2 <= 0 or charge_per_probe_e == 0:
        return bulk_na_M
    T = celsius + 273.15
    sigma = probe_density_per_m2 * charge_per_probe_e * _E      # C/m²
    n0 = bulk_na_M * 1000.0 * _NA                              # ions/m³
    pref = math.sqrt(8.0 * _EPS_WATER * _EPS0 * _KB * T * n0)
    psi0 = (2.0 * _KB * T / _E) * math.asinh(sigma / pref)     # V (negative)
    return bulk_na_M * math.exp(-z * _E * psi0 / (_KB * T))


# ─── bundled correction (plugs into ThermoEngine.correction_model) ──────────────

@dataclass
class SurfaceCorrection:
    """Surface ΔG corrections for a tethered-probe assay.

    Instances are callable ``(seq) -> float`` returning the **per-strand salt
    offset** (ΔG vs bulk), so they drop into ``ThermoEngine(correction_model=…)``
    and compose correctly into multi-strand ΔΔG.  The complex-level tether
    penalty is exposed separately via :meth:`tether_offset` — add it to a formed
    capture ΔΔG, not to every ``pfunc`` call.
    """
    bulk_na_M: float = 0.137
    probe_density_per_m2: float = 1.0e16
    spacer: str | None = None
    n_segments: int | None = None
    celsius: float = 37.0
    z: int = 1
    charge_per_probe_e: float = -1.0
    local_na_M: float = field(init=False)

    def __post_init__(self) -> None:
        self.local_na_M = double_layer_local_salt(
            self.bulk_na_M, self.probe_density_per_m2, self.celsius,
            self.z, self.charge_per_probe_e)

    def salt_offset(self, seq: str) -> float:
        """Per-strand ΔG offset (kcal/mol) from the local vs bulk salt activity.

        Negative (stabilizing) when the monolayer enhances local counterions.
        """
        return (na_correction_dg(seq, self.local_na_M, self.celsius)
                - na_correction_dg(seq, self.bulk_na_M, self.celsius))

    def tether_offset(self) -> float:
        """Complex-level configurational-entropy penalty (kcal/mol, ≥0)."""
        return tether_dg(self.spacer, self.n_segments, self.celsius)

    def __call__(self, seq: str) -> float:
        return self.salt_offset(seq)
