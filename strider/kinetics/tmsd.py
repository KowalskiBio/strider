"""
Toehold-mediated strand displacement (TMSD) kinetics.

Empirical rate models from:
    Zhang & Winfree (2009) JACS 131:17303-17314
    Srinivas et al. (2013) Nucleic Acids Res. 41:10641-10658

Temperature corrections use Arrhenius with Ea ~ 20 kcal/mol (DNA TMSD).

All rate constants:
    kf in M^-1 s^-1  (bimolecular)
    kr in s^-1        (unimolecular)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine

R = 1.987e-3  # kcal / (mol · K)
T_REF = 298.15  # 25°C reference for Zhang & Winfree 2009 data

# Zhang & Winfree 2009 Fig. 4 — empirical kf at 25°C (M^-1 s^-1)
# Toehold length (nt) → kf at 25°C, DNA, 11.5 mM Mg2+
_ZW_KF_25C: dict[int, float] = {
    0:  3.0e0,
    1:  1.0e2,
    2:  1.0e3,
    3:  1.6e4,
    4:  3.2e4,
    5:  8.0e4,
    6:  3.0e5,
    7:  1.0e6,
    8:  2.5e6,
    9:  3.5e6,
    10: 4.0e6,
    11: 4.5e6,
    12: 5.0e6,   # saturating
}

_EA_TMSD = 20.0  # kcal/mol (Arrhenius activation energy for DNA TMSD)


@dataclass
class TMSDRateSet:
    """
    Complete kinetic description of a single TMSD reaction.

    Attributes
    ----------
    kf             : forward rate constant (M⁻¹ s⁻¹)
    kr             : reverse rate constant (s⁻¹)
    k_eq           : equilibrium constant = kf / kr (M⁻¹)
    ddg            : reaction ΔΔG (kcal/mol)
    toehold_length : number of toehold nucleotides used
    mechanism      : "toehold_binding" | "branch_migration" | "leakage"
    """
    kf: float           # M^-1 s^-1 (forward)
    kr: float           # s^-1      (reverse)
    k_eq: float         # M^-1 = kf / kr
    ddg: float          # kcal/mol
    toehold_length: int
    mechanism: str      # "toehold_binding" | "branch_migration" | "leakage"


def toehold_kf(
    n_nt: int,
    material: str = "dna",
    celsius: float = 37.0,
) -> float:
    """
    Forward rate constant for toehold-mediated strand displacement (M^-1 s^-1).

    Uses Zhang & Winfree 2009 empirical lookup at 25°C, then applies
    Arrhenius correction to the target temperature.

    For RNA: apply +30% correction (RNA TMSD is slightly faster on average).
    """
    n_nt = max(0, min(n_nt, 12))

    # Interpolate if not in table
    if n_nt in _ZW_KF_25C:
        kf_25 = _ZW_KF_25C[n_nt]
    else:
        # Log-linear interpolation
        n_lo = max(k for k in _ZW_KF_25C if k <= n_nt)
        n_hi = min(k for k in _ZW_KF_25C if k >= n_nt)
        if n_lo == n_hi:
            kf_25 = _ZW_KF_25C[n_lo]
        else:
            frac = (n_nt - n_lo) / (n_hi - n_lo)
            kf_25 = 10 ** (
                (1 - frac) * math.log10(_ZW_KF_25C[n_lo])
                + frac * math.log10(_ZW_KF_25C[n_hi])
            )

    # Arrhenius temperature correction
    kf = arrhenius_rate(kf_25, _EA_TMSD, T_REF, celsius + 273.15)

    # Material correction
    if material == "rna":
        kf *= 1.3
    elif material == "dna_rna":
        kf *= 1.1

    return kf


def displacement_kf(
    n_nt: int,
    material: str = "dna",
    celsius: float = 37.0,
) -> float:
    """
    Branch migration rate constant for strand displacement without toehold.

    From Zhang & Winfree 2009 Fig. 2: ~1 M^-1 s^-1 per nt at 25°C.
    """
    kf_25 = n_nt * 1.0
    return arrhenius_rate(kf_25, _EA_TMSD, T_REF, celsius + 273.15)


def leakage_kf(
    stem_stability_kcal: float,
    kf_max: float = 1e6,
    celsius: float = 37.0,
) -> float:
    """
    Boltzmann-suppressed leakage forward rate (M^-1 s^-1).

    Models spontaneous hairpin breathing where the stem stability provides
    an activation energy barrier. Based on Turberfield et al. (2003).

        k_leak = kf_max * exp(-|ΔG_stem| / RT)

    stem_stability_kcal: absolute value of the stem free energy (positive number)
    """
    T = celsius + 273.15
    ea = abs(stem_stability_kcal)
    return kf_max * math.exp(-ea / (R * T))


def rates_from_ddg(
    ddg: float,
    kf: float,
    celsius: float = 37.0,
) -> tuple[float, float]:
    """
    Derive (kf, kr) from ΔΔG and empirical kf via detailed balance.

        K_eq = kf / kr = exp(-ΔΔG / RT)
        kr = kf / K_eq = kf * exp(ΔΔG / RT)

    Returns (kf, kr).
    """
    T = celsius + 273.15
    kr = kf * math.exp(ddg / (R * T))
    return kf, max(kr, 1e-30)  # clamp to avoid exact zero


def arrhenius_rate(
    k_ref: float,
    ea_kcal: float,
    T_ref_K: float,
    T_K: float,
) -> float:
    """
    Scale a rate constant from reference temperature to target temperature.

        k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
    """
    return k_ref * math.exp(-ea_kcal / R * (1.0 / T_K - 1.0 / T_ref_K))


class TMSDKineticModel:
    """
    Build complete kinetic rate dictionaries for DNA/RNA circuits.

    Computes ΔΔG internally from ThermoEngine, then applies the
    Zhang & Winfree empirical toehold model to get forward rates.
    Reverse rates derived from detailed balance.
    """

    def __init__(
        self,
        engine: "ThermoEngine",
        celsius: float | None = None,
    ) -> None:
        """
        Initialize the kinetic model.

        Parameters
        ----------
        engine  : ThermoEngine used for ΔΔG calculations
        celsius : temperature override (defaults to engine.celsius)
        """
        self.engine = engine
        self.celsius = celsius if celsius is not None else engine.celsius

    def reaction_rates(
        self,
        reactant_seqs: list[str],
        product_seqs: list[str],
        toehold_length: int | None = None,
        mechanism: str = "toehold_binding",
    ) -> TMSDRateSet:
        """
        Compute TMSD rate set for a single reaction.

        reactant_seqs / product_seqs: sequences for each species
            (single sequences = monomers, lists = complexes)
        toehold_length: if None, defaults to 6 nt
        """
        ddg = self.engine.ddg(reactant_seqs, product_seqs)
        nt = toehold_length if toehold_length is not None else 6
        kf = toehold_kf(nt, self.engine.material, self.celsius)
        kf_val, kr_val = rates_from_ddg(ddg, kf, self.celsius)
        return TMSDRateSet(
            kf=kf_val,
            kr=kr_val,
            k_eq=kf_val / kr_val if kr_val > 0 else float("inf"),
            ddg=ddg,
            toehold_length=nt,
            mechanism=mechanism,
        )

    def circuit_rates(
        self,
        reactions: list[str],
        sequences: dict[str, str],
        toehold_map: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """
        Build a mantis-compatible rate dict for a list of reactions.

        reactions   : mantis-style strings ("A + B <-> C + D")
        sequences   : species_name -> sequence
        toehold_map : reaction_string -> toehold_length (optional)

        Returns dict keyed by canonical mantis rate keys.
        """
        rates: dict[str, float] = {}
        for rxn_str in reactions:
            forward, reverse = _parse_reversible(rxn_str, sequences)
            if forward is None:
                continue
            nt = (toehold_map or {}).get(rxn_str, 6)

            # Only compute ΔΔG if all species have known sequences
            react_seqs = [sequences[s] for s in forward[0] if s in sequences]
            prod_seqs = [sequences[s] for s in forward[1] if s in sequences]
            if react_seqs and prod_seqs:
                fwd_ddg = self.engine.ddg(react_seqs, prod_seqs)
            else:
                fwd_ddg = -5.0  # default moderately favorable
            kf = toehold_kf(nt, self.engine.material, self.celsius)
            kf_val, kr_val = rates_from_ddg(fwd_ddg, kf, self.celsius)

            fwd_key = _mantis_rate_key(forward[0], forward[1])
            rev_key = _mantis_rate_key(forward[1], forward[0])
            rates[fwd_key] = kf_val
            if reverse is not None:
                rates[rev_key] = kr_val

        return rates


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_reversible(
    rxn_str: str,
    sequences: dict[str, str],
) -> tuple[tuple | None, tuple | None]:
    """Parse a mantis-style reaction string into forward and reverse (reactants, products) tuples."""
    sep = "<->" if "<->" in rxn_str else "->"
    parts = rxn_str.split(sep)
    if len(parts) < 2:
        return None, None

    def _parse_side(s: str) -> list[str]:
        return [tok.strip() for tok in s.split("+") if tok.strip()]

    reactants = _parse_side(parts[0])
    products = _parse_side(parts[1])
    fwd = (reactants, products)
    rev = (products, reactants) if sep == "<->" else None
    return fwd, rev


def _mantis_rate_key(reactants: list[str], products: list[str]) -> str:
    """Build a canonical mantis rate dict key of the form 'A + B -> C + D'."""
    r = " + ".join(sorted(reactants))
    p = " + ".join(sorted(products))
    return f"{r} -> {p}"
