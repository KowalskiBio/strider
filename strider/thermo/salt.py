"""
Salt correction models for nucleic acid thermodynamics.

Sources:
  - Owczarzy et al. (2004) Biochemistry 43:3537-3554  (Na+ correction)
  - Owczarzy et al. (2008) Biochemistry 47:5336-5353  (Mg2+ correction)
  - Tan & Chen (2006) Biophys. J. 90:1175-1190         (unified model)
"""

from __future__ import annotations
import math


def owczarzy_tm_correction(
    seq: str,
    sodium_M: float,
    magnesium_M: float = 0.0,
) -> float:
    """
    Tm correction (°C) relative to 1 M NaCl reference.

    Uses Owczarzy 2004 for pure Na+ and Owczarzy 2008 Eq. 16 for Mg2+.
    When both ions present, uses the Mg2+/Na+ ratio to select the regime.
    """
    fGC = _fgc(seq)

    if magnesium_M > 0 and sodium_M > 0:
        ratio = math.sqrt(magnesium_M) / sodium_M
        if ratio < 0.22:
            return _na_correction(fGC, sodium_M)
        elif ratio < 6.0:
            return _mixed_correction(fGC, sodium_M, magnesium_M)
        else:
            return _mg_correction(fGC, magnesium_M)
    elif magnesium_M > 0:
        return _mg_correction(fGC, magnesium_M)
    else:
        return _na_correction(fGC, sodium_M)


def na_correction_dg(seq: str, sodium_M: float, celsius: float = 37.0) -> float:
    """
    ΔG correction (kcal/mol) for non-1M NaCl conditions.

    Approximated from the Tm correction via:
        ΔTm ≈ -ΔΔG / (ΔS)
    Using the simplified linear approximation from Owczarzy 2004.
    """
    n = len(seq) - 1  # number of phosphates
    if n <= 0 or sodium_M <= 0:
        return 0.0
    # Owczarzy 2004 Eq. 4 (simplified): 1/Tm_x = 1/Tm_1M + (0.368/n)·ln([Na+])
    # Convert Tm shift to ΔG shift at working temperature:
    # ΔΔG ≈ ΔΔTm · ΔS  (first-order)
    # We use the simpler per-phosphate formula:
    dG_correction = 0.368 * n * math.log(sodium_M) * 1.987e-3 * (celsius + 273.15) / 1000.0
    return -dG_correction  # stabilizing when [Na+] > 1M


# ─── private ─────────────────────────────────────────────────────────────────

def _fgc(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq) if seq else 0.5


def _na_correction(fGC: float, sodium_M: float) -> float:
    """Owczarzy 2004 Eq. 4 linearized around reference."""
    ln_na = math.log(sodium_M)
    inv_Tm_correction = (4.29 * fGC - 3.95) * 1e-5 * ln_na + 9.40e-6 * ln_na ** 2
    # Return approximate ΔTm by assuming Tm ≈ 340 K (first-order)
    Tm_ref = 340.0
    return -inv_Tm_correction * Tm_ref ** 2


def _mg_correction(fGC: float, mg_M: float) -> float:
    """Owczarzy 2008 Eq. 16."""
    ln_mg = math.log(mg_M)
    a, b, c, d, e, f, g = (
        3.92e-5, -9.11e-6, 6.26e-5, 1.42e-5,
        -4.82e-4, 5.25e-4, 8.31e-5,
    )
    inv_Tm_corr = (
        a + b * ln_mg + fGC * (c + d * ln_mg)
        + (1.0 / (2.0 * (1.0))) * (e + f * ln_mg + g * ln_mg ** 2)
    )
    Tm_ref = 340.0
    return -inv_Tm_corr * Tm_ref ** 2


def _mixed_correction(fGC: float, sodium_M: float, magnesium_M: float) -> float:
    """Owczarzy 2008 mixed-ion regime."""
    na_part = _na_correction(fGC, sodium_M)
    mg_part = _mg_correction(fGC, magnesium_M)
    ratio = math.sqrt(magnesium_M) / sodium_M
    alpha = (ratio - 0.22) / (6.0 - 0.22)
    return (1 - alpha) * na_part + alpha * mg_part
