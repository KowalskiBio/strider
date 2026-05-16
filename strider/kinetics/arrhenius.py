"""
Arrhenius / Eyring / detailed-balance utilities.
"""

from __future__ import annotations
import math

R = 1.987e-3   # kcal / (mol · K)
kB = 1.381e-23  # J / K
h = 6.626e-34   # J · s
NA = 6.022e23


def arrhenius(k_ref: float, ea_kcal: float, T_ref_K: float, T_K: float) -> float:
    """Scale rate constant between temperatures using Arrhenius equation."""
    return k_ref * math.exp(-ea_kcal / R * (1.0 / T_K - 1.0 / T_ref_K))


def eyring_kf(
    dG_barrier_kcal: float,
    celsius: float = 37.0,
    transmission: float = 1.0,
) -> float:
    """
    Transition state theory (Eyring) rate constant.

        k = transmission * (kB T / h) * exp(-ΔG‡ / RT)

    Returns k in s^-1 (first-order) — multiply by concentration for bimolecular.
    """
    T = celsius + 273.15
    prefactor = transmission * kB * T / h * NA  # convert to per mole
    return prefactor * math.exp(-dG_barrier_kcal / (R * T))


def detailed_balance_kr(kf: float, ddg_kcal: float, celsius: float = 37.0) -> float:
    """
    Reverse rate from forward rate and reaction ΔΔG.

        K_eq = kf / kr = exp(-ΔΔG / RT)   (ΔΔG = G_products - G_reactants)
        kr = kf * exp(ΔΔG / RT)

    Negative ΔΔG means exergonic (products lower) → K_eq > 1 → kr < kf.
    """
    T = celsius + 273.15
    return kf * math.exp(ddg_kcal / (R * T))


def activation_energy_from_rates(
    k_low: float,
    k_high: float,
    T_low_C: float,
    T_high_C: float,
) -> float:
    """
    Estimate Ea (kcal/mol) from two rate measurements at different temperatures.

        Ea = R * ln(k_high/k_low) / (1/T_low - 1/T_high)
    """
    T_lo = T_low_C + 273.15
    T_hi = T_high_C + 273.15
    if k_low <= 0 or k_high <= 0 or T_lo == T_hi:
        return 0.0
    return R * math.log(k_high / k_low) / (1.0 / T_lo - 1.0 / T_hi)


def k_eq_from_ddg(ddg_kcal: float, celsius: float = 37.0) -> float:
    """Equilibrium constant from free energy: K_eq = exp(-ΔΔG / RT)."""
    T = celsius + 273.15
    return math.exp(-ddg_kcal / (R * T))


def ddg_from_k_eq(k_eq: float, celsius: float = 37.0) -> float:
    """Free energy from equilibrium constant: ΔΔG = -RT ln(K_eq)."""
    T = celsius + 273.15
    if k_eq <= 0:
        return float("inf")
    return -R * T * math.log(k_eq)
