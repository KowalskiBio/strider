"""
Reporter-label models for the surface transducer.

A :class:`LabelModel` maps one *captured binding event* (e.g. a docked dimer on
an electrode-bound probe) onto an electrical signal — here, a faradaic charge in
Coulombs.  Decoupling this from the capture geometry lets the same
:class:`~strider.surface.transducer.SurfaceModel` serve different read-out
chemistries: a redox nanoparticle today, an enzyme/fluorophore tomorrow (map its
turnover/photon flux onto an equivalent charge).

The reference implementation, :class:`PrussianBlueLabel`, models a solid
Prussian-Blue nanoparticle (PBNP) reporter read by differential-pulse
voltammetry (DPV): each PBNP carries many redox-active Fe centres, but only the
outer shell reached by a charge-compensating counterion (K⁺) within one pulse
actually switches.  :class:`ReadoutChain` turns the analogue front-end (TIA gain
+ ADC resolution + averaging) into the smallest resolvable current/charge.

All SI units unless noted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

E_CHARGE = 1.602176634e-19          # C
N_A = 6.02214076e23                 # /mol
# Prussian Blue Fe4[Fe(CN)6]3: ρ≈1.8 g/cm³, M≈859 g/mol, 7 Fe per formula unit
#   → Fe number density ≈ (1.8/859)·N_A·7 ≈ 8.8e21 /cm³ = 8.8e27 /m³
PB_FE_PER_M3 = 8.8e27


class LabelModel(ABC):
    """Maps one captured binding event onto a faradaic charge (Coulombs)."""

    @abstractmethod
    def signal_per_event(self) -> float:
        """Charge contributed to the read-out by a single captured event (C)."""
        raise NotImplementedError


def pbnp_redox_centres(diameter_nm: float) -> float:
    """Total redox-active Fe centres in a solid PB nanoparticle of given size."""
    r = diameter_nm * 1e-9 / 2.0
    vol = (4.0 / 3.0) * np.pi * r**3            # m³
    return PB_FE_PER_M3 * vol


def pb_addressable_fraction(diameter_nm: float, pulse_s: float,
                            counterion_d_m2_s: float,
                            penetration_factor: float = 2.0) -> float:
    """Fraction of a spherical PB particle's Fe centres that switch in one DPV
    pulse, limited by counterion (K⁺) diffusion into the lattice.

    Each PB ⇌ PW redox switch needs a charge-compensating K⁺ to enter/leave the
    open framework.  During a pulse of width ``pulse_s`` the K⁺ front advances a
    depth  λ = √(penetration_factor · D_K · t)  (a Cottrell-like penetration; the
    order-unity prefactor is absorbed into the poorly-known D_K).  Only the outer
    shell of thickness λ is reached electrochemically; the unreacted core of
    radius (r − λ) stays redox-silent.  For a sphere of radius r:

        f_addr = [r³ − max(r−λ, 0)³] / r³ = 1 − (1 − min(λ/r, 1))³

    Consequences this captures that a constant cannot:
      • small particles are *fully* addressable (λ ≳ r at fM-relevant sizes),
        unlike thick PB films where only a surface layer switches — this favours
        the post-hoc NANOPARTICLE label;
      • f_addr collapses for large particles or pessimistic (low) D_K.
    """
    r = diameter_nm * 1e-9 / 2.0
    lam = np.sqrt(penetration_factor * counterion_d_m2_s * pulse_s)
    x = min(lam / r, 1.0)
    return 1.0 - (1.0 - x) ** 3


@dataclass
class PrussianBlueLabel(LabelModel):
    """Solid Prussian-Blue nanoparticle reporter read by DPV.

    The charge per captured event is

        Q_event = m · Z · f_addr · n_e · e

    where ``m`` reporters dock per event, each a PBNP of ``Z`` Fe centres, of
    which a counterion-limited fraction ``f_addr`` switch per pulse, ``n_e``
    electrons per switch, ``e`` the elementary charge.
    """
    diameter_nm: float = 40.0          # reporter nanoparticle size
    reporters_per_event: float = 1.0   # m: PBNPs recruited per captured event
    n_electrons: int = 1               # electrons per Fe redox event
    dpv_pulse_s: float = 0.05          # effective DPV pulse width (50 ms)
    # PB-film addressability (counterion-diffusion sub-model)
    counterion_d_m2_s: float = 1.0e-14 # apparent K⁺ diffusion in the PB lattice
                                       #   (≈1e-10 cm²/s; literature spans
                                       #    1e-13…1e-16 m²/s → big f_addr lever)
    penetration_factor: float = 2.0    # λ = √(factor·D_K·t_pulse) prefactor
    f_addressable_override: float | None = None  # set to bypass the sub-model

    def f_addressable(self) -> float:
        """Fraction of each PBNP's Fe centres reached in one DPV pulse."""
        if self.f_addressable_override is not None:
            return float(self.f_addressable_override)
        return pb_addressable_fraction(self.diameter_nm, self.dpv_pulse_s,
                                       self.counterion_d_m2_s,
                                       self.penetration_factor)

    def redox_electrons_per_event(self) -> float:
        """Addressable redox electrons contributed by one captured event."""
        Z = pbnp_redox_centres(self.diameter_nm)
        return self.reporters_per_event * Z * self.f_addressable() * self.n_electrons

    def signal_per_event(self) -> float:
        return self.redox_electrons_per_event() * E_CHARGE


@dataclass
class ReadoutChain:
    """Analogue front-end → smallest resolvable current/charge.

    Defaults are the LMP91000 transimpedance amplifier (max gain) + an
    ESP32-S3 12-bit SAR ADC with ×64 averaging.
    """
    tia_gain_ohm: float = 350e3        # LMP91000 R_TIA (max gain → smallest I)
    adc_bits: int = 12                 # ESP32-S3 SAR-ADC resolution
    adc_vref_V: float = 3.0            # usable ADC full-scale (post-attenuation)
    adc_averages: int = 64             # samples averaged per point (√N noise cut)
    dpv_pulse_s: float = 0.05          # pulse width coupling I↔Q
    current_floor_override_A: float | None = None
    charge_floor_override_C: float | None = None

    def current_floor_A(self) -> float:
        """Smallest resolvable peak current from the read-out chain.

        Quantisation step V_LSB = V_ref / 2^bits at the ADC, reduced by √N
        averaging (white/quantisation noise), referred to current through the
        TIA gain:  I_floor = (V_LSB / √N) / R_TIA.
        """
        if self.current_floor_override_A is not None:
            return float(self.current_floor_override_A)
        v_lsb = self.adc_vref_V / (2 ** self.adc_bits)
        v_noise = v_lsb / np.sqrt(max(self.adc_averages, 1))
        return v_noise / self.tia_gain_ohm

    def charge_floor_C(self) -> float:
        """Charge floor; couples to the current floor over one pulse width."""
        if self.charge_floor_override_C is not None:
            return float(self.charge_floor_override_C)
        return self.current_floor_A() * self.dpv_pulse_s
