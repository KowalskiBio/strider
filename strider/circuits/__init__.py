"""
Circuit templates for DSD designs.

Each template is a :class:`~strider.circuits.base.CircuitTemplate` subclass
that bundles a strand set, a reaction topology, a toehold map, and a
default verification suite — and emits a
:class:`~strider.bridge.mantis_bridge.CircuitBridge` ready for mantis
simulation.
"""

from strider.circuits.base import CircuitTemplate
from strider.circuits.checks import (
    CheckContext, CheckResult, CircuitReport, CheckRegistry,
    toehold_accessible, stability_in_range, reaction_driving_force,
    no_spurious_dimer, leakage_below_signal, custom,
)
from strider.circuits.hcr import HCR
from strider.circuits.translator import Translator
from strider.circuits.seesaw import SeesawGate
from strider.circuits.cha import CHA

__all__ = [
    "CircuitTemplate",
    "CheckContext", "CheckResult", "CircuitReport", "CheckRegistry",
    "toehold_accessible", "stability_in_range", "reaction_driving_force",
    "no_spurious_dimer", "leakage_below_signal", "custom",
    "HCR", "Translator", "SeesawGate", "CHA",
]
