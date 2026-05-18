"""
Hybridization Chain Reaction (HCR).

Dirks & Pierce (2004) PNAS 101:15275-15278.

Two metastable hairpins H1 and H2 coexist in solution.  An initiator I
opens H1; the newly exposed region opens H2; the newly exposed region of
H2 opens another H1; and so on, propagating a long double-stranded
polymer until one of the hairpins is exhausted.

Default reaction topology
-------------------------
    I  + H1 <-> I_H1
    I_H1 + H2 <-> I_H1_H2
    I_H1_H2 + H1 <-> I_H1_H2_H1
    H1 + H2 <-> H1_H2     (spontaneous leakage)

The chain grows indefinitely in principle; for tractability we model the
first two extension steps explicitly and rely on the bridge's leakage
enumeration for longer-range checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strider.circuits.base import CircuitTemplate


@dataclass
class HCR(CircuitTemplate):
    """
    HCR amplifier with initiator + two metastable hairpins.

    Parameters
    ----------
    sequences         : dict with keys ``{"I", "H1", "H2"}``.
    toehold_initiator : length of the initiator's toehold onto H1 (nt).
    toehold_branch    : length of H1's newly exposed toehold onto H2 (nt).
    """
    toehold_initiator: int = 6
    toehold_branch: int = 6

    def __post_init__(self):
        if self.name == "circuit":
            self.name = "HCR"
        # Auto-build the standard reaction list if user hasn't customized.
        if not self.reactions:
            self.reactions = [
                "I + H1 <-> I_H1",
                "I_H1 + H2 <-> I_H1_H2",
                "I_H1_H2 + H1 <-> I_H1_H2_H1",
                "H1 + H2 <-> H1_H2",
            ]
        if not self.toehold_map:
            self.toehold_map = {
                "I + H1 <-> I_H1":              self.toehold_initiator,
                "I_H1 + H2 <-> I_H1_H2":        self.toehold_branch,
                "I_H1_H2 + H1 <-> I_H1_H2_H1":  self.toehold_branch,
            }

    def _default_checks(self):
        from strider.circuits.checks import (
            CheckRegistry, stability_in_range, reaction_driving_force,
            toehold_accessible, no_spurious_dimer,
        )
        return (CheckRegistry()
            .add(stability_in_range("H1", -12, -4, name="H1_stability"))
            .add(stability_in_range("H2", -12, -4, name="H2_stability"))
            .add(toehold_accessible("I", range(min(self.toehold_initiator,
                                                   len(self.sequences.get("I", ""))))))
            .add(reaction_driving_force(["I", "H1"], [["I", "H1"]],
                                        max_ddg=-3.0, name="initiation_driving_force"))
            .add(reaction_driving_force([["I", "H1"], "H2"], [["I", "H1", "H2"]],
                                        max_ddg=-3.0, name="propagation_driving_force"))
            .add(no_spurious_dimer("H1", "H2", min_ddg=-6.0,
                                   name="H1_H2_leakage"))
        )
