"""
Catalytic Hairpin Assembly (CHA) as a circuit template.

Same biophysics as :class:`strider.bridge.mantis_bridge.CHABridge`, but
implemented as a :class:`~strider.circuits.base.CircuitTemplate` so it
plugs into the generic verification framework.  Recommended for new code;
``CHABridge`` is retained as a thin compatibility shim.

References
----------
Yin P. et al. (2008) Nature 451:318-322.
"""

from __future__ import annotations

from dataclasses import dataclass

from strider.circuits.base import CircuitTemplate


@dataclass
class CHA(CircuitTemplate):
    """
    Catalytic Hairpin Assembly (4-reaction topology).

    Parameters
    ----------
    sequences   : dict with keys ``{"mirna", "H1", "H2", "CP"}``.
                  Renaming the trigger from ``mirna`` to anything else is
                  fine — pass it as ``mirna``.
    toehold_d1  : initiation toehold length (miRNA·H1), nt.
    toehold_d2  : branch-migration toehold length (H1·H2), nt.
    tail_cp     : capture-probe tail length (H1H2·CP), nt.
    """
    toehold_d1: int = 6
    toehold_d2: int = 11
    tail_cp:    int = 9

    def __post_init__(self):
        if self.name == "circuit":
            self.name = "CHA"
        if not self.reactions:
            self.reactions = [
                "mirna + H1 <-> mirna_H1",
                "mirna_H1 + H2 <-> H1_H2 + mirna",
                "H1_H2 + CP <-> H1_H2_CP",
                "H1 + H2 <-> H1_H2",
            ]
        if not self.toehold_map:
            self.toehold_map = {
                "mirna + H1 <-> mirna_H1":         self.toehold_d1,
                "mirna_H1 + H2 <-> H1_H2 + mirna": self.toehold_d2,
                "H1_H2 + CP <-> H1_H2_CP":         self.tail_cp,
            }

    def _default_checks(self):
        from strider.circuits.checks import (
            CheckRegistry, toehold_accessible, stability_in_range,
            reaction_driving_force, no_spurious_dimer, leakage_below_signal,
        )
        from strider.kinetics.tmsd import toehold_kf

        n_th = self.toehold_d1
        h1_seq = self.sequences.get("H1", "")
        access_positions = list(range(min(n_th, len(h1_seq))))
        # Compute signal forward rate once so the leakage check can compare.
        material = "rna" if any("U" in s for s in self.sequences.values()) else "dna"
        signal_kf = toehold_kf(self.toehold_d1, material, 37.0)

        return (CheckRegistry()
            .add(toehold_accessible("H1", access_positions, min_prob=0.5,
                                    name="toehold_accessible"))
            .add(stability_in_range("H1", -12, -4, name="H1_stability"))
            .add(stability_in_range("H2", -12, -4, name="H2_stability"))
            .add(reaction_driving_force(["mirna", "H1"], [["mirna", "H1"]],
                                        max_ddg=-3.0, name="R1_driving_force"))
            .add(reaction_driving_force([["mirna", "H1"], "H2"],
                                        [["H1", "H2"], "mirna"],
                                        max_ddg=-3.0, name="R2_driving_force"))
            .add(reaction_driving_force([["H1", "H2"], "CP"], [["H1", "H2", "CP"]],
                                        max_ddg=-8.0, name="R3_driving_force"))
            .add(no_spurious_dimer("H2", "CP", min_ddg=-6.0,
                                   name="CP_leakage"))
            .add(leakage_below_signal(signal_kf=signal_kf, hairpin="H1",
                                      ratio=1e-4, name="spontaneous_leakage"))
        )
