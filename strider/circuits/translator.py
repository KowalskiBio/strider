"""
Generic signal translator: input strand X triggers release of output strand Y.

A *translator* is the elementary signal-transduction module of DSD
circuits: an input strand displaces an output strand from a hairpin or a
hybridized duplex, converting an "X is present" signal into a "Y is
present" signal that downstream gates can act on.

Default reaction topology
-------------------------
    X + Gate <-> X_Gate + Y       (toehold-mediated displacement)
    Gate <-> X + Y                (no-input leakage, very slow)
"""

from __future__ import annotations

from dataclasses import dataclass

from strider.circuits.base import CircuitTemplate


@dataclass
class Translator(CircuitTemplate):
    """
    Two-strand translator (X → Y) via toehold-mediated displacement on a Gate.

    Parameters
    ----------
    sequences          : dict with keys ``{"X", "Y", "Gate"}``.
                         ``Gate`` is the duplex that hybridizes Y; X displaces it.
    toehold_x          : length of the toehold on X used to invade Gate.
    """
    toehold_x: int = 6

    def __post_init__(self):
        if self.name == "circuit":
            self.name = "Translator"
        if not self.reactions:
            self.reactions = [
                "X + Gate <-> X_Gate + Y",
                "Gate <-> Y",   # ideally negligible leakage
            ]
        if not self.toehold_map:
            self.toehold_map = {"X + Gate <-> X_Gate + Y": self.toehold_x}

    def _default_checks(self):
        from strider.circuits.checks import (
            CheckRegistry, reaction_driving_force, no_spurious_dimer,
            toehold_accessible,
        )
        reg = (CheckRegistry()
            .add(toehold_accessible("X", range(min(self.toehold_x,
                                                   len(self.sequences.get("X", ""))))))
            .add(reaction_driving_force(["X", "Gate"], [["X", "Gate"], "Y"],
                                        max_ddg=-3.0, name="translation_driving_force"))
            .add(no_spurious_dimer("X", "Y", min_ddg=-4.0,
                                   name="X_Y_crosstalk"))
        )
        return reg
