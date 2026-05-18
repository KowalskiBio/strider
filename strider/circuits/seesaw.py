"""
Seesaw gate (Qian & Winfree 2011, Science 332:1196-1201).

The fundamental compute primitive for DNA-strand-displacement logic.
A seesaw gate has:
  * a Gate strand (carries the output signal initially bound)
  * a Threshold strand (irreversibly absorbs sub-threshold inputs)
  * one or more Input strands
  * a Fuel strand (catalytic recovery — included optionally)

By choosing the threshold concentration and the input fan-in, the same
core motif implements AND, OR, or NOT logic.

Reaction topology (single-input variant)
----------------------------------------
    Input + Threshold -> Waste                 (irreversible thresholding)
    Input + Gate     <-> Input_Gate + Output   (signal release)
    Fuel  + Input_Gate -> Fuel_Gate + Input    (input recycling)

For AND / OR, the reaction list is extended with additional inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from strider.circuits.base import CircuitTemplate


GateLogic = Literal["YES", "AND", "OR", "NOT"]


@dataclass
class SeesawGate(CircuitTemplate):
    """
    Generic seesaw gate with configurable input arity / logic.

    Parameters
    ----------
    sequences   : dict with keys ``{"Gate", "Threshold", "Fuel", "Output",
                  "Input1", "Input2", ...}``.  Only inputs you reference
                  need to be supplied — for a 1-input YES gate just
                  ``Input1``; for AND/OR include ``Input2``.
    logic       : ``"YES" | "AND" | "OR" | "NOT"``.
    toehold     : toehold length used by every TMSD step (nt).
    """
    logic: GateLogic = "YES"
    toehold: int = 6

    def __post_init__(self):
        if self.name == "circuit":
            self.name = f"Seesaw_{self.logic}"
        if not self.reactions:
            self.reactions = self._build_reactions()
        if not self.toehold_map:
            self.toehold_map = {r: self.toehold
                                for r in self.reactions
                                if "<->" in r}

    def _build_reactions(self) -> list[str]:
        rxns: list[str] = []
        inputs = self._input_names()

        # Thresholding: irreversible absorption.  AND uses one threshold per
        # input; OR/NOT/YES use a single threshold strand.
        if self.logic in ("YES", "NOT"):
            rxns.append(f"{inputs[0]} + Threshold -> Waste")
        elif self.logic == "OR":
            for inp in inputs:
                rxns.append(f"{inp} + Threshold -> Waste")
        elif self.logic == "AND":
            for inp in inputs:
                rxns.append(f"{inp} + Threshold_{inp} -> Waste_{inp}")

        # Signal release (TMSD) — reversible
        if self.logic == "NOT":
            # Output is normally released by Gate; input INHIBITS by
            # consuming Gate before release.
            rxns.append(f"Gate <-> Output")
            rxns.append(f"{inputs[0]} + Gate <-> {inputs[0]}_Gate")
        else:
            for inp in inputs:
                rxns.append(f"{inp} + Gate <-> {inp}_Gate + Output")

        # Catalytic recycling via Fuel
        for inp in inputs:
            rxns.append(f"Fuel + {inp}_Gate -> Fuel_Gate + {inp}")

        return rxns

    def _input_names(self) -> list[str]:
        if self.logic in ("YES", "NOT"):
            return ["Input1"]
        return ["Input1", "Input2"]

    def _default_checks(self):
        from strider.circuits.checks import (
            CheckRegistry, reaction_driving_force, no_spurious_dimer,
        )
        reg = CheckRegistry()
        for inp in self._input_names():
            reg.add(reaction_driving_force(
                [inp, "Gate"], [f"{inp}_Gate", "Output"],
                max_ddg=-3.0, name=f"{inp}_release_driving_force"))
            reg.add(no_spurious_dimer(
                inp, "Output", min_ddg=-4.0,
                name=f"{inp}_Output_crosstalk"))
        return reg
