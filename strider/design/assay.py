"""
Assay and AssayPanel — design abstractions for nucleic-acid circuits.

An :class:`Assay` captures the design intent for a single biochemical assay:
a set of *on-target* assemblies that should form a chosen structure at
chosen concentrations, plus an optional set of *off-target* assemblies that
should not form.  An assay turns into a composable
:class:`~strider.design.objective.DesignObjective` and drops straight into
:class:`~strider.design.optimizer.SequenceDesigner`.

An :class:`AssayPanel` is a list of assays — the panel-wide design
objective is the sum of each assay's defect.

The vocabulary is the wet-lab vocabulary of the CHA / biosensor domain
that strider was built for: you design the molecular reagents for an
*assay*, and an *assembly* is the multi-strand complex you expect (or
forbid) at equilibrium.

Example
-------
>>> from strider import Assay, Assembly, DomainSpec
>>> from strider.thermo.engine import ThermoEngine
>>> from strider.design.optimizer import SequenceDesigner
>>> engine = ThermoEngine(material="dna")
>>> assay = Assay(
...     name="hairpin",
...     on_targets=[Assembly("H", ["H"], "((((....))))",
...                          concentration=1e-7)],
...     off_targets=[Assembly("H_H", ["H", "H"])],
... )
>>> obj = assay.to_objective(engine)
>>> designer = SequenceDesigner(engine, seed=0)
>>> result = designer.design(
...     domains={"H": DomainSpec(length=12)}, objective=obj, n_trials=3,
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.design.objective import DesignObjective


@dataclass
class Assembly:
    """
    A multi-strand complex specification used inside an :class:`Assay`.

    Attributes
    ----------
    name          : human-readable label (used in objective breakdowns).
    strands       : ordered list of strand names that form the assembly.
    structure     : optional dot-bracket target.  Required for on-targets
                    (the ensemble-defect computation needs a target).  For
                    off-targets, leave ``None``.
    concentration : intended equilibrium concentration (M).  Used as a
                    weighting factor when summing per-assembly defects.
    """
    name: str
    strands: list[str]
    structure: str | None = None
    concentration: float = 1e-6


@dataclass
class Assay:
    """
    A single biochemical assay design specification.

    On-target assemblies contribute their ensemble defect (lower is better)
    weighted by ``concentration``.  Off-target assemblies contribute a
    penalty proportional to how strongly they form (large negative ΔΔG ⇒
    large penalty).
    """
    name: str
    on_targets: list[Assembly] = field(default_factory=list)
    off_targets: list[Assembly] = field(default_factory=list)
    off_target_ddg_threshold: float = -4.0   # kcal/mol; penalize below this
    off_target_penalty_weight: float = 1.0   # multiplier on the penalty term

    def defect(
        self,
        sequences: dict[str, str],
        engine: "ThermoEngine",
    ) -> float:
        """
        Total assay defect for the given sequence assignment.

        ``Σ_on  c_i · ensemble_defect_i  +  Σ_off  penalty(ΔΔG_j)``
        where ``penalty(ΔΔG) = max(0, threshold − ΔΔG)²`` (quadratic above
        threshold).
        """
        total = 0.0

        for asm in self.on_targets:
            try:
                strands = tuple(sequences[s] for s in asm.strands)
            except KeyError:
                continue
            if not asm.structure:
                continue
            try:
                d = engine.ensemble_defect(strands, asm.structure)
            except ValueError:
                continue
            total += d * max(asm.concentration, 0.0)

        for asm in self.off_targets:
            try:
                strands = tuple(sequences[s] for s in asm.strands)
            except KeyError:
                continue
            if len(strands) < 2:
                continue
            try:
                ddg = engine.ddg(list(strands), [list(strands)])
            except Exception:
                continue
            if ddg < self.off_target_ddg_threshold:
                gap = self.off_target_ddg_threshold - ddg
                total += self.off_target_penalty_weight * gap * gap

        return total

    def to_objective(
        self,
        engine: "ThermoEngine",
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """Wrap the assay's defect computation as a :class:`DesignObjective` term."""
        from strider.design.objective import DesignObjective

        lbl = label or f"assay({self.name})"

        def fn(seqs: dict[str, str]) -> float:
            return self.defect(seqs, engine)

        obj = DesignObjective()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj


@dataclass
class AssayPanel:
    """A collection of :class:`Assay`s — the panel's design objective is summed across all."""
    assays: list[Assay] = field(default_factory=list)

    def add_assay(self, assay: Assay) -> None:
        self.assays.append(assay)

    def defect(self, sequences: dict[str, str], engine: "ThermoEngine") -> float:
        return sum(a.defect(sequences, engine) for a in self.assays)

    def to_objective(
        self,
        engine: "ThermoEngine",
        weight: float = 1.0,
    ) -> "DesignObjective":
        """Sum each assay's individual objective."""
        from strider.design.objective import DesignObjective

        combined = DesignObjective()
        for assay in self.assays:
            combined = combined + assay.to_objective(engine, weight=weight)
        return combined
