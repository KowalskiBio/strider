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

from strider.tube import Complex

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.design.objective import DesignObjective


class Assembly:
    """
    A multi-strand complex specification used inside an :class:`Assay`.

    Composes a :class:`strider.tube.Complex` with the design metadata that
    only exists at design-spec time (target dot-bracket structure, intended
    equilibrium concentration).  At construction the strand identifiers are
    stored as a name-only :class:`Complex` because sequences are not yet
    known; the designer resolves them at evaluation time by calling
    ``engine.ensemble_defect(sequences, structure)``.

    Constructors
    ------------
    The original keyword-positional signature is preserved::

        Assembly("H_H", ["H", "H"], structure=None, concentration=1e-6)

    Or build from a :class:`Complex` directly via :meth:`from_complex`::

        Assembly.from_complex(
            Complex.from_names(["H"], name="hairpin"),
            structure="((((....))))",
            concentration=1e-7,
        )

    Attributes
    ----------
    complex       : the underlying :class:`Complex` (always name-only).
    structure     : optional dot-bracket target (None for off-targets).
    concentration : intended equilibrium concentration (M).
    """

    __slots__ = ("complex", "structure", "concentration")

    def __init__(
        self,
        name: str,
        strands: list[str],
        structure: str | None = None,
        concentration: float = 1e-6,
    ) -> None:
        self.complex = Complex.from_names(list(strands), name=name)
        self.structure = structure
        self.concentration = concentration

    @classmethod
    def from_complex(
        cls,
        complex: Complex,
        structure: str | None = None,
        concentration: float = 1e-6,
    ) -> "Assembly":
        """Build an :class:`Assembly` around an existing :class:`Complex`."""
        obj = cls.__new__(cls)
        obj.complex = complex
        obj.structure = structure
        obj.concentration = concentration
        return obj

    # ─── backward-compatible attribute surface ────────────────────────────────

    @property
    def name(self) -> str:
        """Assembly identifier (mirrors ``complex.canonical_name``)."""
        return self.complex.canonical_name

    @property
    def strands(self) -> list[str]:
        """Ordered strand-name list (kept as ``list[str]`` for backward compat)."""
        return list(self.complex.strand_names)

    def __repr__(self) -> str:
        return (
            f"Assembly(name={self.name!r}, strands={self.strands!r}, "
            f"structure={self.structure!r}, concentration={self.concentration})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Assembly):
            return NotImplemented
        return (
            self.complex == other.complex
            and self.structure == other.structure
            and self.concentration == other.concentration
        )

    def __hash__(self) -> int:
        return hash((self.complex, self.structure, self.concentration))


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
