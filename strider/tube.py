"""
Multi-strand test-tube analysis.

A *tube* is the high-level abstraction for "this set of strands at these
concentrations — what does the equilibrium look like?".  Composing the lower
layers:

    Strand                      one nucleic acid sequence
    Complex                     an ordered tuple of Strand references
    SetSpec / ComplexSet        rules for enumerating which complexes belong
                                in the ensemble (max stoichiometry, includes,
                                excludes, plus any explicit Complex objects)
    Tube                        Strand → total-concentration mapping plus a
                                ComplexSet to consider
    TubeResult                  per-complex equilibrium concentrations and
                                ensemble free energies, plus lazy access to
                                pair probabilities and ensemble defect
    tube_analysis(tubes, eng)   batch driver across multiple tubes

The underlying convex equilibrium problem is solved by
:func:`strider.equilibrium.solve_equilibrium` (Dirks, Bois, Schaeffer,
Winfree & Pierce 2007, SIAM Review 49:65-88).  This module enumerates the
complex ensemble, calls :meth:`ThermoEngine.pfunc` for each species, and
hands the resulting ΔG table to the solver.

References
----------
Dirks R.M., Bois J.S., Schaeffer J.M., Winfree E., Pierce N.A. (2007)
    SIAM Review 49: 65-88.
McCaskill J.S. (1990) Biopolymers 29: 1105-1119.
Zadeh J.N., Wolfe B.R., Pierce N.A. (2011) J. Comput. Chem. 32: 439-452.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Iterable

import numpy as np

from strider.equilibrium import (
    EquilibriumResult,
    cyclic_symmetry,
    solve_equilibrium,
)

if TYPE_CHECKING:
    from strider.sweep.cache import DiskCache
    from strider.thermo.engine import ThermoEngine


__all__ = [
    "Strand",
    "Complex",
    "SetSpec",
    "ComplexSet",
    "Tube",
    "TubeResult",
    "tube_analysis",
]


# ─── Strand ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Strand:
    """
    A nucleic acid strand: name + sequence + material.

    Frozen / hashable so it can be used as a dict key in ``Tube.strands``.

    Parameters
    ----------
    name     : human-readable identifier (must be unique within a tube).
    sequence : nucleotide sequence (case-insensitive; U/T are normalized at
               pfunc time based on the strand's material).
    material : ``"dna"`` or ``"rna"``.
    """
    name: str
    sequence: str
    material: str = "dna"

    def __post_init__(self) -> None:
        if self.material not in ("dna", "rna"):
            raise ValueError(f"material must be 'dna' or 'rna', got {self.material!r}")
        if not self.name:
            raise ValueError("Strand.name must be non-empty")

    def __len__(self) -> int:
        return len(self.sequence)


# ─── Complex ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Complex:
    """
    An ordered multi-strand complex.

    A complex can be built in two modes:

    * **resolved** — ``strands`` is a tuple of :class:`Strand` objects with
      sequences attached.  Use this for analysis (pfunc, equilibrium).
    * **name-only** — ``strands`` is a tuple of strand-name strings.  Use this
      at design-spec time (e.g. inside :class:`~strider.design.assay.Assembly`)
      when sequences are not yet known.  Calling :attr:`sequences` or
      :attr:`total_length` on a name-only complex raises :class:`ValueError`.

    Two complexes that share the same set of strands but differ in cyclic
    rotation describe the same chemical species; equality and hashing are
    therefore based on the canonical (sorted) strand-name tuple.

    Parameters
    ----------
    strands : ordered tuple of :class:`Strand` objects *or* strand-name strings.
    name    : optional explicit name; defaults to ``"_".join(sorted(strand_names))``.
    """
    strands: tuple   # tuple[Strand, ...] | tuple[str, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        if len(self.strands) == 0:
            raise ValueError("Complex must contain at least one strand")
        # Disallow mixed name/Strand tuples.
        types = {type(s).__name__ for s in self.strands}
        if Strand.__name__ in types and "str" in types:
            raise ValueError(
                "Complex.strands must be all Strand objects or all strings, not a mix"
            )

    @classmethod
    def from_names(cls, strand_names: list[str], name: str | None = None) -> "Complex":
        """
        Construct a name-only complex (no sequences attached).

        Useful at design-specification time when only strand identities are
        known.  Resolve later by either rebuilding with real
        :class:`Strand` objects or by passing the names plus a sequence dict
        to the consumer.
        """
        return cls(strands=tuple(strand_names), name=name)

    @property
    def is_resolved(self) -> bool:
        """True if ``strands`` carries :class:`Strand` objects (sequences attached)."""
        return all(isinstance(s, Strand) for s in self.strands)

    @property
    def n_strands(self) -> int:
        """Number of strand copies in the complex (multiplicity included)."""
        return len(self.strands)

    @property
    def strand_names(self) -> tuple[str, ...]:
        """Strand names in the order supplied (used for σ calculation)."""
        return tuple(s.name if isinstance(s, Strand) else s for s in self.strands)

    @property
    def sigma(self) -> int:
        """Cyclic rotational symmetry σ — Dirks et al. 2007 eq. 11."""
        return cyclic_symmetry(list(self.strand_names))

    @property
    def canonical_name(self) -> str:
        """Stable identifier: ``self.name`` if set, else strand names sorted and joined with ``_``."""
        if self.name:
            return self.name
        return "_".join(sorted(self.strand_names))

    @property
    def sequences(self) -> tuple[str, ...]:
        """Sequences in the order the strands were supplied (requires a resolved complex)."""
        if not self.is_resolved:
            raise ValueError(
                f"Complex {self.canonical_name!r} is name-only — call "
                f"Complex(strands=tuple(Strand(...), ...)) instead, or resolve "
                f"the names through a strand_dict at use time."
            )
        return tuple(s.sequence for s in self.strands)

    @property
    def total_length(self) -> int:
        """Sum of all strand lengths — requires a resolved complex."""
        if not self.is_resolved:
            raise ValueError(
                f"Complex {self.canonical_name!r} is name-only; total_length "
                f"requires sequences."
            )
        return sum(len(s) for s in self.strands)

    def resolve(self, strand_dict: dict[str, Strand]) -> "Complex":
        """
        Return a copy with bare strand names replaced by the matching
        :class:`Strand` objects from ``strand_dict``.

        No-op if the complex is already resolved.
        """
        if self.is_resolved:
            return self
        resolved = tuple(strand_dict[name] for name in self.strand_names)
        return Complex(strands=resolved, name=self.name)

    def __eq__(self, other: object) -> bool:
        """Equal if canonical names match — handles cyclic rotations of homomers."""
        if not isinstance(other, Complex):
            return NotImplemented
        return self.canonical_name == other.canonical_name

    def __hash__(self) -> int:
        """Hash by canonical name so dict lookups stay rotation-invariant."""
        return hash(self.canonical_name)

    def __repr__(self) -> str:
        kind = "resolved" if self.is_resolved else "names"
        return f"Complex({self.canonical_name!r}, n={self.n_strands}, {kind})"


# ─── SetSpec ──────────────────────────────────────────────────────────────────

@dataclass
class SetSpec:
    """
    Enumeration rules for a :class:`ComplexSet`.

    Parameters
    ----------
    max_size : enumerate every complex of up to ``max_size`` strands (with
               replacement, so homomers like ``A_A`` appear when ``max_size ≥ 2``).
               Set ``0`` to disable automatic enumeration entirely.
    include  : explicit additional :class:`Complex` objects to force into the
               ensemble even if ``max_size`` would not have produced them.
    exclude  : explicit :class:`Complex` objects to remove from the auto
               enumeration.
    """
    max_size: int = 1
    include: list[Complex] = field(default_factory=list)
    exclude: list[Complex] = field(default_factory=list)


# ─── ComplexSet ───────────────────────────────────────────────────────────────

@dataclass
class ComplexSet:
    """
    Collection of complexes built from a strand set and a :class:`SetSpec`.

    The strand set defines *which* strands are available; the spec defines
    *how* to combine them.  Call :meth:`enumerate` to obtain the resolved list
    of unique :class:`Complex` objects (deduplicated by canonical name).

    Parameters
    ----------
    strands : iterable of available :class:`Strand` objects.
    spec    : :class:`SetSpec` controlling auto-enumeration; defaults to
              ``SetSpec(max_size=1)`` (monomers only) if omitted.
    """
    strands: tuple[Strand, ...]
    spec: SetSpec | None = None

    def __init__(
        self,
        strands: Iterable[Strand],
        spec: SetSpec | None = None,
    ) -> None:
        self.strands = tuple(strands)
        self.spec = spec if spec is not None else SetSpec(max_size=1)

    def enumerate(self) -> list[Complex]:
        """
        Generate the resolved list of :class:`Complex` objects.

        Order: monomers first, then size-2 complexes, then size-3, etc.  Each
        canonical name appears at most once.  Explicit ``include`` entries are
        appended after auto-enumeration; ``exclude`` filters both groups.
        """
        spec = self.spec or SetSpec()
        seen: dict[str, Complex] = {}
        order: list[str] = []

        # Auto-enumeration up to spec.max_size strands.
        # max_size=0 disables auto-enumeration entirely (use include= only).
        for k in range(1, spec.max_size + 1):
            for combo in combinations_with_replacement(self.strands, k):
                cx = Complex(strands=tuple(combo))
                if cx.canonical_name not in seen:
                    seen[cx.canonical_name] = cx
                    order.append(cx.canonical_name)

        # Explicit include — appended in order, deduplicated by canonical name.
        for cx in spec.include:
            if cx.canonical_name not in seen:
                seen[cx.canonical_name] = cx
                order.append(cx.canonical_name)

        # Apply excludes by canonical-name match.
        excluded = {ex.canonical_name for ex in spec.exclude}
        return [seen[k] for k in order if k not in excluded]

    def __iter__(self):
        return iter(self.enumerate())

    def __len__(self) -> int:
        return len(self.enumerate())


# ─── Tube ─────────────────────────────────────────────────────────────────────

@dataclass
class Tube:
    """
    Multi-strand equilibrium "test tube".

    Parameters
    ----------
    strand_totals : dict mapping :class:`Strand` → total concentration (M).
    complexes     : :class:`ComplexSet` defining the species that may form.
    name          : human-readable tube label.

    The strands referenced by every complex in ``complexes`` must appear as
    keys in ``strand_totals``; this is checked at :meth:`analyze` time so that
    constructing a Tube remains cheap.

    Example
    -------
    >>> A = Strand("A", "ACGTACGT")
    >>> B = Strand("B", "ACGTACGT")
    >>> tube = Tube(
    ...     strand_totals={A: 1e-6, B: 1e-6},
    ...     complexes=ComplexSet([A, B], SetSpec(max_size=2)),
    ... )
    """
    strand_totals: dict[Strand, float]
    complexes: ComplexSet
    name: str = "tube"

    def analyze(
        self,
        engine: "ThermoEngine",
        tol: float = 1e-9,
    ) -> "TubeResult":
        """
        Compute equilibrium concentrations for this tube.

        Calls ``engine.pfunc`` on each complex in :attr:`complexes`, then
        invokes :func:`strider.equilibrium.solve_equilibrium`.  Per-complex
        partition functions are read straight from the engine (which has its
        own internal cache — see :class:`strider.thermo.engine.ThermoEngine`).

        Parameters
        ----------
        engine : :class:`ThermoEngine` to evaluate every complex's ΔG.
        tol    : Newton residual tolerance handed to ``solve_equilibrium``.

        Returns
        -------
        TubeResult
        """
        # Validate that every complex's strands are listed in strand_totals.
        known_names = {s.name for s in self.strand_totals}
        complexes = self.complexes.enumerate()
        for cx in complexes:
            for sname in cx.strand_names:
                if sname not in known_names:
                    raise ValueError(
                        f"complex {cx.canonical_name!r} references strand "
                        f"{sname!r} not present in Tube.strand_totals"
                    )

        # Build the (name → (strand_list, ΔG)) map expected by solve_equilibrium.
        cx_map: dict[str, tuple[list[str], float]] = {}
        free_energies: dict[str, float] = {}
        complex_index: dict[str, Complex] = {}

        for cx in complexes:
            dG = float(engine.pfunc(*cx.sequences).free_energy)
            cx_map[cx.canonical_name] = (list(cx.strand_names), dG)
            free_energies[cx.canonical_name] = dG
            complex_index[cx.canonical_name] = cx

        totals = {s.name: float(c) for s, c in self.strand_totals.items()}

        eq: EquilibriumResult = solve_equilibrium(
            complexes=cx_map,
            totals=totals,
            celsius=engine.celsius,
            tol=tol,
            standard_state_M=1.0,
        )

        return TubeResult(
            tube_name=self.name,
            concentrations=dict(eq.concentrations),
            free_energies=free_energies,
            strand_free=dict(eq.strand_free),
            complexes=complex_index,
            converged=eq.converged,
            iterations=eq.iterations,
            residual=eq.residual,
            _engine=engine,
        )


# ─── TubeResult ───────────────────────────────────────────────────────────────

@dataclass
class TubeResult:
    """
    Equilibrium analysis of a :class:`Tube`.

    Attributes
    ----------
    tube_name      : echo of the tube's ``name``.
    concentrations : complex canonical name → equilibrium concentration (M).
    free_energies  : complex canonical name → ensemble ΔG (kcal/mol).
    strand_free    : strand name → free-monomer concentration (M).
    complexes      : complex canonical name → :class:`Complex` object.
    converged      : True iff the Newton solver converged within tolerance.
    iterations     : Newton iterations used.
    residual       : final relative mass-balance residual.

    Pair-probability matrices and ensemble defect are computed lazily through
    :meth:`pair_probabilities` and :meth:`defect` so that tubes can be analyzed
    without paying for matrices the caller never inspects.
    """
    tube_name: str
    concentrations: dict[str, float]
    free_energies: dict[str, float]
    strand_free: dict[str, float]
    complexes: dict[str, Complex]
    converged: bool = False
    iterations: int = 0
    residual: float = 0.0
    _engine: "ThermoEngine | None" = field(default=None, repr=False)

    def pair_probabilities(self, complex_name: str) -> np.ndarray:
        """
        Pair-probability matrix ``P[i,j]`` for the named complex.

        Computed on demand via :meth:`ThermoEngine.pairs`.  Requires the
        result to have been produced by :meth:`Tube.analyze` (so a live
        engine reference is available).
        """
        if self._engine is None:
            raise RuntimeError(
                "TubeResult was constructed without an engine; cannot compute "
                "pair probabilities. Use Tube.analyze() to obtain a result with "
                "engine attached."
            )
        cx = self.complexes[complex_name]
        return self._engine.pairs(*cx.sequences)

    def defect(self, complex_name: str, target_structure: str) -> float:
        """
        Normalized ensemble defect for a target dot-bracket structure on a
        specific complex (Zadeh, Wolfe & Pierce 2011, J. Comput. Chem.
        32:439-452).

        ``target_structure`` may include ``+`` or ``&`` separators between
        strands; they are stripped before scoring.
        """
        if self._engine is None:
            raise RuntimeError("TubeResult missing engine; cannot compute defect")
        cx = self.complexes[complex_name]
        return self._engine.ensemble_defect(cx.sequences, target_structure)

    def tube_ensemble_defect(
        self,
        on_targets: "list[tuple[str, str, float]]",
        normalize: bool = True,
    ) -> float:
        """
        Normalized **test-tube ensemble defect** (Wolfe & Pierce 2015,
        J. Comput. Chem. 36:255-269; Fornace, Porubsky & Pierce 2020,
        ACS Synth. Biol. 9:2665-2678) — the objective NUPACK's ``tube_design``
        minimizes.

        Unlike the per-complex :meth:`defect`, this scores the *whole tube* at
        equilibrium and decomposes into two parts:

        - **structural defect** ``Σ_h c_h · ñ(h, s_h)`` — equilibrium
          concentration of each on-target complex ``h`` times its *unnormalized*
          complex ensemble defect (number of incorrectly paired nucleotides
          relative to target structure ``s_h``);
        - **concentration defect** ``Σ_h |h| · max(0, c_h* − c_h)`` — on-target
          material that failed to form at its target concentration ``c_h*``,
          because the strands were sequestered in off-target complexes.  This is
          how off-targets are penalized: they are *every* complex in the tube
          that is not an on-target, and the equilibrium solve has already
          distributed the strands among them, so no explicit off-target list or
          ΔΔG threshold is needed.

        The sum is normalized by the total on-target nucleotide concentration
        ``Σ_h |h| · c_h*`` so the result lies in ``[0, ~1]`` and is comparable
        across tubes (set ``normalize=False`` for the raw nucleotide count).

        Parameters
        ----------
        on_targets : list of ``(complex_canonical_name, target_dot_bracket,
                     target_concentration_M)``.
        """
        if self._engine is None:
            raise RuntimeError(
                "TubeResult missing engine; cannot compute tube ensemble defect"
            )
        n_struct = 0.0
        n_conc = 0.0
        nt_total = 0.0
        for cx_name, target, c_target in on_targets:
            cx = self.complexes.get(cx_name)
            if cx is None:
                raise KeyError(
                    f"on-target complex {cx_name!r} is not in this tube "
                    f"(have {sorted(self.complexes)})"
                )
            size = cx.total_length
            nt_total += size * max(c_target, 0.0)
            c_eq = self.concentrations.get(cx_name, 0.0)
            if c_eq > 0.0:
                # unnormalized complex defect = incorrectly paired nucleotides
                n_struct += c_eq * self._engine.ensemble_defect(
                    cx.sequences, target, normalize=False
                )
            n_conc += size * max(0.0, c_target - c_eq)
        defect = n_struct + n_conc
        if not normalize:
            return defect
        return defect / nt_total if nt_total > 0.0 else 0.0

    def __repr__(self) -> str:
        return (
            f"TubeResult(tube={self.tube_name!r}, "
            f"n_complexes={len(self.concentrations)}, "
            f"converged={self.converged}, residual={self.residual:.2e})"
        )


# ─── tube_analysis driver ─────────────────────────────────────────────────────

def tube_analysis(
    tubes: Iterable[Tube],
    engine: "ThermoEngine",
    tol: float = 1e-9,
) -> dict[str, TubeResult]:
    """
    Run :meth:`Tube.analyze` on every tube and collect results by tube name.

    Tubes that share complexes benefit automatically from the engine's
    internal pfunc cache, so passing a single engine across tubes is the
    intended usage.

    Parameters
    ----------
    tubes  : iterable of :class:`Tube`.
    engine : :class:`ThermoEngine` used for every pfunc / pairs call.
    tol    : Newton tolerance forwarded to each :meth:`Tube.analyze`.

    Returns
    -------
    dict[str, TubeResult]
        keyed by tube name (raises ``ValueError`` on duplicate names).
    """
    results: dict[str, TubeResult] = {}
    for tube in tubes:
        if tube.name in results:
            raise ValueError(f"duplicate tube name: {tube.name!r}")
        results[tube.name] = tube.analyze(engine, tol=tol)
    return results
