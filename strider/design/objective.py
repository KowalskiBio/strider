"""
Composable design objectives for nucleic acid sequence optimization.

Objectives are callables: (sequences: dict[str, str]) -> float (lower = better).
They compose via addition and scalar multiplication.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine


@dataclass
class DesignObjective:
    """
    A weighted, composable objective for sequence design.

    Objectives are summed: total_score = Σ weight_i * term_i(sequences).
    Lower total_score is better (minimization problem).
    """
    _terms: list[tuple[float, Callable[[dict[str, str]], float]]] = field(
        default_factory=list, repr=False
    )
    _labels: list[str] = field(default_factory=list, repr=False)

    def __call__(self, sequences: dict[str, str]) -> float:
        """Evaluate the total weighted objective score for a given sequence dict."""
        return sum(w * fn(sequences) for w, fn in self._terms)

    def evaluate_breakdown(self, sequences: dict[str, str]) -> dict[str, float]:
        """Return a per-term score dict keyed by label for diagnostic inspection."""
        return {
            label: w * fn(sequences)
            for label, (w, fn) in zip(self._labels, self._terms)
        }

    def __add__(self, other: "DesignObjective") -> "DesignObjective":
        """Combine two objectives by summing their terms."""
        obj = DesignObjective()
        obj._terms = self._terms + other._terms
        obj._labels = self._labels + other._labels
        return obj

    def __mul__(self, weight: float) -> "DesignObjective":
        """Scale all objective term weights by a scalar factor."""
        obj = DesignObjective()
        obj._terms = [(w * weight, fn) for w, fn in self._terms]
        obj._labels = self._labels[:]
        return obj

    def __rmul__(self, weight: float) -> "DesignObjective":
        """Support scalar * objective syntax."""
        return self.__mul__(weight)

    # ─── factory methods ─────────────────────────────────────────────────────

    @classmethod
    def ddg_target(
        cls,
        engine: "ThermoEngine",
        reactants: list[str],
        products: list[str],
        target: float,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Penalize (ΔΔG_actual - target)^2.

        reactants / products: domain name strings resolved from the sequences dict.
        """
        lbl = label or f"ddg_target({'+'.join(reactants)}→{'+'.join(products)}, {target:.1f})"

        def fn(seqs: dict[str, str]) -> float:
            r = [seqs[n] for n in reactants if n in seqs]
            p = [seqs[n] for n in products if n in seqs]
            if not r or not p:
                return 0.0
            ddg = engine.ddg(r, p)
            return (ddg - target) ** 2

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def minimize_leakage(
        cls,
        engine: "ThermoEngine",
        strand_names: list[str],
        threshold: float = -4.0,
        weight: float = 1.0,
        label: str = "minimize_leakage",
    ) -> "DesignObjective":
        """
        Penalize spurious pairwise ΔΔG values below threshold.

        Adds (threshold - ΔΔG)^2 for each pair below threshold.
        """
        from itertools import combinations

        def fn(seqs: dict[str, str]) -> float:
            total = 0.0
            names = [n for n in strand_names if n in seqs]
            for n1, n2 in combinations(names, 2):
                s1, s2 = seqs[n1], seqs[n2]
                try:
                    ddg = engine.ddg([s1, s2], [[s1, s2]])
                except Exception:
                    continue
                if ddg < threshold:
                    total += (threshold - ddg) ** 2
            return total

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [label]
        return obj

    @classmethod
    def toehold_accessible(
        cls,
        engine: "ThermoEngine",
        strand_name: str,
        positions: list[int],
        min_prob: float = 0.8,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Penalize low toehold accessibility.

        Score = max(0, min_prob - P_accessible)^2
        """
        lbl = label or f"toehold_accessible({strand_name}, pos={positions[:3]}...)"

        def fn(seqs: dict[str, str]) -> float:
            if strand_name not in seqs:
                return 0.0
            prob = engine.toehold_accessibility(seqs[strand_name], positions)
            shortfall = max(0.0, min_prob - prob)
            return shortfall ** 2

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def gc_content(
        cls,
        strand_name: str,
        target_gc: float = 0.5,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """Penalize deviation from target GC content (no engine needed)."""
        lbl = label or f"gc_content({strand_name}, {target_gc:.0%})"

        def fn(seqs: dict[str, str]) -> float:
            seq = seqs.get(strand_name, "")
            if not seq:
                return 0.0
            gc = sum(1 for b in seq.upper() if b in "GC") / len(seq)
            return (gc - target_gc) ** 2

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def ddg_range(
        cls,
        engine: "ThermoEngine",
        reactants: list[str],
        products: list[str],
        min_ddg: float,
        max_ddg: float,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """Penalize ΔΔG outside [min_ddg, max_ddg]."""
        lbl = label or f"ddg_range({min_ddg:.1f},{max_ddg:.1f})"

        def fn(seqs: dict[str, str]) -> float:
            r = [seqs[n] for n in reactants if n in seqs]
            p = [seqs[n] for n in products if n in seqs]
            if not r or not p:
                return 0.0
            ddg = engine.ddg(r, p)
            if ddg < min_ddg:
                return (min_ddg - ddg) ** 2
            if ddg > max_ddg:
                return (ddg - max_ddg) ** 2
            return 0.0

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def ensemble_defect(
        cls,
        engine: "ThermoEngine",
        strand_names: str | list[str],
        target_structure: str,
        weight: float = 1.0,
        normalize: bool = True,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Penalize the normalized ensemble defect of a target dot-bracket
        (Zadeh et al. 2011, J. Comput. Chem. 32:439-452).

        ``strand_names`` is either a single domain name or a list of names whose
        sequences are concatenated (in order) to form the complex.  The target
        structure must match the total length and may use ``&``/``+`` separators
        for readability (they are stripped before scoring).
        """
        names = [strand_names] if isinstance(strand_names, str) else list(strand_names)
        lbl = label or f"ensemble_defect({'+'.join(names)})"

        def fn(seqs: dict[str, str]) -> float:
            try:
                strands = tuple(seqs[n] for n in names if n in seqs)
            except KeyError:
                return 0.0
            if not strands or sum(len(s) for s in strands) == 0:
                return 0.0
            try:
                return engine.ensemble_defect(strands, target_structure, normalize=normalize)
            except ValueError:
                return float("inf")

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def ensemble_defect_tube(
        cls,
        engine: "ThermoEngine",
        tube_factory,
        on_targets: list[tuple[str, str]],
        weight: float = 1.0,
        normalize: bool = True,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Equilibrium-weighted ensemble-defect objective.

        ``tube_factory`` is a callable ``(sequences) -> Tube`` that builds
        a fresh :class:`~strider.tube.Tube` from the current sequence
        assignment (so :class:`~strider.tube.Strand` objects pick up the
        latest sequences each call).  ``on_targets`` is a list of
        ``(complex_canonical_name, target_dot_bracket)`` pairs.  Each
        on-target contributes ``c_eq · defect(target)`` to the score
        where ``c_eq`` is the equilibrium concentration of that complex
        in the tube — so the optimiser sees the *true* weighting from
        the equilibrium solve, not a declared design-time concentration
        (Wolfe & Pierce 2015, J. Comput. Chem. 36:255-269 §2.2).
        """
        lbl = label or f"ensemble_defect_tube({len(on_targets)})"

        def fn(seqs: dict[str, str]) -> float:
            try:
                tube = tube_factory(seqs)
                result = tube.analyze(engine)
            except Exception:
                return float("inf")
            total = 0.0
            for cx_name, target in on_targets:
                conc = result.concentrations.get(cx_name, 0.0)
                if conc <= 0.0:
                    continue
                try:
                    d = result.defect(cx_name, target)
                except Exception:
                    continue
                total += conc * (d if normalize else d * len(target.replace("&", "").replace("+", "")))
            return total

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[dict[str, str]], float],
        weight: float = 1.0,
        label: str = "custom",
    ) -> "DesignObjective":
        """Wrap any Python function as an objective."""
        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [label]
        return obj

    def __repr__(self) -> str:
        return f"DesignObjective({', '.join(self._labels)})"
