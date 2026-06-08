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
    def reaction_driving_force(
        cls,
        engine: "ThermoEngine",
        reactants: list,
        products: list,
        max_ddg: float,
        assemble_fn: "Callable[[dict[str, str]], dict[str, str]] | None" = None,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Penalize a reaction whose driving force is weaker than a gate.

        This is the design-time mirror of the
        :func:`strider.circuits.checks.reaction_driving_force` *check* — a
        one-sided coupling constraint that keeps a circuit-level driving force
        on-spec while the optimizer tunes a *different* domain.  It generalizes
        urotrace's R2-preservation term (penalize the with-handle branch-
        migration ΔΔG from drifting above its gate while designing the capture
        handle): the constraint couples the designed domain to a reaction it does
        not appear in directly.

        ``reactants`` / ``products`` are lists whose elements are either a strand
        *name* (a key resolved from the assembled sequences) or a list of names
        (a multi-strand complex), matching :meth:`ThermoEngine.ddg`.

        ``assemble_fn`` maps the optimizer's designed-domain dict to the full
        strand dict before name resolution, so the gate is measured on the
        *assembled* context (e.g. the handle attached to H1), not on the bare
        designed domain.  When ``None`` the names are resolved directly from the
        sequences passed to the objective.

        Penalty: ``max(0, ΔΔG_actual − max_ddg)²`` (0 when the gate is met).
        """
        lbl = label or (
            f"reaction_driving_force(≤{max_ddg:.1f})"
        )

        def _resolve(item, strands: dict[str, str]):
            if isinstance(item, str):
                return strands[item]
            return [strands[n] for n in item]

        def fn(seqs: dict[str, str]) -> float:
            strands = assemble_fn(seqs) if assemble_fn is not None else seqs
            try:
                r = [_resolve(x, strands) for x in reactants]
                p = [_resolve(x, strands) for x in products]
            except KeyError:
                return 0.0
            ddg = engine.ddg(r, p)
            return max(0.0, ddg - max_ddg) ** 2

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
    def multitube_defect(
        cls,
        engine: "ThermoEngine",
        tubes: "list[tuple[Callable, list[tuple[str, str, float]]]]",
        tube_weights: "list[float] | None" = None,
        weight: float = 1.0,
        normalize: bool = True,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        **Multistate / multi-tube** ensemble-defect objective — the design
        capability NUPACK exposes as ``tube_design`` (one tube) and multistate
        test-tube design (several tubes); Wolfe, Mirin & Pierce 2017
        (J. Mol. Biol. 429:220-228) and Fornace et al. 2020.

        Each tube is specified as ``(tube_factory, on_targets)`` where

        - ``tube_factory`` is a callable ``(sequences) -> Tube`` that rebuilds a
          fresh :class:`~strider.tube.Tube` from the current sequence assignment
          (so :class:`~strider.tube.Strand` objects pick up the latest sequences
          each evaluation), and
        - ``on_targets`` is a list of ``(complex_canonical_name,
          target_dot_bracket, target_concentration_M)``.

        For every tube the optimiser runs an equilibrium solve and computes the
        normalized :meth:`~strider.tube.TubeResult.tube_ensemble_defect`
        (structural + concentration defect, so off-target formation is penalized
        through the lost on-target material).  The objective is the weighted sum
        of the per-tube defects::

            score = Σ_t  tube_weights[t] · C_tube(t)

        A single-element ``tubes`` list reproduces NUPACK ``tube_design``; the
        existing :meth:`ensemble_defect` (no concentrations) reproduces
        ``complex_design``.

        ``tube_weights`` defaults to all-ones.  A tube whose equilibrium solve
        raises contributes ``+inf`` (an invalid sequence is rejected by the SA
        move), matching the other equilibrium objectives.
        """
        if tube_weights is None:
            tube_weights = [1.0] * len(tubes)
        if len(tube_weights) != len(tubes):
            raise ValueError("tube_weights length must match number of tubes")
        lbl = label or f"multitube_defect({len(tubes)} tube{'s' if len(tubes) != 1 else ''})"

        def fn(seqs: dict[str, str]) -> float:
            total = 0.0
            for (tube_factory, on_targets), tw in zip(tubes, tube_weights):
                try:
                    tube = tube_factory(seqs)
                    result = tube.analyze(engine)
                    total += tw * result.tube_ensemble_defect(
                        on_targets, normalize=normalize
                    )
                except Exception:
                    return float("inf")
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

    # ─── dynamical (closed-loop) objectives ──────────────────────────────────
    #
    # These factories drive sequence optimization from the kinetic / ODE level
    # rather than from static equilibrium defects: ``network_factory`` is a
    # callable ``(sequences) -> mantis.CRNetwork`` (typically a bound method of
    # ``CircuitBridge.to_crnetwork`` or a closure that rebuilds the bridge from
    # the current sequence assignment).  Each call rebuilds the network so the
    # rate constants reflect the latest sequences, then runs a mantis simulation
    # or bifurcation analysis to derive the cost.

    @classmethod
    def kinetic_trajectory(
        cls,
        network_factory: Callable[[dict[str, str]], object],
        initial_conditions: dict[str, float],
        target_curve: dict[str, "object"],
        times: "object",
        weight: float = 1.0,
        normalize: bool = True,
        label: str = "kinetic_trajectory",
    ) -> "DesignObjective":
        """
        Penalize the squared error between a simulated kinetic trajectory and
        a user-specified target curve.

        ``target_curve`` maps species name → 1-D array of target concentrations
        sampled at ``times``.  The simulator is evaluated at the same times,
        and the per-species MSE is averaged.  When ``normalize=True`` each
        species term is divided by ``max(|target|)²`` so the cost is
        dimensionless and species with very different scales contribute
        comparably.
        """
        import numpy as np

        t_eval = np.asarray(times, dtype=float)
        t_span = (float(t_eval[0]), float(t_eval[-1]))
        target = {sp: np.asarray(arr, dtype=float) for sp, arr in target_curve.items()}

        def fn(seqs: dict[str, str]) -> float:
            try:
                net = network_factory(seqs)
                sim = net.simulate(initial_conditions, t_span=t_span, t_eval=t_eval)
            except Exception:
                return float("inf")
            if not getattr(sim, "success", False):
                return float("inf")
            total = 0.0
            count = 0
            for sp, tgt in target.items():
                arr = sim.concentrations.get(sp)
                if arr is None:
                    continue
                diff = arr[: len(tgt)] - tgt[: len(arr)]
                if normalize:
                    scale = float(np.max(np.abs(tgt))) or 1.0
                    total += float(np.mean((diff / scale) ** 2))
                else:
                    total += float(np.mean(diff ** 2))
                count += 1
            return total / count if count else 0.0

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [label]
        return obj

    @classmethod
    def maximize_kcat(
        cls,
        network_factory: Callable[[dict[str, str]], object],
        species: str,
        initial_conditions: dict[str, float],
        t_window: tuple[float, float],
        weight: float = 1.0,
        scale: float | None = None,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Reward fast accumulation of ``species`` over ``t_window``.

        Score = ``-Δ[species] / Δt`` (in M/s), so lower (more negative) is
        better — i.e. the optimizer drives the catalytic production rate up.
        ``scale`` divides the rate to keep the term comparable to other
        squared-error terms (default: max IC concentration in ``initial_conditions``
        divided by the window length, so the magnitude is O(1) when the species
        accumulates a meaningful fraction of the input over the window).
        """
        import numpy as np

        lbl = label or f"maximize_kcat({species})"
        t0, tf = float(t_window[0]), float(t_window[1])
        dt = max(tf - t0, 1e-30)
        if scale is None:
            ic_max = max((abs(v) for v in initial_conditions.values()), default=1.0)
            scale = max(ic_max / dt, 1e-30)

        def fn(seqs: dict[str, str]) -> float:
            try:
                net = network_factory(seqs)
                sim = net.simulate(initial_conditions, t_span=(t0, tf))
            except Exception:
                return float("inf")
            if not getattr(sim, "success", False):
                return float("inf")
            arr = sim.concentrations.get(species)
            if arr is None or len(arr) < 2:
                return 0.0
            rate = float(arr[-1] - arr[0]) / dt
            return -rate / float(scale)

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def minimize_leak(
        cls,
        network_factory: Callable[[dict[str, str]], object],
        signal_species: str,
        initial_conditions_no_trigger: dict[str, float],
        t_window: tuple[float, float],
        threshold: float = 1e-9,
        weight: float = 1.0,
        label: str = "minimize_leak",
    ) -> "DesignObjective":
        """
        Penalize spontaneous accumulation of ``signal_species`` in a control
        simulation where the trigger / input is set to zero (or to whatever the
        caller chooses to leave in ``initial_conditions_no_trigger``).

        Score is ``(log10(leak / threshold))²`` when leak > threshold, and 0
        otherwise — so the gradient grows logarithmically and is well-behaved
        across many decades of leak rate (the regime the
        outperform_nupack.md spec calls out: leak < 1e-6 M⁻¹s⁻¹).
        """
        t0, tf = float(t_window[0]), float(t_window[1])

        def fn(seqs: dict[str, str]) -> float:
            try:
                net = network_factory(seqs)
                sim = net.simulate(initial_conditions_no_trigger, t_span=(t0, tf))
            except Exception:
                return float("inf")
            if not getattr(sim, "success", False):
                return float("inf")
            arr = sim.concentrations.get(signal_species)
            if arr is None or len(arr) == 0:
                return 0.0
            leaked = float(arr[-1])
            if leaked <= threshold:
                return 0.0
            return math.log10(leaked / threshold) ** 2

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [label]
        return obj

    @classmethod
    def bistable_threshold(
        cls,
        network_factory: Callable[[dict[str, str]], object],
        parameter: str,
        param_range: tuple[float, float],
        species: str,
        target_threshold: float,
        initial_conditions: dict[str, float],
        n_points: int = 21,
        weight: float = 1.0,
        label: str | None = None,
    ) -> "DesignObjective":
        """
        Locate the bistable switching threshold along ``parameter`` and
        penalize its log-deviation from ``target_threshold``.

        The bifurcation is scanned over the log-spaced ``param_range``; the
        stable-branch maximum of ``species`` is tracked, and the threshold is
        identified as the first parameter value where that maximum rises above
        the midpoint between the branch's min and max.  Returns ``inf`` if no
        crossing is found (i.e. no bistability across the scanned range).
        """
        import numpy as np

        lbl = label or f"bistable_threshold({parameter}→{target_threshold:.2g})"

        def fn(seqs: dict[str, str]) -> float:
            try:
                net = network_factory(seqs)
                br = net.bifurcation(
                    parameter, param_range, n_points=n_points,
                    initial_conditions=initial_conditions,
                )
            except Exception:
                return float("inf")
            params = np.asarray(br.parameter_values, dtype=float)
            values = np.full(len(params), np.nan)
            for i, ss_list in enumerate(br.steady_states):
                stable = [s for s in ss_list if s.is_stable]
                if not stable:
                    continue
                values[i] = max(s.concentrations.get(species, 0.0) for s in stable)
            if np.all(np.isnan(values)):
                return float("inf")
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            if hi - lo <= 0.0:
                return float("inf")
            mid = (lo + hi) / 2.0
            crossing = np.where(values >= mid)[0]
            if len(crossing) == 0:
                return float("inf")
            threshold_actual = float(params[crossing[0]])
            if threshold_actual <= 0.0 or target_threshold <= 0.0:
                return float("inf")
            return math.log10(threshold_actual / target_threshold) ** 2

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [lbl]
        return obj

    @classmethod
    def from_simulation(
        cls,
        network_factory: Callable[[dict[str, str]], object],
        initial_conditions: dict[str, float],
        t_span: tuple[float, float],
        cost_fn: Callable[[object], float],
        weight: float = 1.0,
        label: str = "from_simulation",
    ) -> "DesignObjective":
        """
        Escape hatch for arbitrary kinetic cost functions.

        ``cost_fn`` receives the :class:`mantis.SimulationResult` and returns a
        scalar.  Returns ``inf`` if the simulation fails so the optimizer
        rejects the candidate cleanly.
        """
        def fn(seqs: dict[str, str]) -> float:
            try:
                net = network_factory(seqs)
                sim = net.simulate(initial_conditions, t_span=t_span)
            except Exception:
                return float("inf")
            if not getattr(sim, "success", False):
                return float("inf")
            try:
                return float(cost_fn(sim))
            except Exception:
                return float("inf")

        obj = cls()
        obj._terms = [(weight, fn)]
        obj._labels = [label]
        return obj

    def __repr__(self) -> str:
        return f"DesignObjective({', '.join(self._labels)})"
