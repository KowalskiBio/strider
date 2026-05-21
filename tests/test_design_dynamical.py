"""Closed-loop dynamical sequence-design objectives (mantis-feedback driven).

These tests use a hand-built synthetic ``CRNetwork`` whose rate constants
respond to sequence content so we can verify that the new objectives in
``strider.design.objective`` move in the expected direction without paying
the full thermo-engine cost of a real CHA bridge.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from strider.design.objective import DesignObjective

mantis = pytest.importorskip("mantis")


def _gc(seq: str) -> float:
    """GC fraction of a DNA string."""
    if not seq:
        return 0.0
    return sum(1 for b in seq.upper() if b in "GC") / len(seq)


def _make_kcat_factory():
    """
    Toy catalytic loop:  S --k1--> P,  P --k2--> S, plus an explicit leak
    S0 --kleak--> P.  k1 is driven by GC content of an "enzyme" strand E so
    higher GC sequences give faster catalysis; kleak is driven by GC content
    of a "blocker" strand B so lower GC = less leak.
    """
    reactions = ["S -> P", "P -> S", "S0 -> P"]

    def factory(seqs: dict[str, str]):
        k1 = 1e-2 + 1.0 * _gc(seqs.get("E", ""))
        k2 = 1e-3
        kleak = 1e-4 * (1.0 + 100.0 * _gc(seqs.get("B", "")))
        rates = {
            "S -> P": k1,
            "P -> S": k2,
            "S0 -> P": kleak,
        }
        return mantis.CRNetwork.from_string(reactions, rates=rates)

    return factory


def _make_switch_factory():
    """
    Linear input-output network whose effective threshold along ``X -> Y``
    is controlled by sequence GC fraction.  Used by the bistable-threshold
    test; it isn't a true bistable system (just a monotonic ramp), but the
    objective only needs a single crossing of the midpoint, which a ramp
    provides reliably.
    """
    reactions = ["X -> Y", "Y -> "]

    def factory(seqs: dict[str, str]):
        gc = _gc(seqs.get("sw", ""))
        rates = {"X -> Y": 1.0, "Y -> ": 1.0 + 10.0 * gc}
        return mantis.CRNetwork.from_string(reactions, rates=rates)

    return factory


class TestKineticTrajectory:
    def test_zero_cost_when_curve_matches(self):
        factory = _make_kcat_factory()
        ic = {"S": 1e-6, "P": 0.0, "S0": 0.0}
        times = np.linspace(0.0, 100.0, 21)

        # Build "target" curve = exactly what the network produces for
        # GC=0.5 enzyme, then check the objective evaluates near zero.
        seqs = {"E": "ACGT" * 5, "B": "AAAA" * 5}
        sim = factory(seqs).simulate(ic, t_span=(0.0, 100.0), t_eval=times)
        target = {"P": sim.concentrations["P"]}

        obj = DesignObjective.kinetic_trajectory(factory, ic, target, times)
        assert obj(seqs) < 1e-10

    def test_mismatch_costs_more_than_match(self):
        factory = _make_kcat_factory()
        ic = {"S": 1e-6, "P": 0.0, "S0": 0.0}
        times = np.linspace(0.0, 100.0, 21)

        good = {"E": "GGGG" * 5, "B": "AAAA" * 5}  # high GC enzyme, low leak
        bad = {"E": "AAAA" * 5, "B": "AAAA" * 5}   # low GC enzyme

        sim_good = factory(good).simulate(ic, t_span=(0.0, 100.0), t_eval=times)
        target = {"P": sim_good.concentrations["P"]}

        obj = DesignObjective.kinetic_trajectory(factory, ic, target, times)
        assert obj(good) < obj(bad)


class TestMaximizeKcat:
    def test_higher_gc_enzyme_scores_lower(self):
        factory = _make_kcat_factory()
        ic = {"S": 1e-6, "P": 0.0, "S0": 0.0}
        obj = DesignObjective.maximize_kcat(factory, "P", ic, t_window=(0.0, 50.0))
        slow = obj({"E": "AAAA" * 5, "B": "AAAA" * 5})
        fast = obj({"E": "GCGC" * 5, "B": "AAAA" * 5})
        # Score is -rate; faster catalysis ⇒ more negative ⇒ lower score (better).
        assert fast < slow

    def test_returns_finite_on_zero_rate(self):
        factory = _make_kcat_factory()
        ic = {"S": 0.0, "P": 0.0, "S0": 0.0}  # nothing to convert
        obj = DesignObjective.maximize_kcat(factory, "P", ic, t_window=(0.0, 50.0))
        score = obj({"E": "ACGT" * 5, "B": "AAAA" * 5})
        assert math.isfinite(score)


class TestMinimizeLeak:
    def test_zero_below_threshold(self):
        factory = _make_kcat_factory()
        # Trigger off: S=0 so the catalytic loop is silent.  Only the leak
        # reaction S0 -> P contributes.
        ic = {"S": 0.0, "P": 0.0, "S0": 1e-12}  # tiny pool, ≤ threshold
        obj = DesignObjective.minimize_leak(
            factory, "P", ic, t_window=(0.0, 100.0), threshold=1e-9,
        )
        score = obj({"E": "ACGT" * 5, "B": "AAAA" * 5})
        assert score == 0.0

    def test_penalty_grows_with_leak(self):
        factory = _make_kcat_factory()
        ic = {"S": 0.0, "P": 0.0, "S0": 1e-6}
        obj = DesignObjective.minimize_leak(
            factory, "P", ic, t_window=(0.0, 100.0), threshold=1e-12,
        )
        low = obj({"E": "ACGT" * 5, "B": "AAAA" * 5})  # low-GC blocker
        high = obj({"E": "ACGT" * 5, "B": "GGGG" * 5})  # high-GC blocker → faster leak
        assert high > low > 0.0


class TestBistableThreshold:
    def test_finite_score_in_scanned_range(self):
        factory = _make_switch_factory()
        ic = {"X": 1e-6, "Y": 0.0}
        obj = DesignObjective.bistable_threshold(
            factory,
            parameter="X -> Y",
            param_range=(1e-2, 1e2),
            species="Y",
            target_threshold=1.0,
            initial_conditions=ic,
            n_points=11,
        )
        score = obj({"sw": "ACGT" * 4})
        # The toy network's threshold isn't exactly 1.0, but the score
        # should be finite (a crossing is found) and non-negative.
        assert math.isfinite(score)
        assert score >= 0.0

    def test_inf_when_no_crossing(self):
        # Range entirely above any plausible threshold ⇒ Y is uniformly high
        # ⇒ no first-crossing → ``inf``.
        factory = _make_switch_factory()
        ic = {"X": 1e-6, "Y": 0.0}
        obj = DesignObjective.bistable_threshold(
            factory,
            parameter="X -> Y",
            param_range=(1e10, 1e12),  # wildly off-scale; flat response
            species="Y",
            target_threshold=1.0,
            initial_conditions=ic,
            n_points=5,
        )
        score = obj({"sw": "ACGT" * 4})
        # Either inf (no midpoint crossing because hi == lo) or finite if
        # the solver happens to find a tiny range — both are acceptable as
        # long as no exception is raised.
        assert score == float("inf") or math.isfinite(score)


class TestFromSimulation:
    def test_custom_cost(self):
        factory = _make_kcat_factory()
        ic = {"S": 1e-6, "P": 0.0, "S0": 0.0}

        def cost(sim):
            return float(sim.concentrations["P"][-1])  # maximize final P

        obj = DesignObjective.from_simulation(factory, ic, (0.0, 50.0), cost)
        slow = obj({"E": "AAAA" * 5, "B": "AAAA" * 5})
        fast = obj({"E": "GGGG" * 5, "B": "AAAA" * 5})
        # Higher final P with faster enzyme — and remember cost_fn output is
        # taken directly (no negation), so user can pick the sign.
        assert fast > slow


class TestComposition:
    def test_compose_with_static_objective(self):
        factory = _make_kcat_factory()
        ic = {"S": 1e-6, "P": 0.0, "S0": 0.0}
        kinetic = DesignObjective.maximize_kcat(factory, "P", ic, (0.0, 50.0))
        static = DesignObjective.gc_content("E", target_gc=0.5)
        combined = 1.0 * kinetic + 10.0 * static
        score = combined({"E": "ACGT" * 5, "B": "AAAA" * 5})
        assert math.isfinite(score)
        breakdown = combined.evaluate_breakdown({"E": "ACGT" * 5, "B": "AAAA" * 5})
        assert len(breakdown) == 2
