"""Tests for the generalized design patterns lifted from urotrace:
DesignObjective.reaction_driving_force (coupling constraint) and
design_with_rerank (gate-with-downstream re-rank)."""
import pytest

from strider import (
    DesignObjective, SequenceDesigner, DomainSpec, design_with_rerank,
    reverse_complement,
)
from strider.thermo.engine import ThermoEngine


@pytest.fixture
def engine():
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137,
                        magnesium=0.01, backend="native")


# ─── reaction_driving_force objective ──────────────────────────────────────────

class TestReactionDrivingForce:
    def test_zero_when_gate_met(self, engine):
        A = "GCGCGCGCGCGC"
        B = reverse_complement(A)               # strong duplex, ΔΔG ≪ -3
        obj = DesignObjective.reaction_driving_force(
            engine, ["A", "B"], [["A", "B"]], max_ddg=-3.0)
        assert obj({"A": A, "B": B}) == pytest.approx(0.0, abs=1e-9)

    def test_penalized_when_too_weak(self, engine):
        # poly-A strands barely interact ⇒ ΔΔG ≈ 0 > -3 ⇒ penalty > 0
        obj = DesignObjective.reaction_driving_force(
            engine, ["A", "B"], [["A", "B"]], max_ddg=-3.0)
        pen = obj({"A": "AAAAAAAA", "B": "AAAAAAAA"})
        assert pen > 0.0

    def test_assemble_fn_resolves_context(self, engine):
        # The designed domain ("K") is not a reactant; assemble_fn builds the
        # strands the gate is measured on — the coupling-constraint pattern.
        A = "GCGCGCGCGCGC"

        def assemble_fn(seqs):
            return {"A": A, "B": reverse_complement(A), "K": seqs["K"]}

        obj = DesignObjective.reaction_driving_force(
            engine, ["A", "B"], [["A", "B"]], max_ddg=-3.0,
            assemble_fn=assemble_fn)
        # K is irrelevant to A·B; gate is met ⇒ 0 regardless of K
        assert obj({"K": "ACGTACGT"}) == pytest.approx(0.0, abs=1e-9)

    def test_composes_with_other_objectives(self, engine):
        base = DesignObjective.gc_content("A", target_gc=0.5)
        coupled = base + DesignObjective.reaction_driving_force(
            engine, ["A", "B"], [["A", "B"]], max_ddg=-3.0, weight=2.0)
        seqs = {"A": "AAAAAAAA", "B": "AAAAAAAA"}
        # total = gc term + 2× driving-force penalty
        assert coupled(seqs) == pytest.approx(
            base(seqs) + 2.0 * DesignObjective.reaction_driving_force(
                engine, ["A", "B"], [["A", "B"]], max_ddg=-3.0)(seqs))


# ─── design_with_rerank helper ─────────────────────────────────────────────────

class TestDesignWithRerank:
    def _trivial_problem(self, _ctx):
        return {"domains": {"X": DomainSpec(length=6, material="dna")},
                "objective": DesignObjective.from_callable(lambda s: 0.0,
                                                           label="noop")}

    def test_picks_best_on_downstream_score(self, engine):
        designer = SequenceDesigner(engine=engine, seed=0)
        contexts = ["ctxA", "ctxB"]              # pre-ranked order: A first
        # downstream gate says B is better, even though A was pre-ranked first
        scores = {"ctxA": 5.0, "ctxB": 1.0}
        rr = design_with_rerank(
            designer, contexts, self._trivial_problem,
            lambda ctx, result: scores[ctx],
            top_n=2, n_trials=1, max_iterations=1)
        assert rr.context == "ctxB"
        assert rr.score == 1.0
        assert len(rr.all_scores) == 2

    def test_top_n_limits_candidates_designed(self, engine):
        designer = SequenceDesigner(engine=engine, seed=0)
        contexts = ["ctxA", "ctxB", "ctxC"]
        rr = design_with_rerank(
            designer, contexts, self._trivial_problem,
            lambda ctx, result: {"ctxA": 2.0, "ctxB": 3.0, "ctxC": 1.0}[ctx],
            top_n=1, n_trials=1, max_iterations=1)
        # only the first pre-ranked context is designed/scored
        assert len(rr.all_scores) == 1
        assert rr.context == "ctxA"

    def test_empty_contexts_raise(self, engine):
        designer = SequenceDesigner(engine=engine, seed=0)
        with pytest.raises(ValueError):
            design_with_rerank(designer, [], self._trivial_problem,
                               lambda c, r: 0.0)
