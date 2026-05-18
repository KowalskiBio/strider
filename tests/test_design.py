"""Sequence design and mutation analyzer tests."""
import pytest
from strider.design.optimizer import SequenceDesigner, DomainSpec
from strider.design.objective import DesignObjective
from strider.design.constraints import HardConstraint
from strider.design.mutation import MutationAnalyzer


@pytest.fixture
def engine():
    from strider.thermo.engine import ThermoEngine
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


class TestDesignObjective:
    def test_gc_content_objective(self, engine):
        obj = DesignObjective.gc_content("seq", target_gc=0.5)
        # Perfect GC content = 0 penalty
        score = obj({"seq": "ACGT"})
        assert abs(score) < 1e-6

        # All AT = penalty
        score_at = obj({"seq": "AAAA"})
        assert score_at > 0

    def test_objective_composition(self, engine):
        obj1 = DesignObjective.gc_content("seq", target_gc=0.5, weight=1.0)
        obj2 = DesignObjective.gc_content("seq", target_gc=0.5, weight=2.0)
        combined = obj1 + obj2
        score1 = obj1({"seq": "AAAA"})
        score_combined = combined({"seq": "AAAA"})
        assert abs(score_combined - 3 * score1) < 1e-6

    def test_scalar_multiplication(self, engine):
        obj = DesignObjective.gc_content("seq", target_gc=0.5, weight=1.0)
        obj2 = 3.0 * obj
        s1 = obj({"seq": "AAAA"})
        s2 = obj2({"seq": "AAAA"})
        assert abs(s2 - 3 * s1) < 1e-6

    def test_from_callable(self):
        def my_fn(seqs): return len(seqs.get("seq", "")) ** 2
        obj = DesignObjective.from_callable(my_fn, label="len_sq")
        score = obj({"seq": "ACGT"})
        assert score == 16

    def test_evaluate_breakdown(self, engine):
        obj = (
            DesignObjective.gc_content("seq", weight=1.0, label="gc")
            + DesignObjective.gc_content("seq", weight=2.0, label="gc2")
        )
        breakdown = obj.evaluate_breakdown({"seq": "AAAA"})
        assert "gc" in breakdown
        assert "gc2" in breakdown


class TestEnsembleDefect:
    def test_defect_in_unit_interval(self, engine):
        seq = "GCGCAAAAGCGC"
        target = "((((....))))"
        d = engine.ensemble_defect(seq, target)
        assert 0.0 <= d <= 1.0

    def test_hairpin_target_beats_random_for_stable_seq(self, engine):
        # A GC-rich palindrome should match the hairpin target better
        # than an unstructured (all-unpaired) target.
        seq = "GCGCGCAAAAGCGCGC"
        good = engine.ensemble_defect(seq, "((((((....))))))")
        bad = engine.ensemble_defect(seq, "................")
        assert good < bad

    def test_length_mismatch_raises(self, engine):
        with pytest.raises(ValueError):
            engine.ensemble_defect("ACGT", "(.)")

    def test_objective_factory(self, engine):
        obj = DesignObjective.ensemble_defect(
            engine, "H", "((((....))))", weight=1.0
        )
        score = obj({"H": "GCGCAAAAGCGC"})
        assert 0.0 <= score <= 1.0


class TestHardConstraint:
    def test_no_repeats(self):
        c = HardConstraint.no_repeats(["CCCC", "AAAA"])
        assert c.check("seq", "ACGTACGT")
        assert not c.check("seq", "AAAACCCC")

    def test_gc_content(self):
        c = HardConstraint.gc_content(min_gc=0.4, max_gc=0.6)
        assert c.check("seq", "ACGT")   # 50% GC
        assert not c.check("seq", "AAAA")  # 0% GC

    def test_max_run(self):
        c = HardConstraint.max_run(max_run_length=3)
        assert c.check("seq", "ACGTACGT")
        assert not c.check("seq", "AAAAA")

    def test_apply_to_filter(self):
        c = HardConstraint.no_repeats(["AAAA"], apply_to=["H1"])
        assert c.check("H2", "AAAA")  # not applied
        assert not c.check("H1", "AAAA")


class TestSequenceDesigner:
    def test_returns_design_result(self, engine):
        obj = DesignObjective.gc_content("D", target_gc=0.5)
        designer = SequenceDesigner(engine, seed=42)
        result = designer.design(
            domains={"D": DomainSpec(length=8)},
            objective=obj,
            n_trials=2,
            max_iterations=50,
        )
        assert "D" in result.sequences
        assert len(result.sequences["D"]) == 8

    def test_fixed_domain_unchanged(self, engine):
        obj = DesignObjective.gc_content("free", target_gc=0.5)
        designer = SequenceDesigner(engine, seed=1)
        result = designer.design(
            domains={
                "fixed": DomainSpec(sequence="ACGTACGT"),
                "free": DomainSpec(length=8),
            },
            objective=obj,
            n_trials=2,
            max_iterations=30,
        )
        assert result.sequences["fixed"] == "ACGTACGT"

    def test_hard_constraints_satisfied(self, engine):
        obj = DesignObjective.gc_content("D", target_gc=0.5)
        c = HardConstraint.no_repeats(["AAAA", "CCCC", "GGGG", "TTTT"])
        designer = SequenceDesigner(engine, seed=7)
        result = designer.design(
            domains={"D": DomainSpec(length=12)},
            objective=obj,
            hard_constraints=[c],
            n_trials=3,
            max_iterations=100,
        )
        seq = result.sequences["D"]
        for repeat in ["AAAA", "CCCC", "GGGG", "TTTT"]:
            assert repeat not in seq, f"Repeat {repeat} found in {seq}"


class TestMutationAnalyzer:
    def test_returns_profile(self, engine):
        analyzer = MutationAnalyzer(engine)
        profile = analyzer.single_nt_scan("GCATGCATGC")
        assert profile.delta_score.shape == (10, 3)
        assert len(profile.alt_nucleotides) == 10

    def test_robustness_between_0_and_1(self, engine):
        analyzer = MutationAnalyzer(engine)
        r = analyzer.robustness_score("GCGCGCGC", ddg_tolerance=2.0)
        assert 0.0 <= r <= 1.0

    def test_critical_positions_subset_of_all(self, engine):
        analyzer = MutationAnalyzer(engine)
        profile = analyzer.single_nt_scan("GCATGCATGC")
        critical = profile.critical_positions(threshold=0.1)
        assert all(0 <= p < 10 for p in critical)
