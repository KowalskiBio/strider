"""Assay / AssayPanel design-abstraction tests."""
import pytest

from strider import Assay, AssayPanel, Assembly, DomainSpec, SequenceDesigner
from strider.thermo.engine import ThermoEngine


@pytest.fixture
def engine():
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


class TestAssayDefect:
    def test_on_target_defect_returns_positive(self, engine):
        assay = Assay(
            name="t",
            on_targets=[Assembly("H", ["H"], "((((....))))", concentration=1.0)],
        )
        d = assay.defect({"H": "GCGCAAAAGCGC"}, engine)
        assert d > 0.0

    def test_better_target_has_lower_defect(self, engine):
        """A sequence matching the target should beat a random one."""
        assay = Assay(
            name="t",
            on_targets=[Assembly("H", ["H"], "((((....))))", concentration=1.0)],
        )
        good = assay.defect({"H": "GCGCAAAAGCGC"}, engine)
        bad  = assay.defect({"H": "AAAATTTTAAAA"}, engine)
        assert good < bad

    def test_concentration_weights_defect(self, engine):
        target = Assembly("H", ["H"], "((((....))))", concentration=2.0)
        assay = Assay(name="t", on_targets=[target])
        assert assay.defect({"H": "GCGCAAAAGCGC"}, engine) == pytest.approx(
            2.0 * engine.ensemble_defect("GCGCAAAAGCGC", "((((....))))"),
            rel=1e-9,
        )

    def test_off_target_penalty_for_strong_dimer(self, engine):
        """A spontaneous AB dimer with ΔΔG below the threshold should incur a penalty."""
        assay = Assay(
            name="t",
            on_targets=[],
            off_targets=[Assembly("AB", ["A", "B"])],
            off_target_ddg_threshold=-4.0,
            off_target_penalty_weight=1.0,
        )
        # GC-rich complementary strands → strong binding → penalty > 0
        strong = assay.defect({"A": "GCGCGCGCGC", "B": "GCGCGCGCGC"}, engine)
        # Weak: incompatible sequences
        weak = assay.defect({"A": "AAAAAAAAAA", "B": "AAAAAAAAAA"}, engine)
        assert strong > weak

    def test_missing_strand_skipped(self, engine):
        assay = Assay(
            name="t",
            on_targets=[Assembly("X", ["missing"], "((....))")],
        )
        # Should silently skip rather than crash
        assert assay.defect({"H": "ACGT"}, engine) == 0.0


class TestAssayPanel:
    def test_sum_across_assays(self, engine):
        a1 = Assay("a1", on_targets=[
            Assembly("H", ["H"], "((((....))))", concentration=1.0)])
        a2 = Assay("a2", on_targets=[
            Assembly("H", ["H"], "((((....))))", concentration=1.0)])
        panel = AssayPanel(assays=[a1, a2])
        single = a1.defect({"H": "GCGCAAAAGCGC"}, engine)
        both = panel.defect({"H": "GCGCAAAAGCGC"}, engine)
        assert both == pytest.approx(2 * single, rel=1e-9)

    def test_add_assay(self):
        panel = AssayPanel()
        panel.add_assay(Assay("a1"))
        panel.add_assay(Assay("a2"))
        assert [a.name for a in panel.assays] == ["a1", "a2"]


class TestAssayIntegration:
    def test_objective_drives_designer(self, engine):
        """Wire an Assay into SequenceDesigner and verify the best sequence
        scores at least as well as a random one."""
        assay = Assay(
            name="design_me",
            on_targets=[Assembly(
                "H", ["H"], "(((((....)))))", concentration=1.0,
            )],
        )
        designer = SequenceDesigner(engine=engine, seed=0)
        result = designer.design(
            domains={"H": DomainSpec(length=14)},
            objective=assay.to_objective(engine),
            n_trials=2,
            max_iterations=80,
        )
        designed_defect = assay.defect(result.sequences, engine)
        random_defect = assay.defect({"H": "AAAATTTTAAAATT"}, engine)
        assert designed_defect <= random_defect
