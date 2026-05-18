"""Tests for the generic circuit-templates catalog and CircuitChecks framework."""
import pytest

mantis = pytest.importorskip("mantis", reason="mantis not installed")

from strider import (
    CHA, HCR, Translator, SeesawGate,
    CheckRegistry, CircuitReport,
    toehold_accessible, stability_in_range, reaction_driving_force,
    no_spurious_dimer,
)
from strider.thermo.engine import ThermoEngine


# ─── shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return ThermoEngine(material="dna", celsius=37.0,
                        sodium=0.137, magnesium=0.01)


# A modest set of DNA sequences for templates that don't depend on real biology.
_TEST_SEQS = {
    "H1": "GCGCATGCATGCATGCATGCATGCATGCGC",
    "H2": "GCGCATGCATGCATGCATGCATGCATGCGC",
    "I":  "GCATGCATGCATGCATGCAT",
}


# ─── CircuitChecks framework ─────────────────────────────────────────────────

class TestCheckRegistry:
    def test_empty_registry_passes(self, engine):
        report = CheckRegistry().run(engine, {})
        assert isinstance(report, CircuitReport)
        assert report.passed
        assert report.failed_checks == []

    def test_failing_check_reported(self, engine):
        reg = CheckRegistry().add(
            no_spurious_dimer("A", "B", min_ddg=-100.0, name="impossible")
        )
        report = reg.run(engine, {"A": "GCGCGC", "B": "GCGCGC"})
        # A and B can dimerise weakly, ΔΔG > -100 → "passes"
        # Use a more sensible threshold to actually fail
        reg2 = CheckRegistry().add(
            no_spurious_dimer("A", "B", min_ddg=10.0, name="strict")
        )
        r2 = reg2.run(engine, {"A": "GCGCGCGCGC", "B": "GCGCGCGCGC"})
        assert not r2.passed
        assert "strict" in r2.failed_checks

    def test_check_exception_caught(self, engine):
        def boom(ctx):
            raise RuntimeError("kaboom")
        boom.__check_name__ = "boom"
        reg = CheckRegistry().add(boom)
        report = reg.run(engine, {})
        assert not report.passed
        assert any("boom" in r.message or r.name == "boom" for r in report.results)

    def test_stability_check_uses_real_pfunc(self, engine):
        reg = CheckRegistry().add(
            stability_in_range("H", min_dg=-20, max_dg=0, name="hairpin")
        )
        report = reg.run(engine, {"H": "GCGCAAAAGCGC"})
        assert report.passed
        assert report.results[0].value < 0  # negative ΔG

    def test_reaction_driving_force_resolves_complexes(self, engine):
        reg = CheckRegistry().add(
            reaction_driving_force(["A", "B"], [["A", "B"]], max_ddg=0.0,
                                   name="bind")
        )
        report = reg.run(engine,
                         {"A": "GCGCGCAAAA", "B": "TTTTGCGCGC"})
        # Complementary strands → ΔΔG strongly negative → passes
        assert report.passed


# ─── HCR ──────────────────────────────────────────────────────────────────────

class TestHCR:
    def test_default_topology(self):
        hcr = HCR(sequences=_TEST_SEQS)
        assert len(hcr.reactions) == 4
        # All four canonical HCR reactions should be present
        joined = " | ".join(hcr.reactions)
        assert "I + H1" in joined
        assert "H1 + H2" in joined

    def test_toehold_map_populated(self):
        hcr = HCR(sequences=_TEST_SEQS, toehold_initiator=8, toehold_branch=5)
        assert hcr.toehold_map["I + H1 <-> I_H1"] == 8
        assert hcr.toehold_map["I_H1 + H2 <-> I_H1_H2"] == 5

    def test_to_bridge_returns_circuit_bridge(self):
        from strider import CircuitBridge
        hcr = HCR(sequences=_TEST_SEQS)
        assert isinstance(hcr.to_bridge(), CircuitBridge)

    def test_verify_returns_report(self):
        hcr = HCR(sequences=_TEST_SEQS)
        report = hcr.verify()
        assert isinstance(report, CircuitReport)
        # At minimum the H1/H2 stability checks should run
        names = [r.name for r in report.results]
        assert "H1_stability" in names
        assert "H2_stability" in names


# ─── Translator ──────────────────────────────────────────────────────────────

class TestTranslator:
    def test_default_topology(self):
        tr = Translator(sequences={
            "X": "GCATGCATGCAT", "Y": "AAAA",
            "Gate": "GCATGCATGCATAAAA",
        })
        assert len(tr.reactions) == 2
        assert "X + Gate" in tr.reactions[0]

    def test_verify_runs(self):
        tr = Translator(sequences={
            "X": "GCATGCATGCAT", "Y": "AAAA",
            "Gate": "GCATGCATGCATAAAA",
        })
        report = tr.verify()
        assert isinstance(report, CircuitReport)


# ─── Seesaw gate ─────────────────────────────────────────────────────────────

class TestSeesawGate:
    def test_yes_logic_one_input(self):
        sg = SeesawGate(logic="YES", sequences={
            "Input1": "AAAA", "Gate": "GCGC", "Threshold": "TTTT",
            "Fuel": "CCCC", "Output": "AGAG",
        })
        # 1 threshold + 1 signal release + 1 fuel recycle = 3 reactions
        assert len(sg.reactions) == 3

    def test_and_logic_two_inputs(self):
        sg = SeesawGate(logic="AND", sequences={
            "Input1": "AAAA", "Input2": "TTTT",
            "Gate": "GCGC", "Threshold_Input1": "AAAA",
            "Threshold_Input2": "TTTT", "Fuel": "CCCC", "Output": "AGAG",
        })
        # 2 thresholds + 2 releases + 2 recycles = 6 reactions
        assert len(sg.reactions) == 6

    def test_or_logic_two_inputs(self):
        sg = SeesawGate(logic="OR", sequences={
            "Input1": "AAAA", "Input2": "TTTT",
            "Gate": "GCGC", "Threshold": "AAAA",
            "Fuel": "CCCC", "Output": "AGAG",
        })
        # 2 thresholds (single Threshold strand) + 2 releases + 2 recycles
        assert len(sg.reactions) == 6

    def test_not_logic(self):
        sg = SeesawGate(logic="NOT", sequences={
            "Input1": "AAAA", "Gate": "GCGC", "Threshold": "TTTT",
            "Fuel": "CCCC", "Output": "AGAG",
        })
        # 1 threshold + Gate→Output baseline + Input1 inhibition + 1 recycle = 4
        assert len(sg.reactions) == 4

    def test_to_bridge_succeeds(self):
        sg = SeesawGate(logic="YES", sequences={
            "Input1": "GCATGCATGCAT", "Gate": "GCATATGCATGC",
            "Threshold": "AAAATTTT", "Fuel": "GCATGCAT",
            "Output": "AAAATTTT",
        })
        b = sg.to_bridge()
        assert b is not None


# ─── CHA (new framework) ─────────────────────────────────────────────────────

MIR21 = "TAGCTTATCAGACTGATGTTGA"
H1_S  = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2_S  = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
CP_S  = "AAAAA"


class TestCHA:
    def test_default_topology(self):
        cha = CHA(sequences={
            "mirna": MIR21, "H1": H1_S, "H2": H2_S, "CP": CP_S,
        })
        assert len(cha.reactions) == 4

    def test_to_bridge_and_verify(self):
        cha = CHA(sequences={
            "mirna": MIR21, "H1": H1_S, "H2": H2_S, "CP": CP_S,
        })
        bridge = cha.to_bridge()
        assert bridge is not None
        report = cha.verify()
        # Some checks pass and some fail on this real-world sequence set;
        # we just verify the framework wires up correctly.
        assert isinstance(report, CircuitReport)
        names = [r.name for r in report.results]
        for required in ("toehold_accessible", "H1_stability",
                         "R1_driving_force", "spontaneous_leakage"):
            assert required in names

    def test_simulate_via_template(self):
        cha = CHA(sequences={
            "mirna": MIR21, "H1": H1_S, "H2": H2_S, "CP": CP_S,
        })
        result = cha.simulate(
            {"mirna": 10e-9, "H1": 100e-9, "H2": 100e-9, "CP": 100e-9},
            (0.0, 60.0),
        )
        assert result.success


# ─── Custom registry composition ─────────────────────────────────────────────

class TestRegistryComposition:
    def test_user_can_override_default_checks(self):
        cha = CHA(sequences={
            "mirna": MIR21, "H1": H1_S, "H2": H2_S, "CP": CP_S,
        })
        custom_reg = (CheckRegistry()
            .add(stability_in_range("H1", -100, 100, name="trivial"))
        )
        report = cha.verify(registry=custom_reg)
        assert report.passed
        assert len(report.results) == 1
