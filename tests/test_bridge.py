"""
Mantis bridge tests.

Skipped when mantis is not installed.
"""
import pytest

mantis = pytest.importorskip("mantis", reason="mantis not installed")

MIR21_SEQ = "TAGCTTATCAGACTGATGTTGA"
H1_SEQ = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2_SEQ = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
CP_SEQ = "AAAAA"


@pytest.fixture
def bridge():
    from strider.bridge.mantis_bridge import CHABridge
    return CHABridge(
        sequences={"mirna": MIR21_SEQ, "H1": H1_SEQ, "H2": H2_SEQ, "CP": CP_SEQ},
    )


class TestCHABridge:
    def test_ddg_pathway_keys(self, bridge):
        ddg = bridge.ddg_pathway
        for key in ("R1", "R2", "R3", "leakage", "cp_leakage", "g_H1", "g_H2"):
            assert key in ddg, f"Missing key: {key}"

    def test_r1_negative(self, bridge):
        ddg = bridge.ddg_pathway
        assert ddg["R1"] < 0, "R1 (initiation) should be thermodynamically favorable"

    def test_r3_favorable(self, bridge):
        ddg = bridge.ddg_pathway
        assert ddg["R3"] < 0, "R3 (detection) should be favorable"

    def test_rates_dict_has_8_entries(self, bridge):
        rates = bridge.rates
        assert len(rates) == 8, f"CHA has 4 reversible reactions = 8 rate constants, got {len(rates)}"

    def test_rates_all_positive(self, bridge):
        for key, val in bridge.rates.items():
            assert val > 0, f"Rate {key} must be positive, got {val}"

    def test_to_crnetwork(self, bridge):
        rn = bridge.to_crnetwork()
        assert rn is not None
        assert rn.n_species >= 5   # at minimum: H1, H2, CP, mirna, at least one complex
        assert rn.n_reactions == 8

    def test_verify_returns_report(self, bridge):
        from strider.bridge.mantis_bridge import CHAVerificationReport
        report = bridge.verify()
        assert isinstance(report, CHAVerificationReport)
        assert hasattr(report, "all_passed")
        assert hasattr(report, "failed_checks")

    def test_crnetwork_simulate(self, bridge):
        rn = bridge.to_crnetwork()
        ic = bridge._default_ic()
        result = rn.simulate(ic, (0, 7200))
        # Simulation ran and returned species concentrations
        assert len(result.concentrations) == 7
        final = result.final()
        h1h2_cp_key = [k for k in final if "CP" in k and "_" in k]
        assert len(h1h2_cp_key) > 0


class TestRatesToCRNetwork:
    def test_without_leakage(self):
        from strider.bridge.mantis_bridge import rates_to_crnetwork
        seqs = {"H1": H1_SEQ, "H2": H2_SEQ}
        reactions = ["H1 + H2 <-> H1H2"]
        rn = rates_to_crnetwork(reactions, seqs, include_leakage=False)
        assert rn is not None
        assert "H1" in rn.species
        assert "H2" in rn.species
