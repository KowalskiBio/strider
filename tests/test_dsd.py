"""DSDCompiler tests."""
import pytest
from strider import DSDCompiler
from strider.thermo.nn_dna import reverse_complement


class TestDomainResolution:
    def test_base_domain_lookup(self):
        d = DSDCompiler(domains={"a": "GCATGC"})
        assert d.domain_sequence("a") == "GCATGC"

    def test_starred_complement_auto(self):
        d = DSDCompiler(domains={"a": "GCATGC"})
        assert d.domain_sequence("a*") == reverse_complement("GCATGC")

    def test_unknown_domain_raises(self):
        d = DSDCompiler()
        with pytest.raises(KeyError):
            d.domain_sequence("xyz")

    def test_starred_unknown_raises(self):
        d = DSDCompiler()
        with pytest.raises(KeyError):
            d.domain_sequence("xyz*")

    def test_add_starred_domain_rejects(self):
        d = DSDCompiler()
        with pytest.raises(ValueError):
            d.add_domain("a*", "GCAT")

    def test_lowercase_normalised(self):
        d = DSDCompiler(domains={"a": "gcat"})
        assert d.domain_sequence("a") == "GCAT"

    def test_u_converted_to_t(self):
        d = DSDCompiler()
        d.add_domain("r", "GCAU")
        assert d.domain_sequence("r") == "GCAT"


class TestStrandAssembly:
    def test_strand_concatenation(self):
        d = DSDCompiler(domains={"a": "GC", "b": "ATAT"})
        d.add_strand("S", ["a", "b"])
        assert d.strand_sequence("S") == "GCATAT"

    def test_strand_with_complement(self):
        d = DSDCompiler(domains={"a": "GC", "b": "ATAT"})
        d.add_strand("S", ["a", "b*"])
        assert d.strand_sequence("S") == "GC" + reverse_complement("ATAT")

    def test_unknown_strand_raises(self):
        d = DSDCompiler()
        with pytest.raises(KeyError):
            d.strand_sequence("unknown")

    def test_strand_referencing_missing_domain_raises(self):
        d = DSDCompiler(domains={"a": "GC"})
        with pytest.raises(ValueError):
            d.add_strand("S", ["a", "missing"])

    def test_sequences_dict(self):
        d = DSDCompiler(domains={"a": "GC", "b": "ATAT"})
        d.add_strand("S1", ["a", "b"])
        d.add_strand("S2", ["b*", "a*"])
        seqs = d.sequences()
        assert set(seqs.keys()) == {"S1", "S2"}
        assert seqs["S2"] == reverse_complement("ATAT") + reverse_complement("GC")


class TestBridgeIntegration:
    def test_to_bridge_returns_circuit_bridge(self):
        from strider import CircuitBridge
        d = DSDCompiler(domains={
            "t": "GCATGC",          # toehold
            "a": "ATGCATATGC",       # branch migration
        })
        d.add_strand("S1", ["t", "a"])
        d.add_strand("S2", ["a*", "t*"])
        d.add_reaction("S1 + S2 <-> S1_S2", toehold="t")
        bridge = d.to_bridge()
        assert isinstance(bridge, CircuitBridge)

    def test_toehold_length_propagates(self):
        d = DSDCompiler(domains={
            "t": "GCATGC",
            "a": "ATGCATATGC",
        })
        d.add_strand("S1", ["t", "a"])
        d.add_strand("S2", ["a*", "t*"])
        d.add_reaction("S1 + S2 <-> S1_S2", toehold="t")
        bridge = d.to_bridge()
        assert bridge.toehold_map["S1 + S2 <-> S1_S2"] == 6

    def test_bridge_rates_match_strider_kinetics(self):
        from strider.kinetics.tmsd import toehold_kf
        d = DSDCompiler(domains={
            "t": "GCATGC",
            "a": "ATGCATATGC",
        })
        d.add_strand("S1", ["t", "a"])
        d.add_strand("S2", ["a*", "t*"])
        d.add_reaction("S1 + S2 <-> S1_S2", toehold="t")
        bridge = d.to_bridge()
        expected_kf = toehold_kf(6, "dna", celsius=37.0)
        assert abs(bridge.rates["S1 + S2 -> S1_S2"] - expected_kf) / expected_kf < 1e-6
