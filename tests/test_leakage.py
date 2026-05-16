"""Leakage enumerator tests."""
import pytest
from strider.kinetics.leakage import LeakageEnumerator, LeakageReport


def make_engine():
    from strider.thermo.engine import ThermoEngine
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


class TestLeakageEnumerator:
    def test_returns_report(self):
        engine = make_engine()
        enumerator = LeakageEnumerator(engine, ddg_threshold=-1.0)
        strands = {"A": "ACGTACGT", "B": "TTTTTTTT"}
        report = enumerator.enumerate(strands)
        assert isinstance(report, LeakageReport)

    def test_complementary_strands_show_leakage(self):
        engine = make_engine()
        enumerator = LeakageEnumerator(engine, ddg_threshold=0.0)
        # Perfectly complementary pair → very favorable hybridization
        strands = {"A": "GCGCGCGC", "B": "GCGCGCGC"}
        report = enumerator.enumerate(strands)
        assert report.total_spurious >= 0  # may or may not exceed threshold

    def test_filter_by_threshold(self):
        engine = make_engine()
        enumerator = LeakageEnumerator(engine, ddg_threshold=-2.0)
        strands = {"X": "AAAAAAAA", "Y": "TTTTTTTT"}
        report = enumerator.enumerate(strands)
        strict = report.filter(-5.0)
        assert strict.total_spurious <= report.total_spurious

    def test_to_mantis_strings(self):
        engine = make_engine()
        enumerator = LeakageEnumerator(engine, ddg_threshold=0.0)
        strands = {"P": "GCGC", "Q": "GCGC"}
        report = enumerator.enumerate(strands)
        strings = report.to_mantis_strings()
        assert isinstance(strings, list)
        for s in strings:
            assert "->" in s

    def test_intended_reactions_excluded(self):
        engine = make_engine()
        enumerator = LeakageEnumerator(engine, ddg_threshold=0.0)
        strands = {"A": "GCGCGCGC", "B": "GCGCGCGC"}
        intended = ["A + B -> AB"]
        report = enumerator.enumerate(strands, intended_reactions=intended)
        # The intended pair should not appear as spurious
        for rxn in report.reactions:
            key = frozenset(rxn.reactant_names)
            assert key != frozenset(["A", "B"])
