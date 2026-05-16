"""Shared fixtures for cicada tests."""
import pytest
from strider.thermo.engine import ThermoEngine

MIR21_SEQ = "TAGCTTATCAGACTGATGTTGA"

# Best sequences from claude_codesign.py last run
H1_SEQ = "TCAACATCAGTCTGATACCTCCCTCCTTATCAGACTGA"
H2_SEQ = "TCAGTCTGATAAGGGTGGAGGTATCAGACTGATGTTGATTTTT"
CP_SEQ = "AAAAA"

# Short test sequences from SantaLucia 2004 Table 2
BENCHMARK_DUPLEXES = [
    # (seq, ΔG at 37°C, 1M NaCl, kcal/mol)   ± 0.2 kcal/mol tolerance
    ("GCATGC", -8.0),
    ("ATCGAT", -7.0),
    ("AATCGATT", -9.9),
    ("GGGCCC", -13.1),
]


@pytest.fixture
def engine():
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


@pytest.fixture
def engine_1m():
    """Standard 1M NaCl engine for benchmarking against published values."""
    return ThermoEngine(material="dna", celsius=37.0, sodium=1.0, magnesium=0.0)


@pytest.fixture
def mir21():
    return MIR21_SEQ


@pytest.fixture
def cha_seqs():
    return {"mirna": MIR21_SEQ, "H1": H1_SEQ, "H2": H2_SEQ, "CP": CP_SEQ}
