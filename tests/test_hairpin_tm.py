"""Two-state hairpin Tm: self-consistency and reference agreement."""
import pytest

from strider.thermo.hairpin import hairpin_thermo, hairpin_tm, fraction_folded

# A simple 4-bp / 3-nt-loop hairpin used to anchor against an independent engine.
HP = "CTTTCAACACTGTTGCAGTAA"


def test_matches_independent_reference_at_1M():
    # seqfold (same SantaLucia params) gives ~45.5 C at 1 M Na, 0 Mg.
    tm = hairpin_tm(HP, sodium_M=1.0, magnesium_M=0.0)
    assert tm == pytest.approx(45.5, abs=1.5)


def test_dG_and_Tm_are_self_consistent():
    # ΔG(Tm) must vanish for a two-state melt: fraction folded == 0.5 at Tm.
    th = hairpin_thermo(HP, sodium_M=0.05, magnesium_M=0.010)
    assert fraction_folded(HP, th.tm_celsius, 0.05, 0.010) == pytest.approx(0.5, abs=0.02)


def test_magnesium_raises_tm_monotonically():
    base = hairpin_tm(HP, 0.05, 0.0)
    mid = hairpin_tm(HP, 0.05, 0.003)
    hi = hairpin_tm(HP, 0.05, 0.010)
    assert base < mid < hi


def test_lower_sodium_lowers_tm():
    assert hairpin_tm(HP, 1.0, 0.0) > hairpin_tm(HP, 0.05, 0.0)


def test_stronger_stem_has_higher_tm():
    weak = hairpin_tm(HP, 0.05, 0.010)
    strong = hairpin_tm("CGCGAAAAAGCGCG", 0.05, 0.010)
    assert strong > weak


def test_rejects_non_hairpin():
    with pytest.raises(ValueError):
        hairpin_tm("AAAAAAAAAAAA")  # no pairs
