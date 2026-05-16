"""MFE structure prediction tests."""
import pytest
from strider.structure.mfe import fold_mfe
from strider.structure.dot_bracket import parse_pairs, validate, stem_regions


class TestFoldMFE:
    def test_returns_dot_bracket(self):
        structure, energy, pairs = fold_mfe("AAAAAAAAAA")
        assert all(c in ".()" for c in structure)
        assert len(structure) == 10

    def test_completely_unpaired_all_a(self):
        structure, energy, pairs = fold_mfe("AAAAAAAAAA")
        assert structure == "." * 10

    def test_hairpin_forms_stem(self):
        # GCGCAAAAACGCG should form a stem-loop
        seq = "GCGCAAAAACGCG"
        structure, energy, pairs = fold_mfe(seq)
        assert energy < 0.0, "Hairpin should have negative energy"
        assert len(pairs) > 0, "Should form at least one base pair"

    def test_complementary_regions_pair(self):
        # GCGC-TTTT-GCGC: RC("GCGC")="GCGC" (palindrome), pairs at ends
        seq = "GCGCTTTTGCGC"
        structure, energy, pairs = fold_mfe(seq)
        assert len(pairs) >= 2, "Should form stem pairs"

    def test_rna_folding(self):
        seq = "GCGCUUUUCGCG"
        structure, energy, pairs = fold_mfe(seq, material="rna")
        assert len(structure) == len(seq)

    def test_energy_negative_for_stable_hairpin(self):
        seq = "GCGCGCTTTTCGCGCG"
        _, energy, _ = fold_mfe(seq)
        assert energy < 0.0

    def test_valid_dot_bracket(self):
        structure, _, _ = fold_mfe("GCGCAAAAACGCG")
        assert validate(structure)


class TestDotBracket:
    def test_parse_simple(self):
        pairs = parse_pairs("((...))")
        assert (0, 6) in pairs
        assert (1, 5) in pairs

    def test_empty(self):
        pairs = parse_pairs("......")
        assert pairs == []

    def test_nested(self):
        pairs = parse_pairs("(((())))")
        assert len(pairs) == 4

    def test_validate_balanced(self):
        assert validate("((...))")
        assert validate("......")
        assert not validate("((...))(")  # unbalanced

    def test_stem_regions(self):
        stems = stem_regions("(((...)))")
        assert len(stems) >= 1
        _, _, length = stems[0]
        assert length >= 2
