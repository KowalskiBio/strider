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


class TestFullZukerEnergetics:
    """Spot-check the new Zuker MFE engine against canonical small-hairpin physics."""

    def test_tetraloop_hairpin_negative_dg(self):
        # GC-clamped tetraloop hairpin is a textbook stable structure.
        _, energy, pairs = fold_mfe("GCGCAAAAGCGC", material="rna")
        assert energy < -3.0, f"expected stable hairpin (ΔG < -3), got {energy:.2f}"
        assert len(pairs) >= 3

    def test_stack_consistency_with_pfunc(self):
        # MFE energy must be ≤ ensemble ΔG (ensemble averages over all states).
        from strider.thermo.engine import ThermoEngine
        seq = "GCGCAAAAGCGC"
        _, mfe_e, _ = fold_mfe(seq, material="rna")
        e = ThermoEngine(material="rna", celsius=37.0).pfunc(seq).free_energy
        assert mfe_e <= e + 1e-6, f"MFE {mfe_e:.3f} should not exceed ensemble ΔG {e:.3f}"

    def test_internal_loop_handled(self):
        # Sequence with a forced 1×2 internal loop in the optimal fold:
        # outer stem GC|GC, internal loop A vs AA, hairpin loop AAAA.
        seq = "GCAGCAAAAGCAAGC"
        _, energy, pairs = fold_mfe(seq, material="rna")
        # The internal-loop branch should produce more pairs than just a single stack would.
        assert len(pairs) >= 4

    def test_multiloop_branch_explored(self):
        # Two stems separated by a small linker: should find a multiloop OR two
        # external stems.  Either way, total pairs ≥ 6.
        seq = "GCGCGGAAAACCGCGCAACGCGCAAAAGCGCG"
        _, _, pairs = fold_mfe(seq, material="rna")
        assert len(pairs) >= 6


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
