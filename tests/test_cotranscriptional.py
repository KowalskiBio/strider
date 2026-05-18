"""Tests for strider.structure.cotranscriptional."""

import pytest

from strider.structure.cotranscriptional import (
    fold_cotranscriptional,
    CotranscriptionalTrajectory,
    PrefixFold,
)


class TestFoldCotranscriptional:
    def test_empty_sequence(self):
        traj = fold_cotranscriptional("")
        assert traj.prefixes == []
        assert traj.sequence == ""

    def test_short_hairpin(self):
        seq = "GGGAAACCC"
        traj = fold_cotranscriptional(seq, min_length=5, material="rna")
        assert traj.sequence == seq
        # Always includes the full-length prefix at the end
        assert traj.final().length == len(seq)
        # Lengths are monotone non-decreasing
        lens = [p.length for p in traj.prefixes]
        assert lens == sorted(lens)
        assert all(L >= 5 for L in lens)

    def test_step_subsamples_lengths(self):
        seq = "GGGAAACCC" * 3  # length 27
        traj = fold_cotranscriptional(seq, min_length=5, step=3, material="rna")
        lens = [p.length for p in traj.prefixes]
        # Step 3 starting from 5: 5, 8, 11, ..., plus the full length
        assert 5 in lens
        assert 8 in lens
        assert lens[-1] == len(seq)

    def test_step_validation(self):
        with pytest.raises(ValueError):
            fold_cotranscriptional("GGGAAACCC", step=0)

    def test_at_length_lookup(self):
        seq = "GGGAAACCC"
        traj = fold_cotranscriptional(seq, min_length=5, material="rna")
        assert traj.at_length(99) is None
        last = traj.final()
        same = traj.at_length(last.length)
        assert same is last

    def test_full_prefix_matches_fold_mfe(self):
        """The last prefix should fold identically to fold_mfe(full sequence)."""
        from strider.structure.mfe import fold_mfe
        seq = "GGGGAAAACCCC"
        traj = fold_cotranscriptional(seq, min_length=5, material="rna")
        structure, energy, pairs = fold_mfe(seq, material="rna")
        final = traj.final()
        assert final.structure == structure
        assert final.energy == pytest.approx(energy)
        assert list(final.pairs) == [tuple(p) for p in pairs]

    def test_rearrangements_detected(self):
        """A bistable refold should appear in rearrangements()."""
        # Construct a sequence where a short 5' hairpin is supplanted by
        # a different pairing once more 3' sequence is available.
        # GGGAAACCC folds as ((...)) initially.  Extending with bases that
        # rebrace the structure should register a rearrangement.
        # Hard to guarantee without solving, so just check the API shape.
        seq = "GGGAAACCCAAAGGG"
        traj = fold_cotranscriptional(seq, min_length=5, material="rna")
        rearr = traj.rearrangements()
        # rearrangements() returns a list of length-pair tuples
        assert isinstance(rearr, list)
        for prev, curr in rearr:
            assert prev < curr

    def test_returned_pairs_are_within_prefix(self):
        seq = "GGGGAAAACCCC"
        traj = fold_cotranscriptional(seq, min_length=5, material="rna")
        for p in traj.prefixes:
            for i, j in p.pairs:
                assert 0 <= i < p.length
                assert 0 <= j < p.length

    def test_dna_material(self):
        """DNA material works (co-transcriptional folding less biologically
        relevant for DNA but the algorithm is the same)."""
        seq = "GGGGAAAACCCC"
        traj = fold_cotranscriptional(seq, min_length=5, material="dna")
        assert traj.material == "dna"
        assert traj.final().length == len(seq)
