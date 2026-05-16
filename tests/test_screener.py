"""
Tests for strider.screen.offtarget — OffTargetScreener.

Three bugs were present before this test file was written:

  Bug 1 — screen() called engine.duplex_dg(probe, ref) which ignores the
           second strand: every hit returned the same ΔΔG value regardless
           of the reference sequence.
           Caught by: TestScreen.test_energies_are_not_all_identical

  Bug 2 — _build_kmer_index only indexed forward-strand k-mers.  Complementary
           sequences (the typical binding partners of a probe) share no forward
           k-mers with the probe, so they were never found as candidates.
           Caught by: TestKmerIndex.test_rc_kmer_indexed,
                      TestKmerIndex.test_complement_found_as_candidate

  Bug 3 — specificity_vs declared family_members: list[str] but was called
           with a dict {name: seq}.  Iterating over a dict yields keys, so
           engine.duplex_dg was called with name-strings instead of DNA
           sequences, returning garbage (0.0 or NaN for all members).
           Caught by: TestSpecificityVs.test_dict_values_are_finite_floats,
                      TestSpecificityVs.test_dict_keys_are_names
"""
from __future__ import annotations

import math

import pytest

from strider.screen.offtarget import OffTargetScreener, _kmers, _rc
from strider.thermo.engine import ThermoEngine

# ── Non-repetitive probe/target sequences ────────────────────────────────────
# These are intentionally diverse (not CHA-specific) so the tests generalise.

PROBE      = "CGCAGTCGATCAGTACGCTG"   # 20 nt, 50% GC, non-repetitive
RC_PROBE   = "CAGCGTACTGATCGACTGCG"   # exact reverse complement of PROBE
MIS_1      = "CAGCGTTCTGATCGACTGCG"   # 1 substitution vs RC_PROBE (A→T pos 7)
MIS_3      = "CAGCGTATCGATCGACTACG"   # 3 substitutions vs RC_PROBE
UNRELATED  = "TATATATATATATATATATAT"  # TA-repeat, no complementarity to PROBE
UNRELATED2 = "GCGCGCGCGCGCGCGCGCGC"  # GC-repeat, no complementarity to PROBE


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eng():
    return ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)


@pytest.fixture(scope="module")
def screener(eng):
    """Screener pre-loaded with exact complement, two mismatch variants, two unrelated."""
    s = OffTargetScreener(eng, kmer_k=6)
    s.add_sequences({
        "exact":      RC_PROBE,
        "mismatch_1": MIS_1,
        "mismatch_3": MIS_3,
        "unrelated":  UNRELATED,
        "unrelated2": UNRELATED2,
    })
    return s


# ── 1. Helper functions ───────────────────────────────────────────────────────

class TestHelpers:
    def test_rc_known_values(self):
        assert _rc("AAAA") == "TTTT"
        assert _rc("ACGT") == "ACGT"          # palindrome
        assert _rc("GCATGC") == "GCATGC"      # palindrome
        assert _rc("AACG") == "CGTT"

    def test_rc_involution(self):
        """RC(RC(s)) == s for any sequence."""
        for seq in [PROBE, RC_PROBE, MIS_1, "AAAAGGGGCCCCTTTT"]:
            assert _rc(_rc(seq)) == seq

    def test_rc_probe_is_correct(self):
        """Verify our test constants are consistent."""
        assert _rc(PROBE) == RC_PROBE

    def test_kmers_length(self):
        seq = "ACGTACGT"
        assert len(_kmers(seq, 4)) == len(seq) - 4 + 1
        assert len(_kmers(seq, 1)) == len(seq)
        assert len(_kmers(seq, len(seq))) == 1

    def test_kmers_content(self):
        mers = _kmers("ABCDE", 3)
        assert mers == ["ABC", "BCD", "CDE"]

    def test_kmers_first_and_last(self):
        mers = _kmers(PROBE, 6)
        assert mers[0] == PROBE[:6]
        assert mers[-1] == PROBE[-6:]


# ── 2. K-mer index construction ───────────────────────────────────────────────

class TestKmerIndex:

    def test_forward_kmers_are_indexed(self, eng):
        """Forward k-mers of a database sequence must appear in the index."""
        s = OffTargetScreener(eng, kmer_k=6)
        s.add_sequences({"seq": PROBE})
        for kmer in _kmers(PROBE, 6):
            assert kmer in s._kmer_index, f"Forward k-mer {kmer!r} missing from index"

    def test_rc_kmers_are_indexed(self, eng):
        """RC k-mers of each database sequence must also be indexed.

        Regression test for Bug 2: without RC indexing, a complementary probe
        (which shares no forward k-mers with the query) will never be found as
        a candidate and its binding energy will never be evaluated.
        """
        s = OffTargetScreener(eng, kmer_k=6)
        s.add_sequences({"seq": RC_PROBE})
        rc_kmers = set(_kmers(PROBE, 6))          # k-mers of PROBE = RC of RC_PROBE
        indexed   = set(s._kmer_index.keys())
        overlap   = rc_kmers & indexed
        assert overlap, (
            "None of the RC k-mers are in the index. "
            "Complementary sequences will never be found as candidates."
        )

    def test_complement_found_as_candidate(self, screener):
        """The exact complement (RC of PROBE) must appear in k-mer candidates.

        Regression test for Bug 2: this assertion fails if only forward k-mers
        are indexed, because PROBE and RC_PROBE share no forward k-mers.
        """
        candidates = screener._kmer_candidates(PROBE)
        assert "exact" in candidates, (
            "exact (RC of PROBE) not a k-mer candidate — RC k-mers are not indexed"
        )

    def test_mismatch_found_as_candidate(self, screener):
        """Near-complement with one mismatch must also be discovered."""
        candidates = screener._kmer_candidates(PROBE)
        assert "mismatch_1" in candidates

    def test_exact_ranks_above_3mismatch(self, screener):
        """Exact complement must rank higher (more shared k-mers) than 3-mismatch."""
        candidates = screener._kmer_candidates(PROBE)
        if "mismatch_3" not in candidates:
            pytest.skip("mismatch_3 not a candidate — k-mer count too low")
        assert candidates.index("exact") < candidates.index("mismatch_3")

    def test_add_sequences_is_additive(self, eng):
        """Calling add_sequences twice must extend (not replace) the database."""
        s = OffTargetScreener(eng, kmer_k=6)
        s.add_sequences({"A": "GCATGCAT"})
        s.add_sequences({"B": "AAAAGGGG"})
        assert "A" in s._db
        assert "B" in s._db


# ── 3. screen() ───────────────────────────────────────────────────────────────

class TestScreen:

    def test_returns_screening_report(self, screener):
        from strider.screen.offtarget import ScreeningReport
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        assert isinstance(report, ScreeningReport)

    def test_exact_complement_is_top_hit(self, screener):
        """The perfect complement must produce the most negative ΔΔG."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        assert report.hits, "No hits returned — exact complement not found"
        assert report.hits[0].name == "exact", (
            f"Top hit is {report.hits[0].name!r}, expected 'exact'"
        )

    def test_energies_are_not_all_identical(self, screener):
        """ΔΔG values across hits must differ.

        Regression test for Bug 1: before the fix, screen() called
        engine.duplex_dg(probe, ref) which ignored 'ref' and always returned
        the self-energy of 'probe'.  Every hit had the same ΔΔG.
        """
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        ddgs = [h.ddg for h in report.hits]
        assert len(set(ddgs)) > 1, (
            "All binding energies are identical — screen() is ignoring the reference sequence "
            "(duplex_dg must use both strands)"
        )

    def test_energy_gradient_exact_gt_mismatch1_gt_mismatch3(self, screener):
        """Binding strength must decrease with mismatch count:
        exact_complement < 1-mismatch < 3-mismatch (more negative = stronger)."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        hit_map = {h.name: h.ddg for h in report.hits}
        assert "exact" in hit_map,      "exact complement not found"
        assert "mismatch_1" in hit_map, "1-mismatch not found"
        assert "mismatch_3" in hit_map, "3-mismatch not found"
        assert hit_map["exact"] < hit_map["mismatch_1"] - 1.0, (
            "1-mismatch should bind at least 1 kcal/mol weaker than exact complement"
        )
        assert hit_map["mismatch_1"] < hit_map["mismatch_3"] - 1.0, (
            "3-mismatch should bind at least 1 kcal/mol weaker than 1-mismatch"
        )

    def test_exact_complement_much_stronger_than_unrelated(self, screener):
        """Complement must bind at least 10 kcal/mol stronger than unrelated sequence."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        hit_map = {h.name: h.ddg for h in report.hits}
        if "unrelated" not in hit_map:
            pytest.skip("unrelated not in hits (filtered by k-mer prescreen — acceptable)")
        assert hit_map["exact"] < hit_map["unrelated"] - 10.0

    def test_threshold_filters_weak_binders(self, screener):
        """All returned hits must have ΔΔG below the threshold."""
        threshold = -10.0
        report = screener.screen(PROBE, n_top=10, ddg_threshold=threshold)
        for hit in report.hits:
            assert hit.ddg < threshold, (
                f"Hit {hit.name!r} with ddg={hit.ddg:.2f} is above threshold {threshold}"
            )

    def test_hits_sorted_ascending_by_ddg(self, screener):
        """Hits must be sorted from most to least negative (ascending ΔΔG)."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        ddgs = [h.ddg for h in report.hits]
        assert ddgs == sorted(ddgs), "Hits are not sorted by ΔΔG"

    def test_n_top_caps_results(self, screener):
        """n_top must limit the number of returned hits."""
        report = screener.screen(PROBE, n_top=2, ddg_threshold=0.0)
        assert len(report.hits) <= 2

    def test_report_top_ddg_matches_first_hit(self, screener):
        """ScreeningReport.top_ddg must equal the ΔΔG of the first hit."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        if report.hits:
            assert report.top_ddg == report.hits[0].ddg

    def test_is_specific_when_no_hits(self, eng):
        """is_specific must be True when no hits pass the threshold."""
        s = OffTargetScreener(eng, kmer_k=6)
        s.add_sequences({"unrelated": UNRELATED})
        # Use a very tight threshold — nothing should pass
        report = s.screen(PROBE, n_top=10, ddg_threshold=-100.0)
        assert report.is_specific

    def test_empty_database_returns_no_hits(self, eng):
        """Screening against an empty database must return an empty report."""
        s = OffTargetScreener(eng, kmer_k=6)
        report = s.screen(PROBE, n_top=10, ddg_threshold=0.0)
        assert report.hits == []
        assert report.is_specific

    def test_hit_fields_are_populated(self, screener):
        """Each OffTargetHit must have all expected fields with correct types."""
        report = screener.screen(PROBE, n_top=10, ddg_threshold=0.0)
        assert report.hits, "No hits — cannot check fields"
        hit = report.hits[0]
        assert isinstance(hit.name, str)
        assert isinstance(hit.sequence, str)
        assert isinstance(hit.ddg, float)
        assert isinstance(hit.k_score, int)
        assert hit.k_score >= 0

    @pytest.mark.parametrize("kmer_k", [5, 6, 7])
    def test_exact_complement_found_for_various_k(self, eng, kmer_k):
        """RC k-mer indexing must work regardless of k value."""
        s = OffTargetScreener(eng, kmer_k=kmer_k)
        s.add_sequences({"exact": RC_PROBE, "unrelated": UNRELATED})
        report = s.screen(PROBE, n_top=5, ddg_threshold=0.0)
        assert report.hits, f"No hits for kmer_k={kmer_k}"
        assert report.hits[0].name == "exact", (
            f"Exact complement not top hit for kmer_k={kmer_k}"
        )


# ── 4. specificity_vs() ───────────────────────────────────────────────────────

class TestSpecificityVs:

    def test_dict_keys_are_names(self, screener):
        """When family_members is a dict, result keys must be the dict's keys (names),
        not sequences.

        Regression test for Bug 3: before the fix, iterating over a dict yielded
        key strings (e.g. 'mismatch_1') which were then passed as DNA sequences
        to engine.duplex_dg, producing garbage energies.
        """
        family = {"mismatch_1": MIS_1, "unrelated": UNRELATED}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        assert "mismatch_1" in result, "Name key 'mismatch_1' missing from result"
        assert "unrelated"  in result, "Name key 'unrelated' missing from result"

    def test_dict_values_are_finite_floats(self, screener):
        """Selectivity scores must be finite floats — not NaN or strings.

        Regression test for Bug 3: the broken implementation returned NaN (when
        duplex_dg raised an exception on a non-DNA string) or 0.0 (when the
        engine silently accepted invalid input), not a real energy difference.
        """
        family = {"mismatch_1": MIS_1, "mismatch_3": MIS_3, "unrelated": UNRELATED}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        for name, score in result.items():
            assert isinstance(score, float), f"Score for {name!r} is {type(score)}, not float"
            assert not math.isnan(score),    f"Score for {name!r} is NaN"
            assert not math.isinf(score),    f"Score for {name!r} is Inf"

    def test_list_input_also_works(self, screener):
        """list[str] (original type annotation) must still be accepted."""
        result = screener.specificity_vs(
            PROBE,
            family_members=[MIS_1, UNRELATED],
            target=RC_PROBE,
        )
        assert len(result) == 2
        for score in result.values():
            assert isinstance(score, float)
            assert not math.isnan(score)

    def test_selectivity_ordering(self, screener):
        """Unrelated sequence must have higher selectivity (weaker binding) than mismatches.
        Higher selectivity score = more different ΔΔG from target = weaker off-target binding.
        """
        family = {"mismatch_1": MIS_1, "mismatch_3": MIS_3, "unrelated": UNRELATED}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        assert result["unrelated"]  > result["mismatch_1"], (
            "Unrelated sequence should have higher selectivity than 1-mismatch"
        )
        assert result["mismatch_3"] > result["mismatch_1"], (
            "3-mismatch should have higher selectivity than 1-mismatch"
        )

    def test_off_target_selectivity_is_positive(self, screener):
        """Unrelated sequences should bind weaker than the target → positive selectivity."""
        family = {"unrelated": UNRELATED, "unrelated2": UNRELATED2}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        for name, score in result.items():
            assert score > 0.0, (
                f"{name!r} has non-positive selectivity ({score:.2f}) — "
                "it binds as strongly as the specific target"
            )

    def test_exact_target_gives_near_zero_selectivity(self, screener):
        """A family member identical to the target must give selectivity ≈ 0."""
        family = {"same_as_target": RC_PROBE}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        assert abs(result["same_as_target"]) < 1.0, (
            f"Same-as-target selectivity is {result['same_as_target']:.3f}, expected ≈ 0"
        )

    def test_mismatch_selectivity_increases_with_mismatch_count(self, screener):
        """More mismatches → weaker binding → higher selectivity score."""
        family = {"mis1": MIS_1, "mis3": MIS_3}
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        assert result["mis3"] > result["mis1"], (
            "3-mismatch should be more selective (weaker binder) than 1-mismatch"
        )

    def test_target_binding_is_most_negative_ddg(self, screener):
        """Target ΔΔG must be more negative than all family members (positive selectivity)."""
        family = {
            "mismatch_1": MIS_1,
            "mismatch_3": MIS_3,
            "unrelated":  UNRELATED,
        }
        result = screener.specificity_vs(PROBE, family_members=family, target=RC_PROBE)
        # All selectivities positive ↔ all members bind weaker than target
        for name, score in result.items():
            assert score > 0, (
                f"{name!r} has selectivity {score:.2f} ≤ 0 — "
                "it binds as strongly as the intended target"
            )
