"""
G-quadruplex folding, two-state thermodynamics, and the duplex-vs-G4 partition.

The empirical ΔH/ΔS model (``structure/quadruplex.py``) cannot be validated
against NUPACK — NUPACK structurally cannot represent a G4 (pseudoknots off,
WC/wobble only).  Validation is therefore (a) against canonical experimental
*trends* and rough Tm/ΔG anchors, and (b) internal consistency of the partition
competition.  The model is built so the trends are correct even where absolute
numbers carry the field's real construct-to-construct spread.
"""
import math

import pytest

from strider.structure.quadruplex import (
    find_g4_motifs,
    fold_quadruplex,
    g4_thermodynamics,
    quadruplex_ensemble,
    G4Motif,
)

# canonical putative-quadruplex sequences
C_MYC = "TGAGGGTGGGTAGGGTGGGTAA"      # 3 tetrads, short loops — very stable
TELOMERE = "AGGGTTAGGGTTAGGGTTAGGG"   # 3 tetrads, 3-nt TTA loops
TBA = "GGTTGGTGTGGTTGG"              # 2 tetrads, thrombin-binding aptamer
NON_G4 = "ATCGATCGATCGATCGATCGATCG"  # no G-tracts


# ── motif recognition ──────────────────────────────────────────────────────────

class TestMotifFinding:
    def test_canonical_have_motifs(self):
        assert find_g4_motifs(C_MYC)
        assert find_g4_motifs(TELOMERE)
        assert find_g4_motifs(TBA, min_tetrads=2)

    def test_non_g4_has_none(self):
        assert find_g4_motifs(NON_G4) == []

    def test_tetrad_counts(self):
        assert fold_quadruplex(C_MYC).motif.n_tetrads == 3
        assert fold_quadruplex(TELOMERE).motif.n_tetrads == 3
        assert fold_quadruplex(TBA, min_tetrads=2).motif.n_tetrads == 2

    def test_engaged_positions_in_range_and_count(self):
        f = fold_quadruplex(TELOMERE)
        eng = f.motif.engaged_positions()
        assert len(eng) == 4 * f.motif.n_tetrads
        assert all(0 <= p < len(TELOMERE) for p in eng)

    def test_rna_guanine_recognized(self):
        rna = "GGGUUAGGGUUAGGGUUAGGG"
        assert find_g4_motifs(rna)


# ── stability trends ─────────────────────────────────────────────────────────

class TestStabilityTrends:
    def test_folding_is_favorable_for_canonical(self):
        for s in (C_MYC, TELOMERE, TBA):
            assert fold_quadruplex(s, min_tetrads=2).dG < 0

    def test_short_loops_more_stable(self):
        # c-myc (short loops) more stable than telomere (3-nt loops), same tetrads
        assert fold_quadruplex(C_MYC).dG < fold_quadruplex(TELOMERE).dG

    def test_more_tetrads_more_stable(self):
        # identical loops, 2 vs 3 vs 4 G's per tract
        two = "GGTGGTGGTGG"
        three = "GGGTGGGTGGGTGGG"
        four = "GGGGTGGGGTGGGGTGGGG"
        dg2 = fold_quadruplex(two, min_tetrads=2).dG
        dg3 = fold_quadruplex(three, min_tetrads=2).dG
        dg4 = fold_quadruplex(four, min_tetrads=2).dG
        assert dg4 < dg3 < dg2

    def test_loop_length_monotonic(self):
        # same 3 tetrads, growing loop length ⇒ less stable (higher dG)
        prev = -math.inf
        for L in (1, 3, 5, 7):
            loop = "T" * L
            seq = f"GGG{loop}GGG{loop}GGG{loop}GGG"
            dg = fold_quadruplex(seq, min_tetrads=3, max_loop=8).dG
            assert dg > prev
            prev = dg

    def test_tm_ordering(self):
        assert fold_quadruplex(C_MYC).tm_celsius > fold_quadruplex(TELOMERE).tm_celsius


# ── cation dependence (the aptamer-biosensor lever) ───────────────────────────

class TestCationDependence:
    def test_potassium_more_stabilizing_than_sodium(self):
        k = fold_quadruplex(TELOMERE, potassium=0.1, sodium=0.0)
        na = fold_quadruplex(TELOMERE, potassium=0.0, sodium=0.1)
        assert k.dG < na.dG
        assert k.tm_celsius > na.tm_celsius

    def test_higher_potassium_more_stable(self):
        prev = math.inf
        for conc in (0.001, 0.01, 0.1, 1.0):
            dg = fold_quadruplex(TELOMERE, potassium=conc, sodium=0.0).dG
            assert dg < prev
            prev = dg

    def test_reference_condition_matches_fit(self):
        # at 100 mM K+ the c-myc anchor should reproduce the fit target (~ -6 / 88 C)
        f = fold_quadruplex(C_MYC, celsius=37, potassium=0.1, sodium=0.0)
        assert f.dG == pytest.approx(-6.0, abs=0.6)
        assert f.tm_celsius == pytest.approx(88.0, abs=4.0)


# ── folded fraction ────────────────────────────────────────────────────────────

class TestFoldedFraction:
    def test_fraction_bounds(self):
        for s in (C_MYC, TELOMERE, TBA, NON_G4):
            frac = fold_quadruplex(s, min_tetrads=2).folded_fraction
            assert 0.0 <= frac <= 1.0

    def test_non_g4_unfolded(self):
        assert fold_quadruplex(NON_G4).folded_fraction == 0.0

    def test_melts_with_temperature(self):
        cold = fold_quadruplex(TELOMERE, celsius=20).folded_fraction
        hot = fold_quadruplex(TELOMERE, celsius=90).folded_fraction
        assert cold > hot
        assert hot < 0.5  # above its ~57 C Tm

    def test_half_folded_at_tm(self):
        f = fold_quadruplex(C_MYC)
        at_tm = fold_quadruplex(C_MYC, celsius=f.tm_celsius).folded_fraction
        assert at_tm == pytest.approx(0.5, abs=0.02)


# ── partition competition (duplex/hairpin vs G4) ──────────────────────────────

class TestPartitionCompetition:
    def test_probabilities_normalized(self):
        e = quadruplex_ensemble(C_MYC, potassium=0.1)
        assert e.p_g4 + e.p_secondary == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= e.p_g4 <= 1.0

    def test_stable_g4_dominates(self):
        e = quadruplex_ensemble(C_MYC, potassium=0.1)
        assert e.p_g4 > 0.99

    def test_non_g4_has_no_g4_population(self):
        e = quadruplex_ensemble(NON_G4, potassium=0.1)
        assert e.p_g4 == 0.0
        assert e.z_total == pytest.approx(e.z_secondary)

    def test_potassium_shifts_population_to_g4(self):
        lo = quadruplex_ensemble(TELOMERE, potassium=0.0005).p_g4
        hi = quadruplex_ensemble(TELOMERE, potassium=0.5).p_g4
        assert hi > lo

    def test_per_motif_occupancies_sum_to_total(self):
        e = quadruplex_ensemble(C_MYC, potassium=0.1)
        assert sum(p for _, p in e.p_g4_by_motif) == pytest.approx(e.p_g4, abs=1e-9)
