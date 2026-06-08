"""
Correctness of the native McCaskill pair-probability recurrence.

The decisive test is the **unpaired-marginal identity**: for the exact base-pair
marginals of a partition-function ensemble,

    Σ_j P(i,j) + P_unpaired(i) = 1   for every position i,

where P_unpaired(i) = Z(i forbidden to pair) / Z is computed by re-running the
*same* DP with position i blocked.  The identity holds iff the outside recurrence
is the exact adjoint of the inside recurrence — i.e. iff the pair probabilities
(including the previously-missing multiloop contributions) are correct.  It is
energy-model-agnostic, so it validates the whole stack at once.
"""
import numpy as np
import pytest

from strider.thermo.ensemble import ensemble_dg, multistrand_pairs, dangle_free_partition


def _marginal_max_error(seq, material, celsius=37.0, sodium=1.0, magnesium=0.0):
    _, P = ensemble_dg(seq, celsius, material, sodium, magnesium)
    Znd = dangle_free_partition(seq, celsius, material, sodium, magnesium)
    n = len(seq)
    worst = 0.0
    for i in range(n):
        Zb = dangle_free_partition(seq, celsius, material, sodium, magnesium,
                                   blocked={i})
        p_unp = Zb / Znd
        s = sum(P[i][j] for j in range(n) if j != i)
        worst = max(worst, abs(s + p_unp - 1.0))
    return worst


# ─── the rigorous identity ──────────────────────────────────────────────────────

class TestUnpairedMarginalIdentity:
    @pytest.mark.parametrize("seq,material", [
        ("GGGAAACCC", "rna"),                                  # hairpin
        ("GCGCAAAAGCGC", "dna"),                               # DNA hairpin
        ("GGGAAACCCAAAGGGAAACCC", "rna"),                      # two stems → multiloop
        ("GCGCAAAAGCGCAAAGCGCAAAAGCGC", "dna"),                # DNA multiloop
        ("GGGAAACCCAAAGGGAAACCCAAAGGGAAACCC", "rna"),          # cloverleaf (3-way)
    ])
    def test_identity_holds(self, seq, material):
        assert _marginal_max_error(seq, material) < 1e-9

    @pytest.mark.parametrize("seq,material", [
        ("GCGCAAAAGCGC", "dna"),
        ("GGGAAACCCAAAGGGAAACCCAAAGGGAAACCC", "rna"),
    ])
    def test_identity_holds_physiological_salt(self, seq, material):
        # the outside enclosed terms must re-apply the per-pair salt factor
        assert _marginal_max_error(seq, material, sodium=0.137, magnesium=0.01) < 1e-9

    def test_multistrand_marginals_exact_including_nick(self):
        # Multi-strand pair probabilities are the exact adjoint of the inside
        # recurrence at EVERY position, including the two bases flanking the nick
        # (the immediate nick-junction pair i, i+1 across a strand boundary).
        # The unpaired-marginal identity therefore holds to numerical precision
        # everywhere — no nick-flank exemption.
        s1, s2 = "GGGAAAGGG", "CCCAAACCC"
        _, P = multistrand_pairs([s1, s2], 37.0, "rna")
        Znd = dangle_free_partition(sequences=[s1, s2], material="rna")
        n = len(s1) + len(s2)
        for i in range(n):
            Zb = dangle_free_partition(sequences=[s1, s2], material="rna", blocked={i})
            s = sum(P[i][j] for j in range(n) if j != i)
            assert abs(s + Zb / Znd - 1.0) < 1e-9

    def test_multistrand_bounds(self):
        s1, s2 = "AUGGGCAU", "AUGCCCAU"
        _, P = multistrand_pairs([s1, s2], 37.0, "rna")
        assert P.min() >= -1e-9
        assert P.max() <= 1.0 + 1e-6        # no gross > 1 (was ~1.4 before)
        assert np.allclose(P, P.T)


# ─── probability-matrix invariants ──────────────────────────────────────────────

class TestPairProbInvariants:
    def test_bounds_and_symmetry(self):
        _, P = ensemble_dg("GGGAAACCCAAAGGGAAACCCAAAGGGAAACCC", 37.0, "rna")
        assert P.min() >= -1e-12
        assert P.max() <= 1.0 + 1e-9
        assert np.allclose(P, P.T)
        # each base pairs with total probability ≤ 1
        for i in range(P.shape[0]):
            assert P[i].sum() <= 1.0 + 1e-9

    def test_deterministic(self):
        _, P1 = ensemble_dg("GCGCAAAAGCGCAAAGCGCAAAAGCGC", 37.0, "dna")
        _, P2 = ensemble_dg("GCGCAAAAGCGCAAAGCGCAAAAGCGC", 37.0, "dna")
        assert np.array_equal(P1, P2)


# ─── multiloops are no longer underestimated ────────────────────────────────────

class TestMultiloopNotUnderestimated:
    def test_cloverleaf_branches_have_probability(self):
        # A 3-way junction: each 3-bp stem closes a branch of a multiloop.  The
        # OLD recurrence (no multiloop outside term) drove these toward zero.
        seq = "GGGAAACCCAAAGGGAAACCCAAAGGGAAACCC"
        _, P = ensemble_dg(seq, 37.0, "rna")
        expected_bp = sum(P[i][j] for i in range(len(seq)) for j in range(i + 1, len(seq)))
        # three short stems can coexist in the multiloop → several expected pairs
        assert expected_bp > 3.0
        # the outermost branch stem (0,8) sits inside the junction and must carry
        # real probability, not ~0
        assert P[0].sum() > 0.1
