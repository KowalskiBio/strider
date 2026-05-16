"""
TMSD kinetics tests.

Validated against Zhang & Winfree (2009) JACS 131:17303 Fig. 4.
"""
import math
import pytest
from strider.kinetics.tmsd import (
    toehold_kf, displacement_kf, leakage_kf, rates_from_ddg,
    TMSDKineticModel,
)
from strider.kinetics.arrhenius import detailed_balance_kr, k_eq_from_ddg


class TestToeholdKF:
    def test_zero_toehold_very_slow(self):
        kf = toehold_kf(0)
        assert kf < 1e3, f"Zero toehold should be very slow, got {kf:.2e}"

    def test_six_nt_approx_3e5(self):
        kf = toehold_kf(6, celsius=25.0)
        assert 1e5 < kf < 1e6, f"6-nt toehold at 25°C ≈ 3e5, got {kf:.2e}"

    def test_seven_nt_approx_1e6(self):
        kf = toehold_kf(7, celsius=25.0)
        assert 5e5 < kf < 5e6, f"7-nt toehold at 25°C ≈ 1e6, got {kf:.2e}"

    def test_monotonically_increasing(self):
        kfs = [toehold_kf(n) for n in range(0, 12)]
        for i in range(len(kfs) - 1):
            assert kfs[i] <= kfs[i + 1], f"kf should increase with toehold length"

    def test_saturates_at_long_toehold(self):
        kf_10 = toehold_kf(10)
        kf_12 = toehold_kf(12)
        ratio = kf_12 / kf_10
        assert ratio < 3.0, "kf should saturate for long toeholds"

    def test_temperature_increases_rate(self):
        kf_25 = toehold_kf(6, celsius=25.0)
        kf_37 = toehold_kf(6, celsius=37.0)
        assert kf_37 > kf_25, "Higher temperature → faster rate"

    def test_rna_slightly_faster(self):
        kf_dna = toehold_kf(6, material="dna")
        kf_rna = toehold_kf(6, material="rna")
        assert kf_rna > kf_dna, "RNA TMSD is slightly faster"


class TestDetailedBalance:
    def test_detailed_balance_consistency(self):
        """kr/kf must equal exp(ΔΔG/RT)."""
        ddg = -11.42
        celsius = 37.0
        kf = toehold_kf(6, celsius=celsius)
        kf_val, kr_val = rates_from_ddg(ddg, kf, celsius)

        R = 1.987e-3
        T = celsius + 273.15
        expected_ratio = math.exp(ddg / (R * T))
        actual_ratio = kr_val / kf_val

        assert abs(actual_ratio - expected_ratio) / abs(expected_ratio) < 0.01

    def test_favorable_reaction_kf_gt_kr(self):
        """Exergonic reaction (ΔΔG < 0) → kf > kr."""
        kf, kr = rates_from_ddg(-10.0, 1e5, 37.0)
        assert kf > kr

    def test_unfavorable_reaction_kr_gt_kf(self):
        """Endergonic reaction (ΔΔG > 0) → kr > kf."""
        kf, kr = rates_from_ddg(+5.0, 1e5, 37.0)
        assert kr > kf


class TestLeakageKF:
    def test_stable_stem_reduces_leakage(self):
        kf_stable = leakage_kf(stem_stability_kcal=12.0)
        kf_unstable = leakage_kf(stem_stability_kcal=4.0)
        assert kf_stable < kf_unstable

    def test_leakage_never_exceeds_max(self):
        kf = leakage_kf(stem_stability_kcal=0.0, kf_max=1e6)
        assert kf <= 1e6


class TestKEqConversions:
    def test_round_trip(self):
        from strider.kinetics.arrhenius import ddg_from_k_eq
        ddg_orig = -8.5
        keq = k_eq_from_ddg(ddg_orig)
        ddg_back = ddg_from_k_eq(keq)
        assert abs(ddg_back - ddg_orig) < 1e-6

    def test_negative_ddg_keq_gt_1(self):
        keq = k_eq_from_ddg(-5.0)
        assert keq > 1.0

    def test_positive_ddg_keq_lt_1(self):
        keq = k_eq_from_ddg(+5.0)
        assert keq < 1.0
