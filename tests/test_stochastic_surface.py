"""
Low-copy stochastic surface capture — shot-noise-limited detection (Frontier §6).

The deterministic ``SurfaceModel`` assumes an infinite, non-depleting reservoir
and calls an assay detectable when the *mean* signal clears the electronics
floor.  At fM–aM that is physically wrong: the aliquot holds only ~10¹–10⁴
molecules, capture is a counting process, and Poisson shot noise — not the
amplifier — sets the limit of detection.  These tests pin the Currie (1968)
counting-statistics framework, the depletion-corrected capture mean, the
resulting shot-noise-limited LOD, and the mantis Gillespie SSA driver.
"""
import math

import numpy as np
import pytest

from strider.surface import (
    SurfaceModel,
    SurfaceParams,
    StochasticSurfaceModel,
    CurrieLevels,
    currie_levels,
    detection_probability,
    readout_sigma_counts,
    K_CURRIE,
)


def _const_trace(c, t_end=5400.0, n=40):
    times = np.linspace(0.0, t_end, n)
    return times, np.full_like(times, c)


# ── Currie detection-limit framework ──────────────────────────────────────────

class TestCurrieLevels:
    def test_zero_background_floor_is_k_squared(self):
        # perfect electronics, zero blank → the textbook 2.71-count counting floor
        lv = currie_levels(0.0, 0.0)
        assert lv.critical_level == pytest.approx(0.0)
        assert lv.detection_limit == pytest.approx(K_CURRIE ** 2)
        assert lv.detection_limit == pytest.approx(2.706, abs=0.01)

    def test_formula(self):
        lv = currie_levels(blank_mean=10.0, sigma_read=100.0)
        sigma0 = math.sqrt(10.0 + 100.0 ** 2)
        assert lv.sigma_blank == pytest.approx(sigma0)
        assert lv.critical_level == pytest.approx(K_CURRIE * sigma0)
        assert lv.detection_limit == pytest.approx(2 * lv.critical_level + K_CURRIE ** 2)

    def test_detection_limit_never_below_shot_noise_floor(self):
        for mu_b, sr in [(0, 0), (5, 0), (0, 50), (100, 200)]:
            assert currie_levels(mu_b, sr).detection_limit >= K_CURRIE ** 2 - 1e-9


class TestReadoutSigma:
    def test_sigma_is_charge_floor_over_event_charge(self):
        p = SurfaceParams()
        expected = p.readout.charge_floor_C() / p.label.signal_per_event()
        assert readout_sigma_counts(p) == pytest.approx(expected)
        assert readout_sigma_counts(p) > 0


# ── detection probability ─────────────────────────────────────────────────────

class TestDetectionProbability:
    def setup_method(self):
        self.lv = currie_levels(blank_mean=0.0, sigma_read=100.0)

    def test_monotonic_in_signal(self):
        prev = -1.0
        for mu in [0, 50, 100, 200, 400, 800]:
            p = detection_probability(mu, self.lv)
            assert p >= prev
            prev = p

    def test_false_positive_rate_at_zero_signal(self):
        # at μ→0 the test fires at the design false-positive rate α (=Φ(−k)≈0.05)
        assert detection_probability(0.0, self.lv) == pytest.approx(0.05, abs=0.01)

    def test_half_at_critical_level(self):
        assert detection_probability(self.lv.critical_level, self.lv) == pytest.approx(0.5, abs=1e-6)

    def test_confidence_at_detection_limit(self):
        # L_D is constructed so a true signal there is caught with prob 1−β≈0.95
        assert detection_probability(self.lv.detection_limit, self.lv) == pytest.approx(0.95, abs=0.01)


# ── depletion-corrected mean + shot-noise LOD ─────────────────────────────────

class TestShotNoiseLOD:
    def setup_method(self):
        self.p = SurfaceParams()
        self.sto = StochasticSurfaceModel(self.p)
        self.det = SurfaceModel(self.p)

    def test_capture_mean_capped_by_molecule_budget(self):
        # at 1 aM only ~30 molecules exist; the mean capture cannot exceed them,
        # unlike the deterministic transducer which "captures" thousands.
        c = 1e-18
        t, s = _const_trace(c)
        n_total = c * (self.p.sample_volume_L * 6.02214076e23)
        mu = self.sto.capture_mean(t, s)
        assert mu <= n_total + 1e-6
        assert self.det.transduce(t, s).n_captured > 10 * n_total  # overcount

    def test_capture_mean_monotonic(self):
        prev = -1.0
        for c in [1e-18, 1e-17, 1e-16, 1e-15]:
            mu = self.sto.capture_mean(*_const_trace(c))
            assert mu > prev
            prev = mu

    def test_shot_noise_lod_above_deterministic(self):
        triggers = np.array([1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14])
        det_lod = self.det.lod(_const_trace, triggers)
        sto_lod = self.sto.shot_noise_lod(_const_trace, triggers)
        assert det_lod is not None and sto_lod is not None
        # the deterministic LOD is optimistic (infinite reservoir); the
        # shot-noise LOD is strictly higher and physical.
        assert sto_lod > det_lod

    def test_lod_none_when_unreachable(self):
        # only vanishing triggers → never reach L_D
        assert self.sto.shot_noise_lod(_const_trace, [1e-21, 1e-20]) is None


# ── mantis Gillespie SSA driver ───────────────────────────────────────────────

class TestCaptureSSA:
    def setup_method(self):
        pytest.importorskip("mantis")
        self.p = SurfaceParams()
        self.sto = StochasticSurfaceModel(self.p)

    def test_ssa_matches_poisson(self):
        # ~1500-molecule aliquot: captured count ~ Poisson(μ) ⇒ mean=var=μ
        c = 5e-17
        t, s = _const_trace(c)
        samp = self.sto.simulate_capture(t, s, n_trajectories=150, seed=1)
        mu = samp.mean_signal
        se = math.sqrt(mu / 150)
        assert samp.empirical_mean == pytest.approx(mu, abs=5 * se)
        # Poisson variance ≈ mean (loose band: variance estimator is noisy)
        assert 0.6 * mu <= samp.empirical_var <= 1.5 * mu

    def test_ssa_mean_matches_capture_mean(self):
        t, s = _const_trace(5e-17)
        samp = self.sto.simulate_capture(t, s, n_trajectories=120, seed=4)
        assert samp.empirical_mean == pytest.approx(self.sto.capture_mean(t, s), rel=0.1)

    def test_detection_rate_increases_with_concentration(self):
        lv = self.sto.levels()
        lo = self.sto.simulate_capture(*_const_trace(1e-17), n_trajectories=120, seed=2)
        hi = self.sto.simulate_capture(*_const_trace(5e-17), n_trajectories=120, seed=2)
        assert lo.detection_rate(lv) < hi.detection_rate(lv)

    def test_max_molecules_guard(self):
        with pytest.raises(ValueError):
            self.sto.simulate_capture(*_const_trace(1e-12), n_trajectories=2,
                                      max_molecules=10_000)
