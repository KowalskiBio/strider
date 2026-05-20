"""
Tests for the benchmark suite in :mod:`strider.benchmarks`.

Two concerns:

1. The metric helpers (sensitivity / PPV / F-measure / relative error)
   produce the standard formulas on hand-checked inputs.
2. The runners import, execute end-to-end, and produce results that meet
   the receipts we want to publish — F-measure above 0.95 on the canonical
   reference set, 0% relative error on the Zhang TMSD lookup, monotone
   Arrhenius extrapolation, and finite timing rows.

These are functional checks, not numerical benchmarks; the actual
performance numbers come from running ``scripts/bench_accuracy.py``.
"""

from __future__ import annotations

import pytest

from strider.benchmarks import (
    REFERENCES,
    StructureMetrics,
    dot_bracket_metrics,
    get_references,
    mean_abs_dG_diff,
    pair_set_metrics,
    relative_error,
    run_structure_benchmark,
    run_timing_benchmark,
    run_tmsd_benchmark,
)


# ─── metric helpers ──────────────────────────────────────────────────────────


class TestPairSetMetrics:
    def test_exact_match_gives_unity_metrics(self):
        ref = {(0, 9), (1, 8), (2, 7)}
        pred = {(0, 9), (1, 8), (2, 7)}
        m = pair_set_metrics(ref, pred)
        assert m.sensitivity == 1.0
        assert m.ppv == 1.0
        assert m.f_measure == 1.0
        assert m.exact_match is True

    def test_extra_pair_drops_ppv_only(self):
        ref = {(0, 9), (1, 8)}
        pred = {(0, 9), (1, 8), (2, 7)}
        m = pair_set_metrics(ref, pred)
        assert m.sensitivity == 1.0
        assert m.ppv == pytest.approx(2 / 3)
        assert m.exact_match is False

    def test_missing_pair_drops_sensitivity_only(self):
        ref = {(0, 9), (1, 8), (2, 7)}
        pred = {(0, 9), (1, 8)}
        m = pair_set_metrics(ref, pred)
        assert m.sensitivity == pytest.approx(2 / 3)
        assert m.ppv == 1.0

    def test_empty_reference_and_prediction(self):
        m = pair_set_metrics(set(), set())
        assert m.sensitivity == 1.0 and m.ppv == 1.0 and m.f_measure == 1.0


class TestDotBracketMetrics:
    def test_identical_dot_bracket(self):
        m = dot_bracket_metrics("((....))", "((....))")
        assert m.exact_match

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            dot_bracket_metrics("((..))", "((....))")


class TestRelativeError:
    def test_zero_zero(self):
        assert relative_error(0.0, 0.0) == 0.0

    def test_zero_ref_nonzero_obs(self):
        assert relative_error(1.0, 0.0) == float("inf")

    def test_basic(self):
        assert relative_error(110.0, 100.0) == pytest.approx(0.1)

    def test_mean_abs_dG_diff(self):
        assert mean_abs_dG_diff([1.0, 2.0, 3.0], [1.0, 2.5, 4.0]) == pytest.approx(
            (0.0 + 0.5 + 1.0) / 3
        )


# ─── reference dataset ──────────────────────────────────────────────────────


class TestReferenceDataset:
    def test_reference_list_non_empty(self):
        assert len(REFERENCES) >= 10

    def test_filter_by_material(self):
        rna = get_references("rna")
        dna = get_references("dna")
        assert len(rna) > 0 and len(dna) > 0
        assert all(r.material == "rna" for r in rna)
        assert all(r.material == "dna" for r in dna)

    def test_every_reference_has_consistent_lengths(self):
        for r in REFERENCES:
            assert len(r.sequence) == len(r.structure), r.name


# ─── runner outputs ─────────────────────────────────────────────────────────


@pytest.mark.slow
class TestStructureRunner:
    """The structure runner folds every reference — count it as slow."""

    def test_runner_produces_high_quality_metrics(self):
        report = run_structure_benchmark(include_vienna=False)
        assert report.mean_f_measure >= 0.95, (
            f"mean F-measure on the canonical hairpin set should be ≥ 0.95, "
            f"got {report.mean_f_measure:.3f}"
        )
        assert report.n_exact >= len(report.per_sequence) - 2, (
            f"expected at most 2 non-exact matches, got "
            f"{len(report.per_sequence) - report.n_exact}"
        )

    def test_metrics_are_in_unit_interval(self):
        report = run_structure_benchmark(include_vienna=False)
        for s in report.per_sequence:
            assert 0.0 <= s.metrics.sensitivity <= 1.0
            assert 0.0 <= s.metrics.ppv <= 1.0
            assert 0.0 <= s.metrics.f_measure <= 1.0


class TestTMSDRunner:
    """TMSD lookup must round-trip exactly — no slow marker needed."""

    def test_zhang_lookup_self_consistent(self):
        report = run_tmsd_benchmark()
        assert report.max_rel_error == 0.0
        assert report.arrhenius_monotonic is True
        assert len(report.points_25C) == 13   # ZW Fig. 4 covers 0–12 nt toeholds


@pytest.mark.slow
class TestTimingRunner:
    def test_timing_returns_finite_positive_numbers(self):
        report = run_timing_benchmark(lengths=[20], reps=2, include_vienna=False)
        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.mfe_median_ms > 0
        assert row.pfunc_median_ms > 0
