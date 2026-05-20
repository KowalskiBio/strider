"""
Accuracy metrics for secondary-structure prediction benchmarks.

The three numbers anyone reports for RNA structure prediction are
**sensitivity**, **PPV (positive predictive value)**, and the
**F-measure** (harmonic mean of the two).  Defined over the set of
base pairs in the reference vs. the predicted structure:

* sensitivity = |TP| / |reference|
* PPV         = |TP| / |predicted|
* F-measure   = 2 · sensitivity · PPV / (sensitivity + PPV)

This is the same formulation used by Mathews 2004 (NAR 32:1655-1668),
Lu, Turner, Mathews 2006, and ViennaRNA's ``RNAeval`` accuracy reports —
so the numbers strider emits are directly comparable to anything in the
RNA-folding literature.

We also expose a helper that compares two dot-bracket strings directly,
because that's the dominant call shape in the benchmark runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from strider.structure.dot_bracket import parse_pairs


@dataclass(frozen=True)
class StructureMetrics:
    """Base-pair-set comparison between a reference and a prediction."""

    sensitivity: float
    ppv: float
    f_measure: float
    n_reference_pairs: int
    n_predicted_pairs: int
    n_true_positive: int

    @property
    def exact_match(self) -> bool:
        """True iff every reference pair is predicted and no extra pairs."""
        return (
            self.n_true_positive == self.n_reference_pairs
            and self.n_predicted_pairs == self.n_reference_pairs
        )


def pair_set_metrics(
    reference_pairs: set[tuple[int, int]],
    predicted_pairs: set[tuple[int, int]],
) -> StructureMetrics:
    """
    Compute sensitivity / PPV / F-measure from two sets of (i, j) pairs.

    Both sets must use the same indexing convention (typically 0-based,
    ``i < j``).  Empty reference + empty prediction returns 1.0 for all
    three metrics; empty reference + non-empty prediction returns
    sensitivity = 1.0 (vacuously) but PPV = 0.
    """
    tp = len(reference_pairs & predicted_pairs)
    n_ref = len(reference_pairs)
    n_pred = len(predicted_pairs)

    sens = tp / n_ref if n_ref > 0 else 1.0
    ppv = tp / n_pred if n_pred > 0 else (1.0 if n_ref == 0 else 0.0)
    if sens + ppv > 0:
        f = 2 * sens * ppv / (sens + ppv)
    else:
        f = 0.0
    return StructureMetrics(
        sensitivity=sens, ppv=ppv, f_measure=f,
        n_reference_pairs=n_ref, n_predicted_pairs=n_pred,
        n_true_positive=tp,
    )


def dot_bracket_metrics(reference: str, predicted: str) -> StructureMetrics:
    """
    Convenience wrapper: parse two dot-bracket strings of equal length and
    score the predicted structure against the reference.
    """
    if len(reference) != len(predicted):
        raise ValueError(
            f"reference (len {len(reference)}) and predicted (len {len(predicted)}) "
            f"must be the same length"
        )
    ref_pairs = {tuple(sorted(p)) for p in parse_pairs(reference)}
    pred_pairs = {tuple(sorted(p)) for p in parse_pairs(predicted)}
    return pair_set_metrics(ref_pairs, pred_pairs)


# ─── ΔG agreement ────────────────────────────────────────────────────────────


def relative_error(observed: float, reference: float) -> float:
    """|obs − ref| / |ref|; returns 0 if both are zero, ``inf`` if only ref is zero."""
    if reference == 0.0:
        return 0.0 if observed == 0.0 else float("inf")
    return abs(observed - reference) / abs(reference)


def mean_abs_dG_diff(values_a: list[float], values_b: list[float]) -> float:
    """Mean of ``|a_i − b_i|`` (kcal/mol).  Lists must be the same length."""
    if len(values_a) != len(values_b):
        raise ValueError("list length mismatch")
    if not values_a:
        return 0.0
    return sum(abs(a - b) for a, b in zip(values_a, values_b)) / len(values_a)
