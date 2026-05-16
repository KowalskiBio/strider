"""
Mutation sensitivity analysis for nucleic acid sequences.

Computes how much each nucleotide position contributes to a thermodynamic
property. Identifies critical and robust positions for biosensor design.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.sweep.cache import DiskCache

DNA_ALTS = {"A": ["C", "G", "T"], "C": ["A", "G", "T"],
            "G": ["A", "C", "T"], "T": ["A", "C", "G"]}
RNA_ALTS = {"A": ["C", "G", "U"], "C": ["A", "G", "U"],
            "G": ["A", "C", "U"], "U": ["A", "C", "G"]}


@dataclass
class MutationProfile:
    """
    Single-nucleotide mutation sensitivity profile.

    Attributes
    ----------
    sequence        : original sequence
    positions       : array of position indices [0, n-1]
    delta_score     : shape (n, 3) — ΔScore for each alt nucleotide at each position
    alt_nucleotides : shape (n, 3) — alternative bases tried
    robustness      : fraction of single-nt mutations within tolerance
    """
    sequence: str
    positions: np.ndarray
    delta_score: np.ndarray
    alt_nucleotides: list[list[str]]
    robustness: float

    @property
    def max_sensitivity(self) -> np.ndarray:
        """Worst-case (max absolute) ΔScore at each position."""
        return np.max(np.abs(self.delta_score), axis=1)

    def critical_positions(self, threshold: float = 2.0) -> list[int]:
        """Positions where any mutation changes score by > threshold."""
        return [i for i in range(len(self.sequence)) if self.max_sensitivity[i] > threshold]

    def robust_positions(self, tolerance: float = 0.5) -> list[int]:
        """Positions where ALL mutations keep score within tolerance."""
        return [i for i in range(len(self.sequence)) if self.max_sensitivity[i] <= tolerance]

    def plot(self, ax=None, title: str = "Mutation Sensitivity"):
        """Heatmap of ΔScore per position and alternative nucleotide."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(max(8, len(self.sequence) // 3), 3))
        im = ax.imshow(
            self.delta_score.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-max(1, self.delta_score.max()),
            vmax=max(1, self.delta_score.max()),
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("Alt nucleotide")
        ax.set_xticks(range(len(self.sequence)))
        ax.set_xticklabels(list(self.sequence), fontsize=7)
        ax.set_yticks(range(3))
        if self.alt_nucleotides:
            ax.set_yticklabels(self.alt_nucleotides[0])
        plt.colorbar(im, ax=ax, label="ΔScore (kcal/mol)")
        ax.set_title(title)
        return ax


class MutationAnalyzer:
    """
    Analyze how single-nucleotide mutations affect a thermodynamic metric.

    Parameters
    ----------
    engine  : ThermoEngine for computations
    cache   : optional DiskCache for persistent memoization
    """

    def __init__(
        self,
        engine: "ThermoEngine",
        cache: "DiskCache | None" = None,
    ) -> None:
        self.engine = engine
        self.cache = cache

    def single_nt_scan(
        self,
        sequence: str,
        target: str | None = None,
        metric: Callable[[str, "ThermoEngine"], float] | None = None,
        tolerance: float = 1.0,
    ) -> MutationProfile:
        """
        Scan all single-nucleotide mutations of a sequence.

        sequence : sequence to mutate
        target   : hybridization partner; if None → hairpin metric
        metric   : custom fn(seq, engine) -> float; default is duplex_dg or pfunc
        tolerance: kcal/mol tolerance for robustness scoring
        """
        seq = sequence.upper().replace("U", "T")
        n = len(seq)
        material = self.engine.material
        alts_map = RNA_ALTS if material == "rna" else DNA_ALTS

        if metric is None:
            if target is not None:
                tgt = target.upper().replace("U", "T")
                base_score = self.engine.duplex_dg(seq, tgt)
                def metric(s, eng): return eng.duplex_dg(s, tgt)
            else:
                base_score = self.engine.pfunc(seq).free_energy
                def metric(s, eng): return eng.pfunc(s).free_energy
        else:
            base_score = metric(seq, self.engine)

        delta_score = np.zeros((n, 3))
        alt_nucleotides: list[list[str]] = []

        for pos in range(n):
            orig = seq[pos]
            alts = alts_map.get(orig, [b for b in "ACGT" if b != orig])
            alt_nucleotides.append(alts[:3])
            for j, alt in enumerate(alts[:3]):
                mutant = seq[:pos] + alt + seq[pos + 1:]
                try:
                    score = metric(mutant, self.engine)
                except Exception:
                    score = base_score
                delta_score[pos, j] = score - base_score

        n_robust = sum(
            1 for i in range(n) if np.max(np.abs(delta_score[i])) <= tolerance
        )
        robustness = n_robust / n if n > 0 else 1.0

        return MutationProfile(
            sequence=seq,
            positions=np.arange(n),
            delta_score=delta_score,
            alt_nucleotides=alt_nucleotides,
            robustness=robustness,
        )

    def robustness_score(
        self,
        sequence: str,
        target: str | None = None,
        ddg_tolerance: float = 1.0,
    ) -> float:
        """
        Fraction of single-nt mutations within ddg_tolerance (kcal/mol).
        """
        profile = self.single_nt_scan(sequence, target, tolerance=ddg_tolerance)
        return profile.robustness

    def critical_positions(
        self,
        sequence: str,
        target: str | None = None,
        sensitivity_threshold: float = 2.0,
    ) -> list[int]:
        """Positions where any mutation changes ΔΔG by > threshold."""
        profile = self.single_nt_scan(sequence, target)
        return profile.critical_positions(sensitivity_threshold)
