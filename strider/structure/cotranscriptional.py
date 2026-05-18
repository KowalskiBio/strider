"""
Co-transcriptional folding: structure trajectory as the strand is synthesized.

During transcription, RNA folds while it's still being polymerized — only the
5' portion exists at any given moment.  The mature MFE structure may not be
the folding outcome if a kinetic intermediate traps before the full sequence
is available (riboswitches, aptamers, some ribozymes).

This module sweeps prefix lengths and folds each prefix, producing a
``CotranscriptionalTrajectory`` showing how the structure evolves.  Use it
to predict which structures the polymerase actually produces.

References:
    Boyle J., Robillard G.T., Kim S.-H. (1980). Sequential folding of
        transfer RNA. J. Mol. Biol. 139: 601-625.
    Pan T., Sosnick T. (2006). RNA folding during transcription. Annu. Rev.
        Biophys. Biomol. Struct. 35: 161-175.
    Watters K.E. et al. (2016). Cotranscriptional folding of a riboswitch
        at nucleotide resolution. Nat. Struct. Mol. Biol. 23: 1124-1131.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from strider.structure.mfe import fold_mfe


@dataclass(frozen=True)
class PrefixFold:
    """Single prefix folded snapshot."""
    length: int                       # prefix length (number of nt synthesized)
    structure: str                    # dot-bracket of the prefix
    energy: float                     # MFE of the prefix (kcal/mol)
    pairs: tuple[tuple[int, int], ...]  # base pairs (0-based, inside prefix)


@dataclass
class CotranscriptionalTrajectory:
    """
    Result of a co-transcriptional folding sweep.

    Attributes:
        sequence : full sequence the trajectory was computed for
        prefixes : ordered list of PrefixFold, one per prefix length sampled
        celsius  : temperature used
        material : 'dna' or 'rna'

    The trajectory is monotone in prefix length but the **structure** is not
    monotone — pairs can break as new 3' nucleotides allow more favorable
    configurations.  Compare adjacent frames to find structural rearrangements.
    """
    sequence: str
    prefixes: list[PrefixFold]
    celsius: float
    material: str

    def at_length(self, length: int) -> Optional[PrefixFold]:
        """Return the PrefixFold at exactly ``length`` nt, or None."""
        for p in self.prefixes:
            if p.length == length:
                return p
        return None

    def final(self) -> PrefixFold:
        """The fully transcribed structure (last prefix)."""
        return self.prefixes[-1]

    def rearrangements(self) -> list[tuple[int, int]]:
        """
        Indices of consecutive prefixes whose pair sets differ by more than the
        addition of the newly transcribed base's pairings.

        Returns list of (prev_length, next_length) tuples where a non-trivial
        rearrangement occurred (some existing pair broke).
        """
        out: list[tuple[int, int]] = []
        for prev, curr in zip(self.prefixes, self.prefixes[1:]):
            prev_pairs = set(prev.pairs)
            curr_pairs = set(curr.pairs)
            # Pairs that existed and no longer do (excluding pairs that just
            # got their partner trimmed off, which can't happen growing 3'-ward).
            broken = prev_pairs - curr_pairs
            if broken:
                out.append((prev.length, curr.length))
        return out


def fold_cotranscriptional(
    sequence: str,
    celsius: float = 37.0,
    material: str = "rna",
    min_length: int = 5,
    step: int = 1,
) -> CotranscriptionalTrajectory:
    """
    Fold every prefix of ``sequence`` to build a transcription-time trajectory.

    Args:
        sequence   : full strand sequence (RNA by default — co-transcriptional
                     folding is mostly an RNA phenomenon, but DNA is supported)
        celsius    : temperature for MFE folding
        material   : 'rna' or 'dna'
        min_length : skip prefixes shorter than this (no meaningful structure
                     possible below the minimum hairpin loop size + 2)
        step       : sample every ``step`` nucleotides instead of every one;
                     useful for long sequences when the full trajectory is
                     expensive

    Returns:
        CotranscriptionalTrajectory with one PrefixFold per sampled length.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    n = len(sequence)
    if n == 0:
        return CotranscriptionalTrajectory(sequence, [], celsius, material)

    start = max(min_length, 1)
    lengths = list(range(start, n + 1, step))
    if lengths[-1] != n:
        lengths.append(n)  # always include the full-length prefix

    prefixes: list[PrefixFold] = []
    for L in lengths:
        prefix = sequence[:L]
        try:
            structure, energy, pairs = fold_mfe(prefix, celsius=celsius, material=material)
        except Exception:
            # Below the minimum-loop threshold fold_mfe may return trivial output;
            # treat as fully unpaired.
            structure = "." * L
            energy = 0.0
            pairs = []
        prefixes.append(PrefixFold(
            length=L,
            structure=structure,
            energy=float(energy),
            pairs=tuple(tuple(p) for p in pairs),
        ))

    return CotranscriptionalTrajectory(
        sequence=sequence,
        prefixes=prefixes,
        celsius=celsius,
        material=material,
    )
