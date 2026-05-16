"""
Optional ViennaRNA backend for ThermoEngine.

ViennaRNA is GPL-licensed and must be installed separately:
    pip install ViennaRNA   (or via conda)

This module is imported lazily — strider works without it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def is_available() -> bool:
    """Return True if ViennaRNA (RNA module) is importable."""
    try:
        import RNA  # noqa: F401
        return True
    except ImportError:
        return False


def fold(sequence: str, celsius: float = 37.0) -> tuple[str, float]:
    """MFE structure and energy via ViennaRNA RNA.fold()."""
    import RNA
    md = RNA.md()
    md.temperature = celsius
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()
    return structure, mfe


def pf_fold(sequence: str, celsius: float = 37.0) -> tuple[float, "np.ndarray"]:
    """Ensemble ΔG and pair probability matrix via ViennaRNA."""
    import RNA
    import numpy as np
    md = RNA.md()
    md.temperature = celsius
    fc = RNA.fold_compound(sequence, md)
    fc.pf()
    dG_ens = fc.mean_bp_distance()   # proxy; real: fc.gibbs_free_energy()
    # Pair probability matrix
    n = len(sequence)
    bppm = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = fc.get_pr(i, j)
            bppm[i - 1][j - 1] = p
            bppm[j - 1][i - 1] = p
    # Get actual ensemble free energy
    try:
        dG_ens = fc.ensemble_defect(fc.mfe()[0])
    except Exception:
        pass
    gibbs = fc.pf()
    return float(gibbs), bppm


def co_fold(seq1: str, seq2: str, celsius: float = 37.0) -> tuple[str, float]:
    """Co-folding of two strands via ViennaRNA RNA.cofold()."""
    import RNA
    md = RNA.md()
    md.temperature = celsius
    joined = seq1 + "&" + seq2
    fc = RNA.fold_compound(joined, md)
    structure, mfe = fc.mfe_dimer()
    return structure, mfe
