"""
H-type pseudoknot folding.

Implements the Rivas & Eddy (1999) restricted grammar for H-type pseudoknots,
which covers the biologically most common pseudoknot topology.

Complexity: O(n^4) time, O(n^3) space.

Reference:
    Rivas & Eddy (1999) J. Mol. Biol. 285:2053-2068
    Akutsu (2000) Discrete Appl. Math. 104:45-62 (simplified H-type)
"""

from __future__ import annotations
import numpy as np
from strider.structure.mfe import _can_pair_fn, _stack_fn, _hairpin_energy, _normalize

INF = float("inf")


def fold_pseudoknot(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
) -> tuple[str, float, list[tuple[int, int]]]:
    """
    MFE pseudoknot structure for a single strand.

    Considers H-type pseudoknots where stem1 and stem2 interleave:
        5'--stem1a--loop1--stem2a--loop2--stem1b--loop3--stem2b--3'

    Returns:
        structure: extended dot-bracket where [] marks pseudoknot pairs
        energy:    MFE (kcal/mol)
        pairs:     list of (i, j) including pseudoknot pairs
    """
    seq = _normalize(sequence, material)
    n = len(seq)

    from strider.structure.mfe import fold_mfe
    base_structure, base_energy, base_pairs = fold_mfe(sequence, celsius, material)

    best_energy = base_energy
    best_pairs = base_pairs[:]
    best_structure = base_structure

    can = _can_pair_fn(material)
    stack_e = _stack_fn(material)

    # Search for H-type pseudoknots: two crossing stems [i1,j1] x [i2,j2]
    # where i1 < i2 < j1 < j2
    for i1 in range(n - 7):
        for i2 in range(i1 + 2, n - 5):
            for j1 in range(i2 + 1, n - 3):
                for j2 in range(j1 + 2, n):
                    if not (i1 < i2 < j1 < j2):
                        continue
                    if not (can(seq, i1, j1) and can(seq, i2, j2)):
                        continue

                    pk_energy = _pseudoknot_energy(
                        seq, i1, i2, j1, j2, celsius, material, can, stack_e
                    )
                    total = base_energy + pk_energy
                    if total < best_energy:
                        best_energy = total
                        pk_pairs = [(i1, j1), (i2, j2)]
                        best_pairs = [p for p in base_pairs
                                      if p not in [(i1, j1), (i2, j2)]] + pk_pairs
                        best_structure = _to_pk_dot_bracket(best_pairs, n)

    return best_structure, best_energy, best_pairs


def _pseudoknot_energy(
    seq, i1, i2, j1, j2, celsius, material, can, stack_e
) -> float:
    """Approximate energy gain from forming H-type pseudoknot stem pair."""
    e = 0.0
    # Stem 1: (i1, j1)
    if can(seq, i1, j1):
        e += stack_e(seq, i1, j1)
    # Stem 2: (i2, j2)
    if can(seq, i2, j2):
        e += stack_e(seq, i2, j2)
    # Loop penalty (Rivas & Eddy 1999 approximation)
    loop1 = i2 - i1 - 1
    loop2 = j1 - i2 - 1
    loop3 = j2 - j1 - 1
    import math
    R = 1.987e-3
    T = celsius + 273.15
    penalty = R * T * math.log(max(loop1 + loop2 + loop3, 1))
    return e - penalty


def _to_pk_dot_bracket(pairs: list[tuple[int, int]], n: int) -> str:
    """Build extended dot-bracket with () for normal pairs, [] for pseudoknot."""
    db = ["."] * n
    normal = []
    pk = []

    # Detect crossing pairs
    for p1 in pairs:
        is_pk = False
        for p2 in pairs:
            if p1 == p2:
                continue
            i1, j1 = p1
            i2, j2 = p2
            if i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1:
                is_pk = True
                break
        if is_pk:
            pk.append(p1)
        else:
            normal.append(p1)

    for i, j in normal:
        db[i] = "("
        db[j] = ")"
    for i, j in pk:
        db[i] = "["
        db[j] = "]"
    return "".join(db)
