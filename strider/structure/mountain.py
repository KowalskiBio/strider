"""
Mountain plot representation of secondary structure.

The mountain plot assigns each position a height equal to the number of
base pairs that enclose it (its nesting depth). Useful for comparing
structures and visualizing structural elements.
"""

from __future__ import annotations
import numpy as np


def mountain_vector(structure: str) -> np.ndarray:
    """
    Compute nesting depth for each position in a dot-bracket structure.

    Position i has height = number of pairs (a, b) with a < i < b.
    """
    clean = structure.replace("&", "").replace("+", "")
    n = len(clean)
    depth = np.zeros(n, dtype=int)
    current = 0
    for i, ch in enumerate(clean):
        if ch == "(":
            depth[i] = current
            current += 1
        elif ch == ")":
            current -= 1
            depth[i] = current
        else:
            depth[i] = current
    return depth


def mountain_from_probs(pair_probs: np.ndarray) -> np.ndarray:
    """
    Expected mountain height from a base-pair probability matrix.

    height[i] = Σ_{j: j≠i} P(i,j) × (1 if i enclosed by (i',j') else 0)

    Simplified: height[i] = Σ_{a<i} Σ_{b>i} P(a,b)
    """
    n = pair_probs.shape[0]
    height = np.zeros(n)
    for i in range(n):
        for a in range(i):
            for b in range(i + 1, n):
                height[i] += pair_probs[a, b]
    return height


def compare_structures(struct1: str, struct2: str) -> float:
    """
    Mountain distance between two dot-bracket structures.

    Returns the L1 norm of the difference in mountain vectors,
    normalized by sequence length. Range [0, 1].
    """
    m1 = mountain_vector(struct1).astype(float)
    m2 = mountain_vector(struct2).astype(float)
    n = max(len(m1), len(m2), 1)
    return float(np.sum(np.abs(m1 - m2))) / n
