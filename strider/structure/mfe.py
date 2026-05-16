"""
MFE secondary structure prediction via dynamic programming.

Implements a simplified Zuker-style algorithm using nearest-neighbor
stacking energies from SantaLucia 2004 / Turner 2004, with hairpin loop
penalties from Turner tables.

For production accuracy on long sequences, use the ViennaRNA backend.
This native implementation is designed to be correct for short hairpins
(< 60 nt) and diagnostic verification tasks common in CHA design.
"""

from __future__ import annotations
import math
import numpy as np

INF = float("inf")
R = 1.987e-3  # kcal / (mol · K)

MIN_HAIRPIN_LOOP = 3  # minimum unpaired bases in hairpin


def fold_mfe(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
) -> tuple[str, float, list[tuple[int, int]]]:
    """
    Predict MFE secondary structure for a single strand.

    Returns:
        structure  : dot-bracket string
        energy     : MFE (kcal/mol, negative = stable)
        pairs      : list of (i, j) base-pair tuples (0-based)
    """
    seq = _normalize(sequence, material)
    n = len(seq)
    if n == 0:
        return "", 0.0, []

    # DP table: V[i][j] = min energy structure where (i,j) is a base pair
    #           W[i][j] = min energy for subsequence [i..j]
    V = np.full((n, n), INF)
    W = np.full((n, n), 0.0)
    traceback = {}  # (i, j) -> ("hp"|"stack"|"split", args)

    can = _can_pair_fn(material)
    stack_e = _stack_fn(material)

    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # V[i][j] — only if (i,j) can pair
            if can(seq, i, j) and j - i > MIN_HAIRPIN_LOOP:
                # Hairpin
                hp_e = _hairpin_energy(seq, i, j, celsius, material)
                V[i][j] = hp_e
                traceback[(i, j)] = ("hp",)

                # Stack: (i,j) wraps (i+1, j-1)
                if can(seq, i + 1, j - 1) and j - i > MIN_HAIRPIN_LOOP + 1:
                    e_stack = stack_e(seq, i, j) + V[i + 1][j - 1]
                    if e_stack < V[i][j]:
                        V[i][j] = e_stack
                        traceback[(i, j)] = ("stack", i + 1, j - 1)

                # Bifurcation inside (i,j)
                for k in range(i + 1, j):
                    e_bif = V[i][k] + W[k + 1][j - 1] if k + 1 <= j - 1 else V[i][k]
                    if e_bif < V[i][j]:
                        V[i][j] = e_bif
                        traceback[(i, j)] = ("bif", k)

            # W[i][j]
            W[i][j] = W[i][j - 1]  # j unpaired
            for k in range(i, j + 1):
                if V[k][j] < INF:
                    left = W[i][k - 1] if k > i else 0.0
                    e = left + V[k][j]
                    if e < W[i][j]:
                        W[i][j] = e

    # Traceback
    pairs: list[tuple[int, int]] = []
    _traceback_w(W, V, traceback, seq, can, 0, n - 1, pairs)

    energy = W[0][n - 1]
    structure = _to_dot_bracket(pairs, n)
    return structure, float(energy), sorted(pairs)


# ─── energy functions ─────────────────────────────────────────────────────────

def _hairpin_energy(seq: str, i: int, j: int, celsius: float, material: str) -> float:
    """Turner table hairpin loop free energy (kcal/mol) for the closing pair (i, j)."""
    loop = j - i - 1
    if loop < MIN_HAIRPIN_LOOP:
        return INF
    T = celsius + 273.15
    if material == "rna":
        if loop <= 9:
            table = [INF, INF, INF, 5.4, 5.6, 5.7, 5.4, 6.0, 5.5, 6.4]
            base = table[loop]
        else:
            base = 5.4 + 1.75 * R * T * math.log(loop / 9.0)
    else:
        if loop <= 9:
            table = [INF, INF, INF, 4.1, 4.3, 4.9, 4.4, 4.3, 4.1, 4.0]
            base = table[loop]
        else:
            base = 4.0 + 1.75 * R * T * math.log(loop / 9.0)
    return base


def _can_pair_fn(material: str):
    """Return a closure can(seq, i, j) that checks Watson-Crick pairing for the given material."""
    if material == "rna":
        wc = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
    else:
        wc = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}

    def can(seq: str, i: int, j: int) -> bool:
        return (seq[i], seq[j]) in wc and (j - i) > MIN_HAIRPIN_LOOP

    return can


def _stack_fn(material: str):
    """Return a closure stack(seq, i, j) that looks up the ΔG37 stacking energy for pair (i, j)."""
    if material == "dna":
        from strider.thermo.nn_dna import DNA_NN
        params = DNA_NN
    else:
        from strider.thermo.nn_rna import RNA_NN
        params = RNA_NN

    def stack(seq: str, i: int, j: int) -> float:
        dinuc = seq[i] + seq[i + 1]
        if dinuc in params:
            return params[dinuc][2]
        return -1.5

    return stack


# ─── traceback ────────────────────────────────────────────────────────────────

def _traceback_w(W, V, tb, seq, can, i, j, pairs):
    """Traceback through the W table to recover base-pair list for subsequence [i..j]."""
    if i >= j:
        return
    if W[i][j] == W[i][j - 1]:
        _traceback_w(W, V, tb, seq, can, i, j - 1, pairs)
        return
    for k in range(i, j + 1):
        if V[k][j] < INF:
            left = W[i][k - 1] if k > i else 0.0
            if abs(left + V[k][j] - W[i][j]) < 1e-9:
                if k > i:
                    _traceback_w(W, V, tb, seq, can, i, k - 1, pairs)
                _traceback_v(V, tb, seq, can, k, j, pairs)
                return


def _traceback_v(V, tb, seq, can, i, j, pairs):
    """Traceback through the V table starting at forced pair (i, j), appending to pairs."""
    if (i, j) not in tb:
        return
    pairs.append((i, j))
    action = tb[(i, j)]
    if action[0] == "hp":
        return
    if action[0] == "stack":
        _traceback_v(V, tb, seq, can, action[1], action[2], pairs)
    elif action[0] == "bif":
        k = action[1]
        _traceback_v(V, tb, seq, can, i, k, pairs)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _normalize(seq: str, material: str) -> str:
    """Uppercase seq and convert T↔U according to material ('dna' or 'rna')."""
    seq = seq.upper()
    if material == "dna":
        return seq.replace("U", "T")
    return seq.replace("T", "U")


def _to_dot_bracket(pairs: list[tuple[int, int]], n: int) -> str:
    """Convert a list of (i, j) pairs to a length-n dot-bracket string."""
    db = ["."] * n
    for i, j in pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)
