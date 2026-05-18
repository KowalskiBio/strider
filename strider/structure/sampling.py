"""
Boltzmann sampling and suboptimal-structure enumeration.

Both procedures build on the McCaskill partition-function DP in
``strider.thermo.ensemble`` and the MFE DP in ``strider.structure.mfe``.

* ``sample_structures(seq, n, …)`` — Ding-Lawrence (2003) stochastic
  traceback over the Qb / Q matrices, yielding N structures distributed
  according to the equilibrium Boltzmann weights.

* ``subopt_structures(seq, gap, …)`` — Wuchty-style enumeration of all
  structures within ``gap`` kcal/mol of the MFE.  Uses a worklist of partial
  decompositions over the V / W matrices, pruned by a lower-bound energy
  estimate.

References
----------
Wuchty S., Fontana W., Hofacker I.L., Schuster P. (1999) Biopolymers
49:145-165.

Ding Y. & Lawrence C.E. (2003) Nucleic Acids Res. 31:7280-7301.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from strider.thermo.ensemble import (
    R, _MAX_IL, _wc_pairs, _can_pair_nicks,
    _hairpin_loop_energy, _stack_energy, _interior_bulge_energy,
    _terminal_pair_penalty, _boltzmann, _fill_dp_nicks,
    _apply_coaxial_external,
)
from strider.structure.dot_bracket import to_dot_bracket

if TYPE_CHECKING:
    pass


INF = float("inf")


# ─── Boltzmann sampling ───────────────────────────────────────────────────────

def sample_structures(
    sequence: str,
    n_samples: int,
    celsius: float = 37.0,
    material: str = "dna",
    seed: int | None = None,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Draw ``n_samples`` structures from the equilibrium ensemble.

    Returns a list of ``(dot_bracket, pair_list)`` tuples, each entry sampled
    independently with probability ∝ exp(−E / RT).
    """
    rng = random.Random(seed)
    seq = sequence.upper().replace("U", "T") if material == "dna" else sequence.upper().replace("T", "U")
    n = len(seq)
    T = celsius + 273.15
    pairs = _wc_pairs(material)

    Q  = np.zeros((n, n))
    Qb = np.zeros((n, n))
    QM = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    _fill_dp_nicks(seq, Q, Qb, QM, n, T, pairs, material, nicks=[])
    _apply_coaxial_external(seq, Q, Qb, n, T, material)

    results = []
    for _ in range(n_samples):
        pair_list: list[tuple[int, int]] = []
        _sample_Q(seq, 0, n - 1, Q, Qb, QM, T, pairs, material, pair_list, rng)
        pair_list.sort()
        results.append((to_dot_bracket(pair_list, n), pair_list))
    return results


def _sample_Q(seq, i, j, Q, Qb, QM, T, pairs, material, out_pairs, rng):
    """Stochastic traceback through Q[i][j] (external context)."""
    while i <= j:
        # Compute contributions: (j unpaired) + Σ_k (stem (k,j) with various dangles)
        total = Q[i][j]
        if total <= 0:
            return
        target = rng.random() * total
        cum = 0.0

        # Option A: j unpaired → recurse on [i, j-1]
        contrib = Q[i][j - 1] if j > i else 1.0
        cum += contrib
        if target < cum:
            j -= 1
            continue

        # Option B: close (k, j) with no dangles
        chosen = False
        for k in range(i, j + 1):
            if Qb[k][j] <= 0:
                continue
            left = Q[i][k - 1] if k > i else 1.0
            contrib = left * Qb[k][j]
            cum += contrib
            if target < cum:
                out_pairs.append((k, j))
                _sample_Qb(seq, k, j, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                # Continue with left subproblem [i..k-1]
                _sample_Q(seq, i, k - 1, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                chosen = True
                break
        if chosen:
            return
        # Numerical edge case: fall through with j unpaired
        j -= 1


def _sample_Qb(seq, i, j, Q, Qb, QM, T, pairs, material, out_pairs, rng):
    """Stochastic traceback through Qb[i][j] (forced pair at i, j)."""
    if Qb[i][j] <= 0:
        return
    target = rng.random() * Qb[i][j]
    cum = 0.0

    # 1. Hairpin
    cum += _boltzmann(_hairpin_loop_energy(seq, i, j, material, T), T)
    if target < cum:
        return

    # 2. Stack
    if _can_pair_nicks(seq, i + 1, j - 1, pairs, []) and Qb[i + 1][j - 1] > 0:
        cum += _boltzmann(_stack_energy(seq, i, j, material), T) * Qb[i + 1][j - 1]
        if target < cum:
            out_pairs.append((i + 1, j - 1))
            _sample_Qb(seq, i + 1, j - 1, Q, Qb, QM, T, pairs, material, out_pairs, rng)
            return

    # 3. Interior loop / bulge
    for nl in range(_MAX_IL + 1):
        ip = i + nl + 1
        if ip > j - 2:
            break
        for nr in range(_MAX_IL - nl + 1):
            if nl == 0 and nr == 0:
                continue
            jp = j - nr - 1
            if jp <= ip:
                break
            if not _can_pair_nicks(seq, ip, jp, pairs, []):
                continue
            if Qb[ip][jp] <= 0:
                continue
            dG = _interior_bulge_energy(seq, i, j, ip, jp, nl, nr, material)
            cum += Qb[ip][jp] * _boltzmann(dG, T)
            if target < cum:
                out_pairs.append((ip, jp))
                _sample_Qb(seq, ip, jp, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                return

    # 4. Multiloop closed by (i, j)
    if material == "dna":
        from strider.thermo.parameters_dna import ML_INIT, ML_PAIR
    else:
        from strider.thermo.parameters_rna import ML_INIT, ML_PAIR
    bm_ml = _boltzmann(ML_INIT + ML_PAIR, T)
    if j - i > 2:
        cum += bm_ml * QM[i + 1][j - 1]
        if target < cum:
            _sample_QM(seq, i + 1, j - 1, Q, Qb, QM, T, pairs, material, out_pairs, rng)
            return


def _sample_QM(seq, i, j, Q, Qb, QM, T, pairs, material, out_pairs, rng):
    """Stochastic traceback through QM[i][j] (multiloop region with ≥ 1 stem)."""
    if material == "dna":
        from strider.thermo.parameters_dna import ML_PAIR, ML_BASE
    else:
        from strider.thermo.parameters_rna import ML_PAIR, ML_BASE
    bm_ml_pair = _boltzmann(ML_PAIR, T)
    bm_ml_base = _boltzmann(ML_BASE, T)

    while i <= j:
        total = QM[i][j]
        if total <= 0:
            return
        target = rng.random() * total
        cum = 0.0
        # Option 1: single stem (i, j)
        if Qb[i][j] > 0:
            cum += Qb[i][j] * bm_ml_pair
            if target < cum:
                out_pairs.append((i, j))
                _sample_Qb(seq, i, j, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                return
        # Option 2: j unpaired
        if QM[i][j - 1] > 0:
            cum += QM[i][j - 1] * bm_ml_base
            if target < cum:
                j -= 1
                continue
        # Option 3: stem [k, j] + multiloop [i, k-1]
        for k in range(i + 1, j):
            if Qb[k][j] > 0 and QM[i][k - 1] > 0:
                cum += QM[i][k - 1] * Qb[k][j] * bm_ml_pair
                if target < cum:
                    out_pairs.append((k, j))
                    _sample_Qb(seq, k, j, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                    _sample_QM(seq, i, k - 1, Q, Qb, QM, T, pairs, material, out_pairs, rng)
                    return
        return  # safety fallthrough


# ─── Suboptimal structures ────────────────────────────────────────────────────

def subopt_structures(
    sequence: str,
    gap: float = 1.0,
    celsius: float = 37.0,
    material: str = "dna",
    max_structures: int = 200,
) -> list[tuple[str, float, list[tuple[int, int]]]]:
    """
    Enumerate suboptimal structures within ``gap`` kcal/mol of the MFE.

    Returns ``(dot_bracket, energy, pair_list)`` sorted by energy, capped at
    ``max_structures`` results.

    Implementation: Wuchty-style worklist over the V/W matrices, pruned by a
    lower-bound estimate (sum of W on open intervals).
    """
    from strider.structure.mfe import (
        _can_pair_fn, _stack_fn, _hairpin_energy, _normalize,
    )
    seq = _normalize(sequence, material)
    n = len(seq)
    if n == 0:
        return []

    # Re-run fold_mfe internals to get V and W; we re-compute here so we
    # don't have to alter the public fold_mfe signature.
    V, W = _fold_matrices(seq, celsius, material)
    mfe = float(W[0][n - 1])
    bound = mfe + gap

    results: list[tuple[frozenset, float]] = []

    can = _can_pair_fn(material)
    stack_e_fn = _stack_fn(material)

    def hp_fn(i, j):
        return _hairpin_energy(seq, i, j, celsius, material)

    def visit_W(i: int, j: int, committed: float, pairs: frozenset):
        """Enumerate all decompositions of W[i..j], appending complete structures to results."""
        if len(results) >= max_structures:
            return
        if i > j:
            results.append((pairs, committed))
            return
        # Optimistic remaining lower bound = W[i][j]
        if committed + W[i][j] > bound + 1e-6:
            return

        # Option A: j unpaired
        visit_W(i, j - 1, committed, pairs)

        # Option B: stem (k, j) for various k.  v_cap = max energy we can spend
        # on V[k][j] itself (everything else accounted for via W).
        for k in range(i, j + 1):
            if V[k][j] >= INF:
                continue
            left_lb = W[i][k - 1] if k > i else 0.0
            v_cap = bound - committed - left_lb
            if V[k][j] > v_cap + 1e-6:
                continue
            for v_pairs, v_e in _enum_V(seq, V, W, can, hp_fn, stack_e_fn, k, j, v_cap,
                                       max_structures - len(results)):
                new_pairs = pairs | v_pairs
                visit_W(i, k - 1, committed + v_e, new_pairs)
                if len(results) >= max_structures:
                    return

    visit_W(0, n - 1, 0.0, frozenset())

    # Deduplicate and sort
    seen = {}
    for p, e in results:
        if p not in seen or e < seen[p]:
            seen[p] = e
    items = sorted(seen.items(), key=lambda kv: kv[1])

    out: list[tuple[str, float, list[tuple[int, int]]]] = []
    for pset, e in items[:max_structures]:
        plist = sorted(pset)
        out.append((to_dot_bracket(plist, n), e, plist))
    return out


def _enum_V(seq, V, W, can, hp_fn, stack_fn, i, j, cap, remaining):
    """Enumerate decompositions of V[i][j] whose total cost ≤ ``cap``.

    ``cap`` is the absolute energy budget for V[i][j] itself.
    """
    if remaining <= 0:
        return
    if V[i][j] >= INF or V[i][j] > cap + 1e-6:
        return

    # 1. Hairpin
    hp_e = hp_fn(i, j)
    if hp_e <= cap + 1e-6:
        yield frozenset([(i, j)]), hp_e

    # 2. Stack (i+1, j-1)
    if can(seq, i + 1, j - 1) and V[i + 1][j - 1] < INF:
        stk = stack_fn(seq, i, j)
        inner_cap = cap - stk
        for inner_pairs, inner_e in _enum_V(seq, V, W, can, hp_fn, stack_fn,
                                            i + 1, j - 1, inner_cap, remaining):
            yield inner_pairs | {(i, j)}, stk + inner_e

    # 3. Bifurcation: V[i][j] = V[i][k] + W[k+1][j-1]
    for k in range(i + 1, j):
        if k + 1 > j - 1 or V[i][k] >= INF:
            continue
        if V[i][k] + W[k + 1][j - 1] > cap + 1e-6:
            continue
        v_cap = cap - W[k + 1][j - 1]
        for v_pairs, v_e in _enum_V(seq, V, W, can, hp_fn, stack_fn,
                                    i, k, v_cap, remaining):
            yield v_pairs | {(i, j)}, v_e + W[k + 1][j - 1]


def _fold_matrices(seq, celsius, material):
    """Re-derive the V/W matrices from the MFE DP (sans traceback table)."""
    from strider.structure.mfe import (
        _can_pair_fn, _stack_fn, _hairpin_energy, MIN_HAIRPIN_LOOP,
    )
    n = len(seq)
    V = np.full((n, n), INF)
    W = np.zeros((n, n))
    can = _can_pair_fn(material)
    stack_e = _stack_fn(material)

    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if can(seq, i, j) and j - i > MIN_HAIRPIN_LOOP:
                hp_e = _hairpin_energy(seq, i, j, celsius, material)
                V[i][j] = hp_e
                if can(seq, i + 1, j - 1) and j - i > MIN_HAIRPIN_LOOP + 1:
                    e_stack = stack_e(seq, i, j) + V[i + 1][j - 1]
                    if e_stack < V[i][j]:
                        V[i][j] = e_stack
                for k in range(i + 1, j):
                    e_bif = V[i][k] + W[k + 1][j - 1] if k + 1 <= j - 1 else V[i][k]
                    if e_bif < V[i][j]:
                        V[i][j] = e_bif
            W[i][j] = W[i][j - 1]
            for k in range(i, j + 1):
                if V[k][j] < INF:
                    left = W[i][k - 1] if k > i else 0.0
                    e = left + V[k][j]
                    if e < W[i][j]:
                        W[i][j] = e
    return V, W
