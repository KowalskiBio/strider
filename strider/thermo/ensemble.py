"""
Partition function and ensemble free energy for nucleic acid secondary structures.

Single-strand: McCaskill (1990) O(n^3) DP.
Multi-strand:  nick-aware McCaskill DP on the concatenated sequence.
               A nick at strand boundary position k prevents any hairpin loop
               from spanning that position — the same rule used by ViennaRNA
               (RNAcofold) and NUPACK.  Inter-strand base pairs form freely via
               the stack recursion; they cannot close intra-strand hairpins.

References:
    McCaskill (1990) Biopolymers 29:1105-1119
    Markham & Zuker (2008) DINAMelt — nick-aware cofold
    Dirks & Pierce (2003) NUPACK multi-strand partition function
"""

from __future__ import annotations
import math
import numpy as np

R = 1.987e-3        # kcal / (mol · K)
INF = float("inf")

# Approximate Turner multi-loop initiation penalty (kcal/mol)
_G_ML_INIT = 3.4


def _wc_pairs(material: str) -> set[frozenset[str]]:
    """Return the set of valid Watson-Crick (and wobble for RNA) base-pair frozensets."""
    if material == "rna":
        return {frozenset("AU"), frozenset("UA"), frozenset("GC"), frozenset("CG"),
                frozenset("GU"), frozenset("UG")}  # wobble
    return {frozenset("AT"), frozenset("TA"), frozenset("GC"), frozenset("CG")}


def ensemble_dg(
    sequence: str,
    celsius: float = 37.0,
    material: str = "dna",
) -> tuple[float, np.ndarray]:
    """
    Ensemble free energy and base-pair probability matrix.

    Returns:
        dG_ens (kcal/mol): ensemble free energy = -RT ln(Q)
        pair_probs (ndarray, shape (n, n)): prob that positions i-j are paired
    """
    seq = sequence.upper().replace("U", "T") if material == "dna" else sequence.upper().replace("T", "U")
    n = len(seq)
    T = celsius + 273.15
    beta = 1.0 / (R * T)
    pairs = _wc_pairs(material)

    # Q[i][j] = partition function for subsequence [i..j]
    Q = np.zeros((n, n))
    Qb = np.zeros((n, n))  # restricted: i and j form a base pair

    # Base case: single nucleotide
    for i in range(n):
        Q[i][i] = 1.0

    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    _fill_dp(seq, Q, Qb, n, T, pairs, material)

    Z = Q[0][n - 1]
    if Z <= 0:
        Z = 1.0
    dG_ens = -R * T * math.log(Z)

    pair_probs = _pair_probs(Q, Qb, n, Z)
    return dG_ens, pair_probs


def _hairpin_loop_energy(seq: str, i: int, j: int, material: str, T: float) -> float:
    """Simplified hairpin loop free energy (Turner-like)."""
    loop_size = j - i - 1
    if loop_size < 3:
        return INF   # too small
    # Simplified: log(loop_size) penalty + terminal mismatch (ignored here)
    if material == "rna":
        # Turner 2004 approximate: ΔG(loop) ≈ 5.4 + 1.75·RT·ln(n/3) for n ≥ 4
        dG = 5.4 + 1.75 * R * T * math.log(loop_size / 3.0) if loop_size >= 4 else 5.4
    else:
        dG = 4.0 + 1.75 * R * T * math.log(max(loop_size, 3) / 3.0)
    return dG


def _stack_energy(seq: str, i: int, j: int, material: str) -> float:
    """Stacking energy for closing pair (i,j) over interior pair (i+1,j-1)."""
    if material == "dna":
        from strider.thermo.nn_dna import DNA_NN
        dinuc = seq[i] + seq[i + 1]
        if dinuc in DNA_NN:
            return DNA_NN[dinuc][2]
        rc = seq[j] + seq[j - 1]  # bottom strand
        if rc in DNA_NN:
            return DNA_NN[rc][2]
        return -1.5
    else:
        from strider.thermo.nn_rna import RNA_NN
        dinuc = seq[i] + seq[i + 1]
        if dinuc in RNA_NN:
            return RNA_NN[dinuc][2]
        return -2.0


def _boltzmann(dG: float, T: float) -> float:
    """Return the Boltzmann factor exp(-dG / RT), zero if dG is infinite."""
    if dG == INF:
        return 0.0
    R_val = 1.987e-3
    return math.exp(-dG / (R_val * T))


def _can_pair(seq: str, i: int, j: int, pairs: set) -> bool:
    """Return True if positions i and j form a valid Watson-Crick pair with minimum hairpin size."""
    return frozenset([seq[i], seq[j]]) in pairs and (j - i) > 3


def _can_pair_nicks(seq: str, i: int, j: int, pairs: set, nicks: list) -> bool:
    """
    Base-pair check for nick-aware DP.

    Inter-strand pairs (one end in each strand) have no minimum-distance
    requirement — two strands can pair right at the nick (j - i == 1).
    Intra-strand pairs still require the minimum hairpin loop size.
    """
    if j <= i:
        return False
    if frozenset([seq[i], seq[j]]) not in pairs:
        return False
    # Inter-strand: nick falls strictly between the two ends OR at j
    if any(i < k <= j for k in nicks):
        return True
    return (j - i) > 3  # intra-strand minimum


# SantaLucia & Hicks 2004 duplex initiation energies (kcal/mol)
_INIT_GC = 0.98   # G-C or C-G terminal pair
_INIT_AT = 1.03   # A-T or T-A terminal pair


def _duplex_init(seq: str, i: int, j: int) -> float:
    """Return the SantaLucia initiation penalty (kcal/mol) for the terminal pair (i, j)."""
    bp = frozenset([seq[i], seq[j]])
    return _INIT_GC if bp in (frozenset("GC"), frozenset("CG")) else _INIT_AT


def _fill_dp(seq, Q, Qb, n, T, pairs, material):
    """Run the single-strand McCaskill DP (no nicks) by delegating to _fill_dp_nicks."""
    _fill_dp_nicks(seq, Q, Qb, n, T, pairs, material, nicks=[])


def _fill_dp_nicks(seq, Q, Qb, n, T, pairs, material, nicks: list):
    """
    McCaskill DP with nick-aware hairpin suppression (bottom-up, O(n^3)).

    nicks: list of positions k (first nt of each strand after the first).
           Hairpin loops are disallowed when any nick k satisfies i < k < j.
           Inter-strand pairs (spanning a nick) have no minimum-distance
           constraint and use duplex initiation energy for the terminal pair.
    """
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # Qb[i][j]: partition function with (i,j) forced paired.
            # All Qb and Q entries with shorter span are already filled.
            if _can_pair_nicks(seq, i, j, pairs, nicks):
                Qb[i][j] = _qb_val_nicks(seq, i, j, Q, Qb, T, pairs, material, nicks)

            # Q[i][j]: partition function for external loop over [i..j]
            Q[i][j] = Q[i][j - 1]  # j unpaired
            for k in range(i, j - 1):  # k can be j-1 for inter-strand terminal pairs
                if Qb[k][j] > 0:
                    left = Q[i][k - 1] if k > i else 1.0
                    Q[i][j] += left * Qb[k][j]


def _qb_val_nicks(seq, i, j, Q, Qb, T, pairs, material, nicks: list) -> float:
    """
    Partition function with (i,j) forced paired, respecting nick positions.

    Contributions:
      hairpin    — if no nick strictly inside [i+1..j-1] and loop ≥ 3 nt
      initiation — for terminal inter-strand pairs: the SantaLucia initiation
                   free energy (applied once at the innermost duplex pair)
      stack      — (i+1,j-1) also paired; propagates both intra- and inter-strand
      multiloop  — (i,j) closes a multi-loop with an interior stem and free region
    """
    val = 0.0
    spans_nick = any(i < k < j for k in nicks)
    is_inter = any(i < k <= j for k in nicks)

    # ── Hairpin (intra-strand only) ──────────────────────────────────────────
    if not spans_nick:
        dG_hp = _hairpin_loop_energy(seq, i, j, material, T)
        val += _boltzmann(dG_hp, T)

    # ── Duplex initiation (inter-strand terminal pair) ───────────────────────
    # Applies when (i,j) is an inter-strand pair AND there is no valid inner
    # pair to stack on.  This represents the "end" of a duplex — the SantaLucia
    # model charges an initiation penalty once per duplex at the terminal pair.
    if is_inter and not _can_pair_nicks(seq, i + 1, j - 1, pairs, nicks):
        val += _boltzmann(_duplex_init(seq, i, j), T)

    # ── Stack: (i+1, j-1) also paired (shorter length → already filled) ──────
    if _can_pair_nicks(seq, i + 1, j - 1, pairs, nicks):
        qb_inner = Qb[i + 1][j - 1]
        if qb_inner > 0:
            val += _boltzmann(_stack_energy(seq, i, j, material), T) * qb_inner

    # ── Multi-loop: interior stem [i+1..m] + free region [m+1..j-1] ─────────
    # Both Qb[i+1][m] and Q[m+1][j-1] have shorter span → already filled.
    for m in range(i + 2, j - 1):
        qb_left = Qb[i + 1][m]
        if qb_left > 0:
            q_right = Q[m + 1][j - 1] if m + 1 <= j - 1 else 1.0
            val += qb_left * q_right * _boltzmann(_G_ML_INIT, T)

    return val


def _pair_probs(Q, Qb, n, Z):
    """Compute the n×n base-pair probability matrix from the forward DP tables."""
    probs = np.zeros((n, n))
    if Z == 0:
        return probs
    for i in range(n):
        for j in range(i + 4, n):
            if Qb[i][j] > 0:
                left = Q[0][i - 1] if i > 0 else 1.0
                right = Q[j + 1][n - 1] if j < n - 1 else 1.0
                probs[i][j] = (left * Qb[i][j] * right) / Z
                probs[j][i] = probs[i][j]
    return probs


def bimolecular_dg(
    seq1: str,
    seq2: str,
    celsius: float = 37.0,
    material: str = "dna",
) -> float:
    """
    Standard-state binding ΔG° for a two-strand complex (kcal/mol).

    Uses the nick-aware McCaskill DP:
      ΔG°_bind = ΔG(complex) − ΔG(seq1) − ΔG(seq2)

    The complex ΔG is computed on the concatenated sequence S = seq1 + seq2
    with a nick at position len(seq1).  Hairpin loops spanning the nick are
    disallowed; inter-strand stacks build up correctly through the stack
    recursion.  This is the same algorithm used by ViennaRNA (RNAcofold) and
    NUPACK for two-strand complexes.
    """
    dG1, _ = ensemble_dg(seq1, celsius, material)
    dG2, _ = ensemble_dg(seq2, celsius, material)
    dG_complex = _multistrand_dg([seq1, seq2], celsius, material)
    return dG_complex - dG1 - dG2


def _multistrand_dg(
    sequences: list[str],
    celsius: float,
    material: str,
) -> float:
    """
    Ensemble ΔG for the concatenated multi-strand complex (kcal/mol).

    Runs nick-aware McCaskill DP with nicks at each strand boundary.
    Does NOT subtract individual strand energies — use bimolecular_dg or
    engine.ddg() for the binding free energy.
    """
    seq = "".join(sequences)
    n = len(seq)
    if n == 0:
        return 0.0
    T = celsius + 273.15
    pairs = _wc_pairs(material)

    # Nick positions: first nt of each strand after the first
    nicks: list[int] = []
    pos = 0
    for s in sequences[:-1]:
        pos += len(s)
        nicks.append(pos)

    Q = np.zeros((n, n))
    Qb = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = 1.0
    for i in range(n - 1):
        Q[i][i + 1] = 1.0

    _fill_dp_nicks(seq, Q, Qb, n, T, pairs, material, nicks)

    Z = Q[0][n - 1]
    if Z <= 0:
        Z = 1.0
    return -R * T * math.log(Z)
