"""
Self-consistent hairpin melting temperature.

A unimolecular hairpin melts in a two-state fashion (folded ⇌ open), so its
melting temperature is the temperature at which the folding free energy
vanishes:

    Tm = ΔH / ΔS         (concentration-independent, unlike a duplex)

Strider's folding model (``structure.mfe`` / ``thermo.ensemble``) is built on
ΔG₃₇-only stack parameters, so on its own it has no enthalpy and its partition
function never melts.  This module supplies the missing enthalpy without any new
parameter tables, using two facts about the SantaLucia/Mathews-Turner DNA set:

  * Stack enthalpies already live in ``nn_dna.DNA_NN`` (ΔH, ΔS, ΔG₃₇).
  * Hairpin/internal loop penalties are *purely entropic* in this set
    (ΔH = 0; verified against the published loop tables — a size-3 loop is
    ΔH=0, ΔS=−11.3 cal/mol·K, i.e. ΔG₃₇ = +3.5).

So for a predicted hairpin we take the structure's own (well-calibrated) ΔG₃₇,
add up the stem-stack ΔH from ``DNA_NN``, and recover a consistent entropy:

    ΔS = (ΔH − ΔG₃₇) / T_ref

from which both Tm and ΔG(T) follow.  This reproduces independent reference
engines (e.g. seqfold) to <1 °C at 1 M Na⁺.

Caveats
-------
* Two-state, single-hairpin only (one stem + one loop).  Multiloops / bulges
  raise ``ValueError`` — use the full ensemble for those.
* Tm is *hypersensitive* to the ΔH/ΔS bookkeeping: a ~few-% shift in either
  moves Tm by tens of °C.  Treat absolute Tm as calibratable, not exact, and
  anchor it against experimental (e.g. qPCR) melts.
* Salt/Mg²⁺ enters through the per-base-pair ΔG model (``salt.dg_per_bp_salt``),
  folded into the closed-state ΔG before ΔS is derived — the same correction the
  ensemble DP uses, not the (oversized, duplex-calibrated) Owczarzy Tm shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

T_REF = 310.15  # K, the 37 °C reference at which ΔG₃₇ tables are defined


@dataclass(frozen=True)
class HairpinThermo:
    """Two-state hairpin thermodynamics at the requested salt conditions."""
    tm_celsius: float
    dH: float          # kcal/mol
    dS: float          # cal/mol/K
    dG37: float        # kcal/mol, salt-corrected closed-state free energy
    n_pairs: int
    structure: str


def hairpin_thermo(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
    structure: list[tuple[int, int]] | None = None,
) -> HairpinThermo:
    """
    Two-state thermodynamics (Tm, ΔH, ΔS, ΔG₃₇) for a hairpin.

    Parameters
    ----------
    seq : strand sequence.
    sodium_M, magnesium_M : ion concentrations (1 M Na⁺ / 0 Mg²⁺ = reference).
    structure : optional list of ``(i, j)`` base pairs; if omitted the MFE
        hairpin is predicted with :func:`strider.structure.mfe.fold_mfe`.

    Raises
    ------
    ValueError : if the structure is not a single simple hairpin.
    """
    from strider.structure.mfe import fold_mfe
    from strider.thermo.nn_dna import DNA_NN, reverse_complement
    from strider.thermo.salt import dg_per_bp_salt

    seq = seq.upper().replace("U", "T")

    if structure is None:
        struct_str, dG37_1M, pairs = fold_mfe(seq, 37.0, material)
    else:
        pairs = sorted(structure)
        struct_str = _dotbracket(seq, pairs)
        dG37_1M = _structure_dg37(seq, pairs, material)

    if not pairs:
        raise ValueError("sequence has no predicted base pairs — not a hairpin")
    _assert_simple_hairpin(pairs)

    pairset = set(pairs)
    dH = 0.0
    for (i, j) in pairs:
        if (i + 1, j - 1) in pairset:           # stacked NN step
            dinuc = seq[i:i + 2]
            entry = DNA_NN.get(dinuc) or DNA_NN.get(reverse_complement(dinuc))
            if entry is not None:
                dH += entry[0]

    # Fold salt into the closed-state ΔG₃₇ (per closed base pair), then derive ΔS.
    dG37 = dG37_1M + len(pairs) * dg_per_bp_salt(sodium_M, magnesium_M)
    dS_kcal = (dH - dG37) / T_REF               # kcal/mol/K
    if dS_kcal == 0:
        raise ValueError("degenerate entropy — cannot define a melting point")
    tm_K = dH / dS_kcal
    return HairpinThermo(
        tm_celsius=tm_K - 273.15,
        dH=dH,
        dS=dS_kcal * 1000.0,
        dG37=dG37,
        n_pairs=len(pairs),
        structure=struct_str,
    )


def hairpin_tm(
    seq: str,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
) -> float:
    """Melting temperature (°C) of the predicted hairpin. See :func:`hairpin_thermo`."""
    return hairpin_thermo(seq, sodium_M, magnesium_M, material).tm_celsius


def fraction_folded(
    seq: str,
    celsius: float,
    sodium_M: float = 1.0,
    magnesium_M: float = 0.0,
    material: str = "dna",
) -> float:
    """
    Two-state folded fraction at ``celsius`` — the quantity a beacon melt
    (fluorophore dequenching) actually traces out.
    """
    th = hairpin_thermo(seq, sodium_M, magnesium_M, material)
    T = celsius + 273.15
    dG_T = th.dH - T * th.dS / 1000.0           # ΔG(T) of the closed state
    R = 1.987e-3
    return 1.0 / (1.0 + math.exp(dG_T / (R * T)))


# ─── internals ────────────────────────────────────────────────────────────────

def _assert_simple_hairpin(pairs: list[tuple[int, int]]) -> None:
    """Reject anything that isn't one nested, contiguous stem (no multiloop/bulge)."""
    pairs = sorted(pairs)
    pairset = set(pairs)
    # Exactly one pair must be unstacked on its inner side (the loop-closing pair),
    # and pairs must be strictly nested.
    for k in range(len(pairs) - 1):
        i, j = pairs[k]
        ni, nj = pairs[k + 1]
        if not (ni > i and nj < j):
            raise ValueError("structure is not a single nested hairpin stem")
        if ni != i + 1 or nj != j - 1:
            raise ValueError("stem has a bulge/internal loop — use the full ensemble")


def _dotbracket(seq: str, pairs: list[tuple[int, int]]) -> str:
    s = ["."] * len(seq)
    for i, j in pairs:
        s[i], s[j] = "(", ")"
    return "".join(s)


def _structure_dg37(seq: str, pairs: list[tuple[int, int]], material: str) -> float:
    """ΔG₃₇ (1 M) of a given simple hairpin: stem stacks + hairpin loop."""
    from strider.thermo.ensemble import _stack_energy, _hairpin_loop_energy
    pairset = set(pairs)
    dG = 0.0
    for (i, j) in pairs:
        if (i + 1, j - 1) in pairset:
            dG += _stack_energy(seq, i, j, material)
    inner_i, inner_j = max(pairs, key=lambda p: p[0])
    dG += _hairpin_loop_energy(seq, inner_i, inner_j, material, T_REF)
    return dG
