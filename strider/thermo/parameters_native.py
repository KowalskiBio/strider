"""
Built-in MIT-licensed parameter set.

Assembled from primary-literature constants — SantaLucia 2004 (DNA) and
Mathews 1999 / Turner 2004 (RNA) — covering the minimum set of tables needed
by strider's native MFE / pfunc engines: stack, hairpin loop sizes, bulge
loop sizes, interior loop sizes, terminal penalty, and multiloop
coefficients.  Higher-order tables (hairpin/interior mismatch, 1×1/1×2/2×2
interior, dangle, coaxial) are absent — code that requires them should branch
on :meth:`ParameterSet.has` and either fall back or ask the user to supply a
JSON parameter file via ``STRIDER_PARAMS_DIR``.
"""

from __future__ import annotations

import numpy as np

from strider.thermo.parameters import ParameterSet


# ─── stack tables ─────────────────────────────────────────────────────────────

# Build the Turner-format stack dict from the dinucleotide NN ΔG37 table.
# Stack key is top5+top3+bot5+bot3, where the bottom strand is read 5'→3'.
# For a Watson–Crick stack, bot5+bot3 = reverse-complement(top5+top3).

_COMPL_DNA = str.maketrans("ACGT", "TGCA")
_COMPL_RNA = str.maketrans("ACGU", "UGCA")


def _stack_key(top: str, complement_map) -> str:
    """Return the 4-char Turner-format stack key for the Watson-Crick stack on top dinuc ``top``."""
    bot = top.translate(complement_map)[::-1]
    return top + bot


def _stack_dict_dna() -> dict[str, float]:
    """16-entry DNA stack table from SantaLucia 2004 NN parameters (ΔG37 kcal/mol)."""
    from strider.thermo.nn_dna import DNA_NN
    out: dict[str, float] = {}
    for dinuc, (_h, _s, g) in DNA_NN.items():
        key = _stack_key(dinuc, _COMPL_DNA)
        out[key] = float(g)
    return out


def _stack_dict_rna() -> dict[str, float]:
    """16-entry RNA stack table from Mathews 1999 NN parameters (ΔG37 kcal/mol)."""
    from strider.thermo.nn_rna import RNA_NN
    out: dict[str, float] = {}
    for dinuc, (_h, _s, g) in RNA_NN.items():
        # Keys use T (not U) so DNA and RNA stack tables share the same alphabet.
        key = _stack_key(dinuc, _COMPL_RNA).replace("U", "T")
        out[key] = float(g)
    return out


def _stack_dh_dna() -> dict[str, float]:
    """16-entry DNA stack ΔH (kcal/mol) for temperature extrapolation."""
    from strider.thermo.nn_dna import DNA_NN
    return {_stack_key(d, _COMPL_DNA): float(h) for d, (h, _s, _g) in DNA_NN.items()}


def _stack_dh_rna() -> dict[str, float]:
    """16-entry RNA stack ΔH (kcal/mol) for temperature extrapolation."""
    from strider.thermo.nn_rna import RNA_NN
    return {_stack_key(d, _COMPL_RNA).replace("U", "T"): float(h)
            for d, (h, _s, _g) in RNA_NN.items()}


# ─── loop size tables ─────────────────────────────────────────────────────────

# Source: Turner 2004 / Mathews 1999 loop initiation tables.  Index = unpaired
# nucleotides in the loop; entries before MIN_HAIRPIN_LOOP=3 are sentinels.

_INF = 30.0  # large positive number used in place of float('inf')

# DNA hairpin loop initiation (SantaLucia / Mathews compromise; same numbers
# already inlined in strider.structure.mfe._hairpin_energy).
_HAIRPIN_DNA = np.array(
    [_INF, _INF, _INF, 4.1, 4.3, 4.9, 4.4, 4.3, 4.1, 4.0,
     4.7, 4.9, 5.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
     5.6, 5.7, 5.7, 5.8, 5.8, 5.9, 5.9, 6.0, 6.0, 6.1]
)

_HAIRPIN_RNA = np.array(
    [_INF, _INF, _INF, 5.4, 5.6, 5.7, 5.4, 6.0, 5.5, 6.4,
     6.5, 6.6, 6.7, 6.8, 6.9, 6.9, 7.0, 7.1, 7.1, 7.2,
     7.2, 7.3, 7.3, 7.4, 7.4, 7.5, 7.5, 7.6, 7.6, 7.7]
)

# Bulge loop initiation (Turner 2004 short-bulge table extended logarithmically).
_BULGE = np.array(
    [_INF, 3.8, 2.8, 3.2, 3.6, 4.0, 4.4, 4.6, 4.7, 4.8,
     4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.4, 5.5, 5.5, 5.6,
     5.7, 5.7, 5.8, 5.8, 5.8, 5.9, 5.9, 6.0, 6.0, 6.0]
)

# Interior loop initiation (Turner 2004).  Index 1 is unused (no length-1 interior).
_INTERIOR = np.array(
    [_INF, _INF, 4.1, 5.1, 4.9, 5.3, 5.7, 5.9, 6.1, 6.3,
     6.5, 6.6, 6.8, 6.9, 7.0, 7.1, 7.2, 7.2, 7.3, 7.4,
     7.5, 7.5, 7.6, 7.6, 7.7, 7.7, 7.8, 7.8, 7.9, 7.9]
)

# Asymmetric loop penalty (Ninio, 1979); five-term lookup matching the Turner schema.
_ASYM_NINIO = np.array([0.4, 0.3, 0.2, 0.1, 3.0])


# ─── builder ──────────────────────────────────────────────────────────────────

def build_native_paramset(material: str = "DNA") -> ParameterSet:
    """
    Construct a :class:`ParameterSet` from strider's built-in NN tables.

    The result has only the subset of tables needed by the existing
    duplex / hairpin / bulge / interior energetics — mismatch, dangle, and
    coaxial tables are absent.  Code that needs them should branch on
    ``params.has("dangle_5")`` etc.

    Parameters
    ----------
    material : ``"DNA"`` or ``"RNA"``
    """
    material = material.upper()
    if material == "DNA":
        stack = _stack_dict_dna()
        stack_dH = _stack_dh_dna()
        hairpin = _HAIRPIN_DNA
        wobble = False
        terminal_penalty = {"AT": 0.45, "TA": 0.45}    # SantaLucia INIT_AT penalty
    elif material == "RNA":
        stack = _stack_dict_rna()
        stack_dH = _stack_dh_rna()
        hairpin = _HAIRPIN_RNA
        wobble = True
        terminal_penalty = {"AT": 0.45, "TA": 0.45, "GT": 0.45}  # AU/GU penalty
    else:
        raise ValueError(f"material must be 'DNA' or 'RNA', got {material!r}")

    dG = {
        "stack": stack,
        "hairpin_size": hairpin,
        "bulge_size": _BULGE,
        "interior_size": _INTERIOR,
        "asymmetry_ninio": _ASYM_NINIO,
        "terminal_penalty": terminal_penalty,
        # Multiloop coefficients: Turner 2004 / SantaLucia defaults.
        "multiloop_init": 3.4,
        "multiloop_pair": 0.4,
        "multiloop_base": 0.0,
        "join_penalty": 1.96,
        "log_loop_penalty": 1.07,
    }
    dH = {
        "stack": stack_dH,
        # Loop-size ΔH copied from ΔG as a coarse approximation.  Strider's
        # native engine currently does not extrapolate loop energies with T.
        "hairpin_size": hairpin,
        "bulge_size": _BULGE,
        "interior_size": _INTERIOR,
    }

    return ParameterSet(
        name=f"native-{material.lower()}",
        material=material,
        default_wobble_pairing=wobble,
        dG=dG,
        dH=dH,
        source_path=None,
        comment="strider built-in: SantaLucia 2004 (DNA) / Mathews 1999 (RNA)",
    )
