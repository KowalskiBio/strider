"""
Modified nucleotide thermodynamic corrections.

Sources:
  LNA: Owczarzy et al. (2011) Nucleic Acids Res. 39:880-889
       Kierzek & Kierzek (2003) Nucleic Acids Res. 31:4472-4480
  2'OMe: Freier & Altmann (1997) Nucleic Acids Res. 25:4429-4443
  Phosphorothioate (PS): Kibler-Herzog et al. (1994) Nucleic Acids Res. 22:2140-2148

All values are average ΔΔG37 corrections per modified position (kcal/mol).
Positive values DESTABILIZE, negative STABILIZE relative to unmodified.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


ModType = Literal["LNA", "2OMe", "PS"]


# Average ΔΔG37 per modification (kcal/mol, negative = more stable)
_DDG_PER_MOD: dict[ModType, float] = {
    "LNA":  -1.5,   # range -0.5 to -2.5 depending on sequence context
    "2OMe": -0.3,   # mild stabilization
    "PS":   +0.4,   # slight destabilization
}

# Correction factors as a function of fraction modified
_FRACTION_SCALE: dict[ModType, float] = {
    "LNA":  1.0,
    "2OMe": 0.8,
    "PS":   1.0,
}


@dataclass
class ModificationSite:
    """
    A single chemical modification at a specific nucleotide position.

    Attributes
    ----------
    position : 0-based index in the sequence string
    mod_type : 'LNA', '2OMe', or 'PS'
    """
    position: int           # 0-indexed position in sequence
    mod_type: ModType


def apply_modifications(
    dg_unmodified: float,
    sequence: str,
    modifications: list[ModificationSite],
) -> float:
    """
    Apply modification corrections to a baseline duplex ΔG.

    Returns adjusted ΔG (kcal/mol).
    """
    correction = 0.0
    for mod in modifications:
        base_ddg = _DDG_PER_MOD[mod.mod_type]
        scale = _FRACTION_SCALE[mod.mod_type]
        correction += base_ddg * scale
    return dg_unmodified + correction


def lna_ddg_per_position(
    sequence: str,
    position: int,
    context_window: int = 2,
) -> float:
    """
    Sequence-context-aware LNA ΔΔG at a specific position.

    GC-rich context gives stronger stabilization (up to -2.5 kcal/mol);
    AT-rich context gives moderate stabilization (-0.8 to -1.2 kcal/mol).
    Based on Owczarzy 2011 Table 1.
    """
    seq = sequence.upper()
    n = len(seq)
    start = max(0, position - context_window)
    end = min(n, position + context_window + 1)
    window = seq[start:end]
    gc_frac = sum(1 for b in window if b in "GC") / len(window) if window else 0.5
    return -0.8 - 1.7 * gc_frac  # interpolate -0.8 (pure AT) to -2.5 (pure GC)


def modification_string(sequence: str, modifications: list[ModificationSite]) -> str:
    """
    Return a human-readable annotated sequence string.
    E.g. 'AT+CGTA' where + marks LNA positions.
    """
    symbols = {"LNA": "+", "2OMe": "*", "PS": "~"}
    seq = list(sequence.upper())
    for mod in sorted(modifications, key=lambda m: m.position):
        i = mod.position
        if 0 <= i < len(seq):
            seq[i] = symbols.get(mod.mod_type, "?") + seq[i]
    return "".join(seq)
