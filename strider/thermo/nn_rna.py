"""
RNA nearest-neighbor thermodynamic parameters.

Source: Mathews et al. (1999) J. Mol. Biol. 288:911-940
        Turner & Mathews (2010) Nucleic Acids Res. 38:D280-D282

Conditions: 1 M NaCl, pH 7, 37°C.
Units: ΔH in kcal/mol, ΔS in cal/mol/K, ΔG37 in kcal/mol.

Keys are 5'→3' dinucleotide on the top strand (RNA, U instead of T).
"""

from __future__ import annotations

RNA_NN: dict[str, tuple[float, float, float]] = {
    "AA": (-6.82,  -19.0, -0.93),
    "UU": (-6.82,  -19.0, -0.93),
    "AU": (-9.38,  -26.7, -1.10),
    "UA": (-7.69,  -20.5, -1.33),
    "CA": (-10.44, -26.9, -2.11),
    "UG": (-10.44, -26.9, -2.11),
    "GU": (-11.40, -29.5, -2.24),
    "AC": (-11.40, -29.5, -2.24),
    "CU": (-10.48, -27.1, -2.08),
    "AG": (-10.48, -27.1, -2.08),
    "GA": (-12.44, -32.5, -2.35),
    "UC": (-12.44, -32.5, -2.35),
    "CG": (-10.64, -26.7, -2.36),
    "GC": (-14.88, -36.9, -3.42),
    "GG": (-13.39, -32.7, -3.26),
    "CC": (-13.39, -32.7, -3.26),
}

INIT_GC_RNA = (0.0,  -10.8, 3.61)
INIT_AU_RNA = (0.0,  -10.8, 3.61)   # same by Mathews 1999

COMPLEMENT_RNA = str.maketrans("ACGU", "UGCA")
COMPLEMENT_DNA = str.maketrans("ACGT", "TGCA")
R = 1.987e-3


def reverse_complement_rna(seq: str) -> str:
    """Return the reverse complement of an RNA sequence (5'→3'), converting T→U first."""
    seq = seq.upper().replace("T", "U")
    return seq.translate(COMPLEMENT_RNA)[::-1]


def duplex_dg_rna(
    seq: str,
    celsius: float = 37.0,
    sodium_M: float = 1.0,
) -> float:
    """ΔG (kcal/mol) for RNA duplex formation."""
    seq = seq.upper().replace("T", "U")
    T = celsius + 273.15
    dH, dS = _sum_nn_rna(seq)
    dH_i, dS_i = INIT_GC_RNA[0], INIT_GC_RNA[1]
    dH += dH_i * 2
    dS += dS_i * 2
    return dH - T * (dS / 1000.0)


def _sum_nn_rna(seq: str) -> tuple[float, float]:
    """Sum nearest-neighbor ΔH and ΔS for an RNA sequence."""
    dH = dS = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        if dinuc in RNA_NN:
            h, s, _ = RNA_NN[dinuc]
        else:
            h, s = -10.0, -26.0
        dH += h
        dS += s
    return dH, dS
