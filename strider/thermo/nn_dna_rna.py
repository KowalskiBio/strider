"""
DNA:RNA hybrid nearest-neighbor parameters.

Source: Sugimoto et al. (1995) Biochemistry 34:11211-11216

Used for RNA target (e.g. miRNA) hybridizing to a DNA probe.
DNA strand is top (5'→3'), RNA strand is bottom complement.
Units: ΔH in kcal/mol, ΔS in cal/mol/K, ΔG37 in kcal/mol.
"""

from __future__ import annotations

# Keys: 5'→3' DNA dinucleotide on top strand
DNA_RNA_NN: dict[str, tuple[float, float, float]] = {
    "AA": (-7.8,  -21.9, -1.0),
    "AC": (-5.9,  -12.3, -2.1),
    "AG": (-9.1,  -23.5, -1.8),
    "AT": (-8.3,  -23.9, -0.9),
    "CA": (-9.0,  -26.1, -0.9),
    "CC": (-9.3,  -23.2, -2.1),
    "CG": (-16.3, -47.1, -1.7),
    "CT": (-7.0,  -19.7, -0.9),
    "GA": (-5.5,  -13.5, -1.3),
    "GC": (-8.0,  -17.1, -2.7),
    "GG": (-12.8, -31.9, -2.9),
    "GT": (-7.8,  -21.6, -1.1),
    "TA": (-7.8,  -23.2, -0.6),
    "TC": (-8.6,  -22.9, -1.5),
    "TG": (-10.5, -28.4, -1.6),
    "TT": (-11.5, -36.4, -0.2),
}

INIT_DNA_RNA = (1.9,  -3.9,  3.1)   # Sugimoto 1995 initiation
R = 1.987e-3


def hybrid_duplex_dg(
    dna_seq: str,
    celsius: float = 37.0,
    sodium_M: float = 1.0,
) -> float:
    """
    ΔG (kcal/mol) for DNA:RNA hybrid duplex formation.

    dna_seq: DNA strand (5'→3'). RNA complement is inferred.
    """
    seq = dna_seq.upper().replace("U", "T")
    T = celsius + 273.15
    dH = dS = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        if dinuc in DNA_RNA_NN:
            h, s, _ = DNA_RNA_NN[dinuc]
        else:
            h, s = -9.0, -25.0
        dH += h
        dS += s
    dH += INIT_DNA_RNA[0] * 2
    dS += INIT_DNA_RNA[1] * 2
    return dH - T * (dS / 1000.0)
