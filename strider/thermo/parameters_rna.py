"""
RNA thermodynamic nearest-neighbor parameters at 37 °C, 1 M NaCl.

Primary sources:
    Mathews D.H., Sabina J., Zuker M., Turner D.H. (1999). Expanded sequence
        dependence of thermodynamic parameters improves prediction of RNA
        secondary structure. J. Mol. Biol. 288: 911-940.
    Mathews D.H., Disney M.D., Childs J.L., Schroeder S.J., Zuker M.,
        Turner D.H. (2004). Incorporating chemical modification constraints
        into a dynamic programming algorithm for prediction of RNA secondary
        structure. PNAS 101: 7287-7292.
    Turner D.H. & Mathews D.H. (2010). NNDB: the nearest neighbor parameter
        database for predicting stability of nucleic acid secondary structure.
        Nucleic Acids Res. 38: D280-D282.

Additional measurements consolidated in the above:
    Xia T. et al. (1998). Biochemistry 37: 14719-14735 (Watson-Crick stacks).
    Schroeder S.J. & Turner D.H. (2000). Biochemistry 39: 9257-9274
        (terminal-mismatch and dangle measurements).
    Walter A.E. et al. (1994). PNAS 91: 9218-9222 (coaxial stacking).

All energies in kcal/mol at 37 °C, 1 M NaCl, neutral pH.
"""

# ── Loop size penalties ────────────────────────────────────────────────────────
HAIRPIN_SIZE = [
    0.0, 0.0, 5.4, 5.6, 5.7, 5.4, 6.0, 5.5, 6.4, 6.5,
    6.6, 6.7, 6.8, 6.9, 6.9, 7.0, 7.1, 7.1, 7.2, 7.2,
    7.3, 7.3, 7.4, 7.4, 7.5, 7.5, 7.5, 7.6, 7.6, 7.7,
]

BULGE_SIZE = [
    3.8, 2.8, 3.2, 3.6, 4.0, 4.4, 4.6, 4.7, 4.8, 4.9,
    5.0, 5.1, 5.2, 5.3, 5.4, 5.4, 5.5, 5.5, 5.6, 5.7,
    5.7, 5.8, 5.8, 5.8, 5.9, 5.9, 6.0, 6.0, 6.0, 6.1,
]

INTERIOR_SIZE = [
    0.0, 0.0, 0.0, 1.1, 2.0, 2.0, 2.1, 2.3, 2.4, 2.5,
    2.6, 2.7, 2.8, 2.9, 2.9, 3.0, 3.1, 3.1, 3.2, 3.3,
    3.3, 3.4, 3.4, 3.5, 3.5, 3.5, 3.6, 3.6, 3.7, 3.7,
]

LOG_LOOP_PENALTY = 1.0703471894499998

# ── Multiloop parameters ───────────────────────────────────────────────────────
ML_BASE = 0.1711
ML_INIT = 8.1142
ML_PAIR = 0.8361

# ── Bimolecular association penalty ───────────────────────────────────────────
JOIN_PENALTY = 4.09

# ── Ninio asymmetry correction ─────────────────────────────────────────────────
ASYMMETRY_NINIO = [0.6, 0.6, 0.6, 0.6, 9999.0]

# ── Terminal AU/GU penalties ──────────────────────────────────────────────────
TERMINAL_PENALTY: dict[str, float] = {
    "AT": 0.45, "GT": 0.45, "TA": 0.45, "TG": 0.45,
}

# ── Nearest-neighbour stack energies ──────────────────────────────────────────
STACK: dict[str, float] = {
    # WC stacks (16)
    "AATT": -0.9,
    "ACGT": -2.2,
    "AGCT": -2.1,
    "ATAT": -1.1,
    "CATG": -2.1,
    "CCGG": -3.3,
    "CGCG": -2.4,
    "CTAG": -2.1,
    "GATC": -2.4,
    "GCGC": -3.4,
    "GGCC": -3.3,
    "GTAC": -2.2,
    "TATA": -1.3,
    "TCGA": -2.4,
    "TGCA": -2.1,
    "TTAA": -0.9,
    # Mismatch stacks (20)
    "AGTT": -0.6,
    "ATGT": -1.4,
    "CGTG": -1.4,
    "CTGG": -2.1,
    "GATT": -1.3,
    "GCGT": -2.5,
    "GGCT": -2.1,
    "GGTC": -1.5,
    "GGTT": -0.5,
    "GTAT": -1.4,
    "GTGC": -2.5,
    "GTGT":  1.3,
    "TATG": -1.0,
    "TCGG": -1.5,
    "TGCG": -1.4,
    "TGTA": -1.0,
    "TGTG":  0.3,
    "TTAG": -0.6,
    "TTGA": -1.3,
    "TTGG": -0.5,
}

# ── Terminal mismatch energies ─────────────────────────────────────────────────
TERMINAL_MISMATCH: dict[str, float] = {
    "AATA": -1.0,  "AATC": -0.7,  "AATG": -1.1,  "AATT": -0.7,
    "ACGA": -1.1,  "ACGC": -1.1,  "ACGG": -1.6,  "ACGT": -1.1,
    "AGCA": -1.5,  "AGCC": -1.0,  "AGCG": -1.4,  "AGCT": -1.0,
    "AGTA": -1.0,  "AGTC": -0.7,  "AGTG": -0.5,  "AGTT": -0.7,
    "ATAA": -0.8,  "ATAC": -0.6,  "ATAG": -0.8,  "ATAT": -0.6,
    "ATGA": -0.3,  "ATGC": -0.6,  "ATGG": -0.6,  "ATGT": -0.6,
    "CATA": -0.8,  "CATC": -0.6,  "CATG": -0.8,  "CATT": -0.6,
    "CCGA": -1.5,  "CCGC": -0.7,  "CCGG": -1.5,  "CCGT": -1.0,
    "CGCA": -1.5,  "CGCC": -1.1,  "CGCG": -1.5,  "CGCT": -1.4,
    "CGTA": -0.8,  "CGTC": -0.6,  "CGTG": -0.8,  "CGTT": -0.6,
    "CTAA": -1.0,  "CTAC": -0.7,  "CTAG": -1.0,  "CTAT": -0.8,
    "CTGA": -1.0,  "CTGC": -0.7,  "CTGG": -1.0,  "CTGT": -0.8,
    "GATA": -1.1,  "GATC": -0.7,  "GATG": -1.2,  "GATT": -0.7,
    "GCGA": -1.3,  "GCGC": -1.1,  "GCGG": -1.4,  "GCGT": -1.1,
    "GGCA": -1.4,  "GGCC": -1.0,  "GGCG": -1.6,  "GGCT": -1.0,
    "GGTA": -1.1,  "GGTC": -0.7,  "GGTG": -0.8,  "GGTT": -0.7,
    "GTAA": -0.8,  "GTAC": -0.6,  "GTAG": -0.8,  "GTAT": -0.6,
    "GTGA": -0.8,  "GTGC": -0.6,  "GTGG": -0.8,  "GTGT": -0.6,
    "TATA": -0.8,  "TATC": -0.5,  "TATG": -0.8,  "TATT": -0.5,
    "TCGA": -1.5,  "TCGC": -0.5,  "TCGG": -1.5,  "TCGT": -0.7,
    "TGCA": -1.5,  "TGCC": -0.8,  "TGCG": -1.5,  "TGCT": -1.2,
    "TGTA": -0.8,  "TGTC": -0.5,  "TGTG": -0.8,  "TGTT": -0.5,
    "TTAA": -1.0,  "TTAC": -0.7,  "TTAG": -1.0,  "TTAT": -0.8,
    "TTGA": -1.0,  "TTGC": -0.7,  "TTGG": -1.0,  "TTGT": -0.6,
}

# ── Dangling-end energies ──────────────────────────────────────────────────────
# DANGLE_3: Key = seq[j-1]+seq[j]+seq[j+1]  (inner-adjacent + 3'-terminal + dangle)
# DANGLE_5: Key = seq[k]+seq[j]+seq[k-1]    (5'-terminal + 3'-terminal + 5'-dangle)
DANGLE_3: dict[str, float] = {
    "AAT": -0.3,  "ACG": -0.5,  "AGC": -0.2,  "AGT": -0.3,
    "ATA": -0.3,  "ATG": -0.3,  "CAT": -0.3,  "CCG": -0.3,
    "CGC": -0.3,  "CGT": -0.3,  "CTA": -0.1,  "CTG": -0.1,
    "GAT": -0.4,  "GCG": -0.2,  "GGT": -0.4,  "GTA": -0.2,
    "GTG": -0.2,  "TAT": -0.2,  "TCG": -0.1,  "TGT": -0.2,
    "TTA": -0.2,  "TTG": -0.2,
}

DANGLE_5: dict[str, float] = {
    "ATA": -0.7,  "ATC": -0.1,  "ATG": -0.7,  "ATT": -0.1,
    "CGA": -1.1,  "CGC": -0.4,  "CGG": -1.3,  "CGT": -0.6,
    "GCA": -1.7,  "GCC": -0.8,  "GCG": -1.7,  "GCT": -1.2,
    "GTA": -0.7,  "GTC": -0.1,  "GTG": -0.7,  "GTT": -0.1,
    "TAA": -0.8,  "TAC": -0.5,  "TAG": -0.8,  "TAT": -0.6,
    "TGA": -0.8,  "TGC": -0.5,  "TGG": -0.8,  "TGT": -0.6,
}

# ── Special hairpin sequences ──────────────────────────────────────────────────
HAIRPIN_TRILOOP: dict[str, float] = {
    "CAACG": 6.8,
    "GUUAC": 6.9,
}

HAIRPIN_TETRALOOP: dict[str, float] = {
    "CAACGG": 5.5, "CCAAGG": 3.3, "CCACGG": 3.7, "CCCAGG": 3.4,
    "CCGAGG": 3.5, "CCGCGG": 3.6, "CCUAGG": 3.7, "CCUCGG": 2.5,
    "CUAAGG": 3.6, "CUACGG": 2.8, "CUAGGG": 3.5, "CUCCGG": 2.5,
    "CUCGGG": 3.3, "CUUGGG": 2.5, "GAUUC":  0.0, "GCAAAC": 3.7,
}
