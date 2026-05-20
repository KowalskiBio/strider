"""
Secondary structure export to standard bioinformatics formats.

Supported: Vienna (.rna), CT (connectivity table), BPSEQ, FASTA, oxDNA.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def to_vienna(sequence: str, structure: str, name: str = "seq") -> str:
    """
    Vienna format: two-line header/sequence + dot-bracket.

    Compatible with mfold, RNAfold, and most secondary-structure tools.
    """
    return f">{name}\n{sequence}\n{structure}\n"


def to_ct(
    sequence: str,
    structure: str,
    name: str = "seq",
    energy: float = 0.0,
) -> str:
    """
    Connectivity Table (CT) format used by mfold and RNAstructure.

    Columns: position, base, prev, next, paired_with, original_position
    """
    from strider.structure.dot_bracket import parse_pairs
    seq = sequence.upper()
    n = len(seq)
    pairs = dict(parse_pairs(structure))
    pairs.update({j: i for i, j in pairs.items()})

    header = f"{n} dG = {energy:.2f}  {name}\n"
    rows = []
    for i in range(n):
        pos = i + 1
        base = seq[i]
        prev_ = i if i > 0 else 0
        next_ = i + 2 if i < n - 1 else 0
        paired = pairs.get(i, -1) + 1   # 0 = unpaired
        rows.append(f"{pos}\t{base}\t{prev_}\t{next_}\t{paired}\t{pos}")
    return header + "\n".join(rows) + "\n"


def to_bpseq(
    sequence: str,
    structure: str,
    header: str = "# strider output",
) -> str:
    """
    BPSEQ format: position, base, paired_position (0 = unpaired).

    Used by several RNA database tools.
    """
    from strider.structure.dot_bracket import parse_pairs
    seq = sequence.upper()
    pairs = dict(parse_pairs(structure))
    pairs.update({j: i for i, j in pairs.items()})

    lines = [header]
    for i, base in enumerate(seq):
        paired = pairs.get(i, -1) + 1
        lines.append(f"{i + 1} {base} {paired}")
    return "\n".join(lines) + "\n"


def to_fasta(sequence: str, name: str = "seq", description: str = "") -> str:
    """FASTA format."""
    header = f">{name}"
    if description:
        header += f" {description}"
    return f"{header}\n{sequence}\n"


def to_oxdna(
    sequence: str,
    structure: str | None = None,
    box_nm: float = 20.0,
) -> str:
    """
    oxDNA configuration format header for MD simulations.

    Generates a minimal topology + configuration skeleton.
    Full 3D coordinates require oxDNA's own generator (tacoxDNA recommended).
    This outputs the sequence-level topology file.
    """
    seq = sequence.upper().replace("U", "T")
    n = len(seq)

    # topology file content
    topo_lines = [f"{n} 1"]  # N strands=1
    for i, base in enumerate(seq):
        prev_ = i - 1 if i > 0 else -1
        next_ = i + 1 if i < n - 1 else -1
        topo_lines.append(f"0 {base} {prev_} {next_}")

    return "\n".join(topo_lines) + "\n"


def write(
    sequence: str,
    structure: str | None = None,
    path: str | Path = "output.rna",
    fmt: str = "vienna",
    name: str = "seq",
    energy: float = 0.0,
) -> Path:
    """
    Write structure to file in the specified format.

    fmt options: 'vienna', 'ct', 'bpseq', 'fasta', 'oxdna'
    """
    path = Path(path)
    struct = structure or "." * len(sequence)

    if fmt == "vienna":
        content = to_vienna(sequence, struct, name)
    elif fmt == "ct":
        content = to_ct(sequence, struct, name, energy)
    elif fmt == "bpseq":
        content = to_bpseq(sequence, struct)
    elif fmt == "fasta":
        content = to_fasta(sequence, name)
    elif fmt == "oxdna":
        content = to_oxdna(sequence, struct)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Choose from: vienna, ct, bpseq, fasta, oxdna")

    path.write_text(content)
    return path
