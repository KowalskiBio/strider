"""
Dot-bracket secondary structure parser and validator.

Supports: ( ) standard pairs, [ ] pseudoknot level 1, { } level 2.
Also handles multi-strand separator '&' and '+'.
"""

from __future__ import annotations


OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {v: k for k, v in OPEN.items()}
PAIR_CHARS = set(OPEN) | set(CLOSE)


def parse_pairs(structure: str) -> list[tuple[int, int]]:
    """
    Parse a dot-bracket string into a list of (i, j) base-pair indices (0-based).

    Handles nested pairs at three levels: () [] {}
    """
    structure = structure.replace("&", "").replace("+", "")
    pairs: list[tuple[int, int]] = []
    stacks: dict[str, list[int]] = {c: [] for c in OPEN}

    for idx, ch in enumerate(structure):
        if ch in OPEN:
            stacks[ch].append(idx)
        elif ch in CLOSE:
            opener = CLOSE[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append((j, idx))
            # mismatched bracket: skip

    return sorted(pairs)


def to_dot_bracket(pairs: list[tuple[int, int]], length: int) -> str:
    """Convert a list of (i, j) pairs to dot-bracket notation."""
    db = ["."] * length
    for i, j in pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)


def validate(structure: str) -> bool:
    """Return True if the dot-bracket string is valid (balanced brackets)."""
    stacks: dict[str, list[int]] = {c: [] for c in OPEN}
    for ch in structure:
        if ch in OPEN:
            stacks[ch].append(ch)
        elif ch in CLOSE:
            opener = CLOSE[ch]
            if not stacks[opener]:
                return False
            stacks[opener].pop()
        elif ch not in (".", "&", "+"):
            return False
    return all(len(s) == 0 for s in stacks.values())


def stem_regions(structure: str) -> list[tuple[int, int, int]]:
    """
    Find contiguous stem regions in a dot-bracket structure.

    Returns list of (start_i, start_j, length) for each stem.
    A stem is a run of consecutively nested base pairs.
    """
    pairs = parse_pairs(structure)
    if not pairs:
        return []

    pair_set = {(i, j) for i, j in pairs}
    visited: set[tuple[int, int]] = set()
    stems = []

    for i, j in sorted(pairs):
        if (i, j) in visited:
            continue
        length = 1
        visited.add((i, j))
        i2, j2 = i + 1, j - 1
        while (i2, j2) in pair_set and (i2, j2) not in visited:
            visited.add((i2, j2))
            length += 1
            i2 += 1
            j2 -= 1
        stems.append((i, j, length))

    return stems


def split_strands(structure: str) -> list[str]:
    """Split a multi-strand dot-bracket into per-strand structures."""
    return [s for s in structure.split("&") if s]


def unpaired_positions(structure: str) -> list[int]:
    """Return 0-based indices of all unpaired positions."""
    clean = structure.replace("&", "").replace("+", "")
    return [i for i, c in enumerate(clean) if c == "."]
