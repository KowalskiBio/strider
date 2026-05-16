"""
Hard constraints for sequence design.

A HardConstraint returns True if the constraint is SATISFIED.
The optimizer rejects sequences that violate any hard constraint.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable


IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]",
    "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]",
    "N": "[ACGT]",
}


@dataclass
class HardConstraint:
    """
    A single hard constraint on sequence content.

    name : human-readable description
    fn   : callable(strand_name, sequence) -> bool (True = satisfied)
    """
    name: str
    fn: Callable[[str, str], bool]

    def check(self, strand_name: str, sequence: str) -> bool:
        """Return True if this constraint is satisfied for the given strand name and sequence."""
        return self.fn(strand_name, sequence)

    def __repr__(self) -> str:
        return f"HardConstraint({self.name!r})"

    # ─── factory methods ─────────────────────────────────────────────────────

    @classmethod
    def no_repeats(
        cls,
        motifs: list[str],
        apply_to: list[str] | None = None,
    ) -> "HardConstraint":
        """Forbid any of the given sequence motifs."""
        patterns = [m.upper() for m in motifs]

        def fn(name: str, seq: str) -> bool:
            if apply_to and name not in apply_to:
                return True
            s = seq.upper()
            return not any(p in s for p in patterns)

        return cls(name=f"no_repeats({motifs})", fn=fn)

    @classmethod
    def gc_content(
        cls,
        min_gc: float = 0.3,
        max_gc: float = 0.7,
        apply_to: list[str] | None = None,
    ) -> "HardConstraint":
        """Require GC fraction within [min_gc, max_gc]."""

        def fn(name: str, seq: str) -> bool:
            if apply_to and name not in apply_to:
                return True
            if not seq:
                return True
            gc = sum(1 for b in seq.upper() if b in "GC") / len(seq)
            return min_gc <= gc <= max_gc

        return cls(name=f"gc_content([{min_gc:.0%}, {max_gc:.0%}])", fn=fn)

    @classmethod
    def no_self_complement(
        cls,
        min_length: int = 4,
        apply_to: list[str] | None = None,
    ) -> "HardConstraint":
        """Forbid self-complementary runs of >= min_length nt."""
        from strider.thermo.nn_dna import reverse_complement

        def fn(name: str, seq: str) -> bool:
            if apply_to and name not in apply_to:
                return True
            rc = reverse_complement(seq)
            for i in range(len(seq) - min_length + 1):
                sub = seq[i : i + min_length]
                if sub in rc:
                    return False
            return True

        return cls(name=f"no_self_complement(>={min_length}nt)", fn=fn)

    @classmethod
    def iupac_pattern(
        cls,
        strand_name: str,
        pattern: str,
        start: int = 0,
    ) -> "HardConstraint":
        """
        Require a specific IUPAC pattern at a given offset.

        pattern: IUPAC string, e.g. 'RNNNNNNY' means R at pos 0, Y at pos 7.
        start  : 0-based offset into the sequence.
        """
        regex = "".join(IUPAC.get(ch.upper(), ch) for ch in pattern)
        compiled = re.compile(regex)

        def fn(name: str, seq: str) -> bool:
            if name != strand_name:
                return True
            sub = seq[start : start + len(pattern)]
            return bool(compiled.fullmatch(sub))

        return cls(name=f"iupac({strand_name}, {pattern!r}@{start})", fn=fn)

    @classmethod
    def min_length(cls, length: int, apply_to: list[str] | None = None) -> "HardConstraint":
        """Require sequence length >= length."""
        def fn(name: str, seq: str) -> bool:
            if apply_to and name not in apply_to:
                return True
            return len(seq) >= length
        return cls(name=f"min_length({length})", fn=fn)

    @classmethod
    def max_run(
        cls,
        max_run_length: int = 4,
        apply_to: list[str] | None = None,
    ) -> "HardConstraint":
        """Forbid runs of the same nucleotide > max_run_length."""
        pattern = re.compile(r"(.)\1{" + str(max_run_length) + r",}")

        def fn(name: str, seq: str) -> bool:
            if apply_to and name not in apply_to:
                return True
            return not bool(pattern.search(seq.upper()))

        return cls(name=f"max_run({max_run_length})", fn=fn)

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[str, str], bool],
        name: str = "custom",
    ) -> "HardConstraint":
        return cls(name=name, fn=fn)
