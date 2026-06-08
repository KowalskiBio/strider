"""
CHA domain architecture — lay out the strands of a catalytic-hairpin-assembly
biosensor from a target sequence.

Given a target (e.g. a miRNA), the 3′ end is the initiation toehold **D1** and
the adjacent block is the branch-migration stem **D2**; **L** is the hairpin
loop and **K** an optional orthogonal capture handle on H1 (CP = K*):

    target =  [ … 5' overhang … ][   D2   ][  D1  ]   (5'→3')
                                  (stem)    (3' toehold)

    H1  =  D1*  D2*  L   D2   K          (5'→3')
    H2  =  D2*  L   D2   D1              (5'→3')
    CP  =  K*                            (5'→3', when a handle is designed)

Pure string manipulation; all thermodynamics live in
:class:`strider.thermo.engine.ThermoEngine`.  This is the *generator* half of
:class:`strider.circuits.cha.CHA`, which on its own only *checks* finished
sequences.

References
----------
Yin P. et al. (2008) Nature 451:318-322.
"""

from __future__ import annotations

from dataclasses import dataclass

from strider.thermo.nn_dna import reverse_complement


def normalize(seq: str) -> str:
    """Upper-case and map RNA→DNA (U→T)."""
    return seq.upper().replace("U", "T").strip()


@dataclass(frozen=True)
class CHADomains:
    """Resolved domain sequences for one CHA design (all 5'→3', DNA)."""
    target: str         # target, normalised to DNA alphabet
    d1: str             # 3' toehold of the target
    d2: str             # branch-migration stem region of the target
    overhang: str       # 5' target nucleotides that dangle (unused)
    loop_h1: str        # H1 hairpin loop
    loop_h2: str        # H2 hairpin loop
    capture: str        # K, the orthogonal capture handle on H1 (CP = K*)

    @property
    def lengths(self) -> dict[str, int]:
        return {"d1": len(self.d1), "d2": len(self.d2),
                "overhang": len(self.overhang), "loop_h1": len(self.loop_h1),
                "loop_h2": len(self.loop_h2), "capture": len(self.capture)}


def split_target(target: str, d1_len: int, d2_len: int) -> tuple[str, str, str]:
    """
    Split a target into (overhang, D2, D1) from 5'→3'.

    D1 is the 3'-terminal ``d1_len`` nt (initiation toehold); D2 is the
    ``d2_len`` nt immediately 5' of D1 (branch-migration stem); anything further
    5' is the unused overhang.
    """
    seq = normalize(target)
    if d1_len + d2_len > len(seq):
        raise ValueError(
            f"target length {len(seq)} too short for d1={d1_len} + d2={d2_len}")
    d1 = seq[len(seq) - d1_len:]
    d2 = seq[len(seq) - d1_len - d2_len: len(seq) - d1_len]
    overhang = seq[: len(seq) - d1_len - d2_len]
    return overhang, d2, d1


def build_domains(target: str, *, d1_len: int, d2_len: int,
                  loop_h1: str, loop_h2: str, capture: str = "") -> CHADomains:
    seq = normalize(target)
    overhang, d2, d1 = split_target(seq, d1_len, d2_len)
    return CHADomains(
        target=seq, d1=d1, d2=d2, overhang=overhang,
        loop_h1=normalize(loop_h1), loop_h2=normalize(loop_h2),
        capture=normalize(capture))


def assemble(domains: CHADomains, trigger_key: str = "mirna") -> dict[str, str]:
    """
    Concrete strand sequences for ``{trigger_key, H1, H2, CP}`` (5'→3', DNA).

    H1 carries the capture handle K at its 3' end; CP = K*.  H2 has no capture
    tail.  When ``capture`` is empty the CP entry is empty (cascade-only
    assembly, e.g. during split selection).  ``trigger_key`` names the target
    species in the returned dict (default ``"mirna"`` for CHA compatibility).
    """
    d1, d2 = domains.d1, domains.d2
    h1 = (reverse_complement(d1) + reverse_complement(d2)
          + domains.loop_h1 + d2 + domains.capture)
    h2 = reverse_complement(d2) + domains.loop_h2 + d2 + d1
    cp = reverse_complement(domains.capture) if domains.capture else ""
    return {trigger_key: domains.target, "H1": h1, "H2": h2, "CP": cp}


def h1_core(domains: CHADomains) -> str:
    """H1 without the capture handle (the cascade-relevant strand)."""
    return (reverse_complement(domains.d1) + reverse_complement(domains.d2)
            + domains.loop_h1 + domains.d2)


def structures(domains: CHADomains) -> dict[str, str]:
    """Intended dot-bracket structures of the closed hairpins (for reporting)."""
    n_d1, n_d2 = len(domains.d1), len(domains.d2)
    n_l1, n_l2 = len(domains.loop_h1), len(domains.loop_h2)
    n_k = len(domains.capture)
    h1 = "." * n_d1 + "(" * n_d2 + "." * n_l1 + ")" * n_d2 + "." * n_k
    h2 = "(" * n_d2 + "." * n_l2 + ")" * n_d2 + "." * n_d1
    return {"H1": h1, "H2": h2, "CP": "." * n_k}


def toehold_positions(domains: CHADomains) -> list[int]:
    """0-based positions of the H1 initiation toehold (D1*, the 5' end)."""
    return list(range(len(domains.d1)))


# ─── loop-motif library (low-structure, AT-rich clamps) ─────────────────────────

LOOP_MOTIFS = [
    "ACTTAATTAAGT", "TACAATTACAAT", "ATCATACAATCA",
    "AACTTACAATCA", "TCATTACAATGA",
]


def _tile(motif: str, n: int) -> str:
    return (motif * (n // len(motif) + 1))[:n]


def clean_sequence(seq: str, forbidden: tuple[str, ...], max_run: int) -> bool:
    """True iff ``seq`` has no forbidden motif and no homopolymer run > max_run."""
    s = seq.upper()
    if any(f in s for f in forbidden):
        return False
    run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] else 1
        if run > max_run:
            return False
    return True


def loop_candidates(loop_len: int, forbidden: tuple[str, ...],
                    max_run: int) -> list[str]:
    """Curated low-structure loop motifs tiled/trimmed to ``loop_len``."""
    cands = [_tile(m, loop_len) for m in LOOP_MOTIFS]
    cands = [c for c in cands if clean_sequence(c, forbidden, max_run)]
    return cands or [_tile(LOOP_MOTIFS[0], loop_len)]
