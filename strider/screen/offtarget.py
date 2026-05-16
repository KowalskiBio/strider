"""
Off-target sequence screening.

Uses k-mer indexing against a reference FASTA database,
then computes ΔΔG for candidate hits via ThermoEngine.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine


@dataclass
class OffTargetHit:
    name: str
    sequence: str
    ddg: float
    alignment: str
    k_score: int    # number of shared k-mers


@dataclass
class ScreeningReport:
    query: str
    hits: list[OffTargetHit]
    top_ddg: float
    is_specific: bool
    n_screened: int

    def __repr__(self) -> str:
        return (
            f"ScreeningReport(query={self.query[:10]}..., "
            f"hits={len(self.hits)}, specific={self.is_specific})"
        )


class OffTargetScreener:
    """
    Screen a probe sequence against a reference database.

    Parameters
    ----------
    engine       : ThermoEngine for ΔΔG calculations
    reference_db : path to FASTA file of reference sequences (e.g. miRBase)
    kmer_k       : k-mer length for pre-filtering candidates
    """

    def __init__(
        self,
        engine: "ThermoEngine",
        reference_db: str | Path | None = None,
        kmer_k: int = 7,
    ) -> None:
        self.engine = engine
        self.kmer_k = kmer_k
        self._db: dict[str, str] = {}   # name → sequence
        self._kmer_index: dict[str, list[str]] = {}  # kmer → list of names

        if reference_db is not None:
            self.load_database(reference_db)

    def load_database(self, path: str | Path) -> int:
        """Load a FASTA database. Returns number of sequences loaded."""
        path = Path(path)
        seqs = _parse_fasta(path)
        self._db.update(seqs)
        self._build_kmer_index(seqs)
        return len(seqs)

    def add_sequences(self, sequences: dict[str, str]) -> None:
        """Add sequences directly (no file needed)."""
        self._db.update(sequences)
        self._build_kmer_index(sequences)

    def screen(
        self,
        sequence: str,
        n_top: int = 20,
        ddg_threshold: float = -4.0,
    ) -> ScreeningReport:
        """
        Screen a query against the loaded database.

        Returns top hits ranked by ΔΔG (most negative = most concerning).
        """
        seq = sequence.upper().replace("U", "T")

        # K-mer pre-filter
        candidates = self._kmer_candidates(seq)
        if not candidates:
            candidates = list(self._db.keys())[:100]  # fallback: scan all

        hits: list[OffTargetHit] = []
        for name in candidates[:200]:  # cap at 200 for speed
            ref = self._db[name].upper().replace("U", "T")
            try:
                ddg = self.engine.duplex_dg(seq, ref)
            except Exception:
                continue
            if ddg < ddg_threshold:
                shared = len(set(_kmers(seq, self.kmer_k)) & set(_kmers(ref, self.kmer_k)))
                hits.append(OffTargetHit(
                    name=name,
                    sequence=ref,
                    ddg=ddg,
                    alignment=_simple_align(seq, ref),
                    k_score=shared,
                ))

        hits.sort(key=lambda h: h.ddg)
        top = hits[:n_top]
        return ScreeningReport(
            query=sequence,
            hits=top,
            top_ddg=top[0].ddg if top else 0.0,
            is_specific=(len(top) == 0),
            n_screened=len(candidates),
        )

    def specificity_vs(
        self,
        sequence: str,
        family_members: list[str],
        target: str,
    ) -> dict[str, float]:
        """
        Compute ΔΔG selectivity vs. a set of related sequences.

        Returns {family_member_seq: ΔΔG_vs_target} where positive = more selective.
        """
        seq = sequence.upper().replace("U", "T")
        tgt = target.upper().replace("U", "T")
        ddg_target = self.engine.duplex_dg(seq, tgt)
        results: dict[str, float] = {}
        for member in family_members:
            m = member.upper().replace("U", "T")
            try:
                ddg_m = self.engine.duplex_dg(seq, m)
                results[member] = ddg_m - ddg_target  # positive = more selective vs target
            except Exception:
                results[member] = float("nan")
        return results

    def _build_kmer_index(self, seqs: dict[str, str]) -> None:
        for name, seq in seqs.items():
            for kmer in _kmers(seq.upper().replace("U", "T"), self.kmer_k):
                self._kmer_index.setdefault(kmer, []).append(name)

    def _kmer_candidates(self, query: str) -> list[str]:
        counts: dict[str, int] = {}
        for kmer in _kmers(query, self.kmer_k):
            for name in self._kmer_index.get(kmer, []):
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts, key=lambda n: counts[n], reverse=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _kmers(seq: str, k: int) -> list[str]:
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def _parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    current_name = None
    current_seq: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_name:
                seqs[current_name] = "".join(current_seq)
            current_name = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_name:
        seqs[current_name] = "".join(current_seq)
    return seqs


def _simple_align(s1: str, s2: str) -> str:
    """Very basic alignment string for display."""
    min_len = min(len(s1), len(s2))
    match = "".join("|" if s1[i] == s2[i] else "." for i in range(min_len))
    return f"{s1[:min_len]}\n{match}\n{s2[:min_len]}"
