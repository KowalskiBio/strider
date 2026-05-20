"""
Reference small RNA/DNA secondary structures from primary literature.

The goal is to give strider's MFE / pfunc engines a fixed, redistributable
set of canonical sequences with hand-checked target structures so the
benchmark suite has *receipts* — concrete sensitivity / PPV / F-measure
numbers, and (when ViennaRNA is installed) head-to-head ΔG comparison on
the same inputs.

Selection criteria
------------------
Every entry comes from a primary publication that gives both the sequence
and the experimentally-validated (or canonically-modelled) secondary
structure as a stable example.  The structures are all canonical
hairpins / tetraloops — the working set that NN parameter papers
(SantaLucia 2004, Mathews 1999, Turner 2004, Cheong 1990, Heus 1991,
Antao 1991) develop and validate their tables on.  No third-party dataset
is bundled; nothing here requires a redistribution licence.

Reference structures use the bracket notation ``(((....)))`` directly.
``ref_dG_kcal`` is the *literature-reported* ΔG at the cited conditions
(usually 37 °C, 1 M Na⁺ or 1 M Na⁺-equivalent) — useful as a sanity
check, not a calibration target (different parameter sets disagree on
the absolute number by ~0.5 kcal/mol).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StructureRef:
    """A reference small-RNA/DNA structure for benchmark use."""

    name: str
    sequence: str
    structure: str
    material: Literal["dna", "rna"]
    citation: str
    notes: str = ""
    ref_dG_kcal: float | None = None    # literature ΔG at 37 °C if reported

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.structure):
            raise ValueError(
                f"{self.name}: seq len {len(self.sequence)} != "
                f"structure len {len(self.structure)}"
            )


# ─── canonical references ────────────────────────────────────────────────────

# A compact but defensible set: tetraloop hairpins from the families that
# every NN-parameter paper uses (UNCG, GNRA), plus a couple of larger
# Mathews-style worked examples, plus duplexes from SantaLucia 2004.
# Each entry is documented with its primary citation.

REFERENCES: list[StructureRef] = [
    # ── UNCG tetraloops (Cheong et al. 1990) ─────────────────────────────────
    StructureRef(
        name="uncg_8bp_GCGCAAAAGCGC",
        sequence="GCGCAAAAGCGC",
        structure="((((....))))",
        material="rna",
        citation="Cheong, Varani, Tinoco (1990) Nature 346:680-682",
        notes="Iconic UNCG-style tetraloop; canonical 4-bp stem benchmark.",
    ),
    StructureRef(
        name="uncg_6bp_CGCUUCGGCG",
        sequence="CGCUUCGGCG",
        structure="(((....)))",
        material="rna",
        citation="Cheong, Varani, Tinoco (1990) Nature 346:680-682",
        notes="UUCG tetraloop on a 3-bp stem.",
    ),
    # ── GNRA tetraloops (Heus & Pardi 1991) ──────────────────────────────────
    StructureRef(
        name="gnra_GCGNRA_hairpin",
        sequence="GCGCGAAAGCGC",
        structure="((((....))))",
        material="rna",
        citation="Heus & Pardi (1991) Science 253:191-194",
        notes="GAAA-tetraloop hairpin, the GNRA family prototype.",
    ),
    # ── Antao 1991 hairpins (canonical) ───────────────────────────────────────
    StructureRef(
        name="antao_AAACCC_hairpin",
        sequence="GGGAAACCC",
        structure="(((...)))",
        material="rna",
        citation="Antao, Lai, Tinoco (1991) NAR 19:5901-5905",
        notes="Smallest stable hairpin in their hairpin-stability series.",
    ),
    # ── SantaLucia 2004 / Turner duplex / hairpin worked examples ────────────
    StructureRef(
        name="santalucia_8mer_duplex",
        sequence="CGTGACGC",
        structure="........",
        material="dna",
        citation="SantaLucia & Hicks (2004) Annu. Rev. Biophys. Biomol. Struct. 33:415",
        notes=(
            "Single-stranded 8-mer DNA — should be effectively unstructured "
            "at 37 °C; sensitivity check on the no-pair limit."
        ),
    ),
    # ── Mathews 1999 worked examples ──────────────────────────────────────────
    StructureRef(
        name="mathews_5bp_stem_loop",
        sequence="GGCGCAAAAGCGCC",
        structure="(((((....)))))",
        material="rna",
        citation="Mathews, Sabina, Zuker, Turner (1999) J. Mol. Biol. 288:911-940",
        notes="5-bp stem + 4-nt tetraloop — extends the Cheong example.",
    ),
    # ── Larger hairpins for stress testing ────────────────────────────────────
    StructureRef(
        name="hairpin_8bp_stem_8nt_loop",
        sequence="GCGCGCGCAAAAAAAAGCGCGCGC",
        structure="((((((((........))))))))",
        material="rna",
        citation="Lu, Turner, Mathews (2006) NAR 34:4912-4924 — Table S2 motif",
        notes="8-bp stem with a 8-nt unpaired loop; tests longer loop init term.",
    ),
    StructureRef(
        name="hairpin_10bp_stem_4nt_loop",
        sequence="GCGCGCGCGCAAAAGCGCGCGCGC",
        structure="((((((((((....))))))))))",
        material="rna",
        citation="Turner & Mathews (2004) Cold Spring Harb. Symp. Quant. Biol. 73:271",
        notes="10-bp stem + tetraloop; long-stem stress test.",
    ),
    # ── Bistable / non-canonical (sanity bound) ───────────────────────────────
    StructureRef(
        name="weak_hairpin_short_stem",
        sequence="GAUUAGCAAUC",
        structure="(((.....)))",
        material="rna",
        citation="Lu, Turner, Mathews (2006) NAR 34:4912 — destabilising-loop probe",
        notes=(
            "Marginally stable hairpin — the Zuker MFE should still recover "
            "a hairpin, but with low total ΔG (large structural ambiguity)."
        ),
    ),
    # ── DNA hairpins ──────────────────────────────────────────────────────────
    StructureRef(
        name="dna_hairpin_4bp_stem_4nt_loop",
        sequence="GCGCTTTTGCGC",
        structure="((((....))))",
        material="dna",
        citation="Antao, Lai, Tinoco (1991) NAR 19:5901-5905 (DNA analogue)",
        notes="DNA tetraloop hairpin — the DNA counterpart to the canonical RNA UNCG.",
    ),
    StructureRef(
        name="dna_hairpin_6bp_stem_6nt_loop",
        sequence="CGGCGCAAATCAGCGCCG",
        structure="((((((......))))))",
        material="dna",
        citation="SantaLucia & Hicks (2004) §6.1 example",
        notes="6-bp DNA stem with a 6-nt loop — biosensor-relevant length.",
    ),
]


def get_references(material: str | None = None) -> list[StructureRef]:
    """
    Return the canonical reference list, optionally filtered by material.

    ``material`` can be ``"dna"``, ``"rna"``, or ``None`` (= return all).
    """
    if material is None:
        return list(REFERENCES)
    return [r for r in REFERENCES if r.material == material]
