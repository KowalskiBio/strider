"""
Spurious reaction pathway enumeration (LeakageEnumerator).

Systematically checks all pairwise and tripartite strand combinations
for thermodynamically favorable off-pathway complexes. This automates
what is typically done by hand in hairpin/CHA design.

Algorithm:
    1. For each subset of 2 (or 3) strands, compute the ΔΔG of complex
       formation using ThermoEngine.ddg().
    2. Filter by ddg_threshold (keep reactions with ΔΔG < threshold).
    3. Classify by mechanism (hybridization, displacement, cooperative).
    4. Return LeakageReport with mantis-compatible reaction strings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine


@dataclass
class SpuriousReaction:
    reactant_names: list[str]
    product_complex: list[str]        # strands forming the spurious complex
    ddg: float                        # kcal/mol (negative = thermodynamically favored)
    pathway_type: str                 # "hybridization" | "displacement" | "cooperative"
    mantis_string: str

    def __repr__(self) -> str:
        return f"SpuriousReaction({self.mantis_string}, ΔΔG={self.ddg:.2f})"

    def product_complex_name(self) -> str:
        """Underscore-joined name of the product complex (e.g. 'H1_H2')."""
        return "_".join(self.product_complex)


@dataclass
class LeakageReport:
    reactions: list[SpuriousReaction] = field(default_factory=list)
    strands_checked: int = 0
    total_spurious: int = 0
    worst_ddg: float = 0.0
    summary: str = ""

    def to_mantis_strings(self) -> list[str]:
        """Return a list of mantis-style reaction strings for all spurious reactions."""
        return [r.mantis_string for r in self.reactions]

    def filter(self, ddg_threshold: float) -> "LeakageReport":
        """Return a new LeakageReport containing only reactions with ΔΔG < ddg_threshold."""
        kept = [r for r in self.reactions if r.ddg < ddg_threshold]
        return LeakageReport(
            reactions=kept,
            strands_checked=self.strands_checked,
            total_spurious=len(kept),
            worst_ddg=min((r.ddg for r in kept), default=0.0),
            summary=f"{len(kept)} spurious reactions below {ddg_threshold} kcal/mol",
        )

    def __repr__(self) -> str:
        return (
            f"LeakageReport({self.total_spurious} spurious reactions, "
            f"worst ΔΔG={self.worst_ddg:.2f} kcal/mol)"
        )


class LeakageEnumerator:
    """
    Enumerate thermodynamically favorable spurious reactions between a set of strands.

    Parameters
    ----------
    engine          : ThermoEngine for free energy calculations
    ddg_threshold   : only report reactions with ΔΔG < threshold (kcal/mol)
    max_complex_size: max number of strands in a spurious complex (2 or 3)
    max_pathways    : stop after this many spurious reactions found
    """

    def __init__(
        self,
        engine: "ThermoEngine",
        ddg_threshold: float = -4.0,
        max_complex_size: int = 3,
        max_pathways: int = 100,
    ) -> None:
        self.engine = engine
        self.ddg_threshold = ddg_threshold
        self.max_complex_size = max(2, min(max_complex_size, 3))
        self.max_pathways = max_pathways

    def enumerate(
        self,
        strands: dict[str, str],
        intended_reactions: list[str] | None = None,
    ) -> LeakageReport:
        """
        Enumerate spurious reactions for a given set of strands.

        strands             : name → sequence mapping
        intended_reactions  : list of mantis-style reaction strings to EXCLUDE

        Returns LeakageReport with all spurious reactions.
        """
        intended_set = _parse_intended(intended_reactions or [])
        names = list(strands.keys())
        reactions: list[SpuriousReaction] = []
        checked = 0

        # Pairwise complexes
        for n1, n2 in combinations(names, 2):
            checked += 1
            product = [n1, n2]
            key = frozenset(product)
            if key in intended_set:
                continue

            seq1 = strands[n1]
            seq2 = strands[n2]
            try:
                ddg = self.engine.ddg([seq1, seq2], [[seq1, seq2]])
            except Exception:
                ddg = 0.0

            if ddg < self.ddg_threshold:
                rxn = SpuriousReaction(
                    reactant_names=[n1, n2],
                    product_complex=product,
                    ddg=ddg,
                    pathway_type="hybridization",
                    mantis_string=f"{n1} + {n2} -> {n1}_{n2}",
                )
                reactions.append(rxn)
                if len(reactions) >= self.max_pathways:
                    break

        # Tripartite complexes (optional)
        if self.max_complex_size >= 3 and len(reactions) < self.max_pathways:
            for n1, n2, n3 in combinations(names, 3):
                checked += 1
                key = frozenset([n1, n2, n3])
                if key in intended_set:
                    continue

                seqs = [strands[n1], strands[n2], strands[n3]]
                try:
                    ddg = self.engine.ddg(seqs, [seqs])
                except Exception:
                    ddg = 0.0

                if ddg < self.ddg_threshold:
                    rxn = SpuriousReaction(
                        reactant_names=[n1, n2, n3],
                        product_complex=[n1, n2, n3],
                        ddg=ddg,
                        pathway_type="cooperative",
                        mantis_string=f"{n1} + {n2} + {n3} -> {n1}_{n2}_{n3}",
                    )
                    reactions.append(rxn)
                    if len(reactions) >= self.max_pathways:
                        break

        # Also check displacement-type leakage: A + BC -> AC + B
        displacement = self._check_displacement_leakage(strands, intended_set, reactions)
        reactions.extend(displacement)

        reactions.sort(key=lambda r: r.ddg)
        worst = reactions[0].ddg if reactions else 0.0
        summary = (
            f"Found {len(reactions)} spurious reactions (threshold {self.ddg_threshold} kcal/mol). "
            f"Worst: {worst:.2f} kcal/mol."
            if reactions else "No spurious reactions found above threshold."
        )

        return LeakageReport(
            reactions=reactions,
            strands_checked=checked,
            total_spurious=len(reactions),
            worst_ddg=worst,
            summary=summary,
        )

    def _check_displacement_leakage(
        self,
        strands: dict[str, str],
        intended_set: set,
        existing: list,
    ) -> list[SpuriousReaction]:
        """
        Check leakage-style displacement: free strand opens a complex spontaneously.

        For each pair (invader, complex_strand), estimate ΔΔG of:
            invader + target_strand → invader:target_strand (opens hairpin)
        """
        names = list(strands.keys())
        result: list[SpuriousReaction] = []
        found_keys = {frozenset(r.reactant_names) for r in existing}

        for invader, target in permutations(names, 2):
            key = frozenset([invader, target])
            if key in found_keys or key in intended_set:
                continue
            found_keys.add(key)

            seq_inv = strands[invader]
            seq_tgt = strands[target]
            try:
                # ΔΔG of partial hybridization via hairpin breathing
                ddg_mono = self.engine.duplex_dg(seq_inv, seq_tgt)
                g_inv = self.engine.pfunc(seq_inv).free_energy
                g_tgt = self.engine.pfunc(seq_tgt).free_energy
                ddg = ddg_mono - g_inv - g_tgt
            except Exception:
                continue

            if ddg < self.ddg_threshold:
                rxn = SpuriousReaction(
                    reactant_names=[invader, target],
                    product_complex=[invader, target],
                    ddg=ddg,
                    pathway_type="displacement",
                    mantis_string=f"{invader} + {target} -> {invader}_{target}",
                )
                result.append(rxn)

        return result


def _parse_intended(reactions: list[str]) -> set[frozenset]:
    """Parse mantis-style reaction strings into a set of frozensets of species names for fast exclusion lookup."""
    intended: set[frozenset] = set()
    for rxn in reactions:
        sep = "<->" if "<->" in rxn else "->"
        parts = rxn.split(sep)
        for part in parts:
            species = frozenset(s.strip() for s in part.split("+") if s.strip())
            intended.add(species)
    return intended
