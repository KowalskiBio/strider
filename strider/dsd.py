"""
Lightweight domain-level (DSD) compiler.

Users describe a DNA-strand-displacement circuit in *domain space* —
e.g. ``S1 = <a b c>``, ``S2 = <c* b* a*>`` — and the compiler:

  1. Resolves every domain name to a concrete nucleotide sequence, generating
     reverse complements (``a*``) automatically from the base domain ``a``.
  2. Concatenates domain sequences to assemble each strand.
  3. Hands the resulting ``{strand_name: sequence}`` dict off to
     :class:`strider.bridge.mantis_bridge.CircuitBridge`, which derives
     ΔΔG via the active ``ThermoEngine`` and emits a mantis ``CRNetwork``.

This is intentionally minimal — it does NOT enumerate reactions from the
strand topology the way Visual DSD or Peppercorn do.  Reactions are still
written explicitly by the user; the compiler's job is just to keep the
sequence layer in sync with the symbolic layer.

Example
-------
>>> dsd = DSDCompiler(domains={
...     "a":  "GCATGC",            # toehold
...     "b":  "ATGCATATGC",        # branch migration region
... })
>>> dsd.add_strand("S1", ["a", "b"])
>>> dsd.add_strand("S2", ["b*", "a*"])
>>> dsd.add_reaction("S1 + S2 <-> S1S2", toehold="a")
>>> bridge = dsd.to_bridge()       # CircuitBridge
>>> rn = bridge.to_crnetwork()     # mantis CRNetwork
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strider.thermo.nn_dna import reverse_complement

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.bridge.mantis_bridge import CircuitBridge


@dataclass
class DSDCompiler:
    """
    Compile a domain-level DSD circuit to nucleotide sequences and a
    strider ``CircuitBridge``.

    Attributes
    ----------
    domains   : domain name → nucleotide sequence (DNA, upper-case).  Star
                names (``a*``) are derived automatically as reverse complements
                of their base (``a``) if not given explicitly.
    strands   : strand name → ordered list of domain names.
    reactions : list of (mantis-style reaction string, toehold domain or None).
    """
    domains: dict[str, str] = field(default_factory=dict)
    strands: dict[str, list[str]] = field(default_factory=dict)
    reactions: list[tuple[str, str | None]] = field(default_factory=list)

    def __post_init__(self):
        # Normalize all base domain sequences to upper-case DNA.
        self.domains = {
            n: v.upper().replace("U", "T") for n, v in self.domains.items()
        }

    # ─── domain / strand / reaction registration ─────────────────────────────

    def add_domain(self, name: str, sequence: str) -> None:
        """Register a new base domain.  Its complement ``name*`` is derived on demand."""
        if name.endswith("*"):
            raise ValueError(
                f"register the base domain {name[:-1]!r}, not the complement {name!r}"
            )
        self.domains[name] = sequence.upper().replace("U", "T")

    def add_strand(self, name: str, domains: list[str]) -> None:
        """Define a strand as an ordered list of domain names (5' → 3')."""
        for d in domains:
            base = d.rstrip("*")
            if base not in self.domains:
                raise ValueError(
                    f"strand {name!r} references unknown domain {d!r}; "
                    f"register {base!r} first"
                )
        self.strands[name] = list(domains)

    def add_reaction(self, reaction_string: str, toehold: str | None = None) -> None:
        """
        Register a mantis-style reaction.

        ``toehold`` (optional) is the name of the toehold domain that triggers
        the reaction; its length sets the forward rate via the Zhang & Winfree
        empirical model.
        """
        self.reactions.append((reaction_string, toehold))

    # ─── sequence resolution ─────────────────────────────────────────────────

    def domain_sequence(self, name: str) -> str:
        """Resolve a domain (or its starred complement) to its DNA sequence."""
        if name in self.domains:
            return self.domains[name]
        if name.endswith("*"):
            base = name[:-1]
            if base in self.domains:
                return reverse_complement(self.domains[base])
        raise KeyError(f"unknown domain {name!r}")

    def strand_sequence(self, name: str) -> str:
        """Concatenated DNA sequence of a registered strand (5' → 3')."""
        if name not in self.strands:
            raise KeyError(f"unknown strand {name!r}")
        return "".join(self.domain_sequence(d) for d in self.strands[name])

    def sequences(self) -> dict[str, str]:
        """All strand sequences as a dict suitable for :class:`CircuitBridge`."""
        return {n: self.strand_sequence(n) for n in self.strands}

    # ─── bridge construction ─────────────────────────────────────────────────

    def to_bridge(
        self,
        engine: "ThermoEngine | None" = None,
        include_leakage: bool = False,
        leakage_threshold: float = -4.0,
    ) -> "CircuitBridge":
        """Build a :class:`CircuitBridge` consuming the compiled sequences and reactions."""
        from strider.bridge.mantis_bridge import CircuitBridge

        toehold_map: dict[str, int] = {}
        rxn_strings = []
        for rxn, th in self.reactions:
            rxn_strings.append(rxn)
            if th is not None:
                toehold_map[rxn] = len(self.domain_sequence(th))

        return CircuitBridge(
            reactions=rxn_strings,
            sequences=self.sequences(),
            engine=engine,
            toehold_map=toehold_map,
            include_leakage=include_leakage,
            leakage_threshold=leakage_threshold,
        )

    # ─── pretty-printing ─────────────────────────────────────────────────────

    def __str__(self) -> str:
        lines = ["DSDCompiler:"]
        if self.domains:
            lines.append("  Domains:")
            for n, s in self.domains.items():
                lines.append(f"    {n:>6} = {s}  (len {len(s)})")
                comp = reverse_complement(s)
                lines.append(f"    {n+'*':>6} = {comp}")
        if self.strands:
            lines.append("  Strands:")
            for n, ds in self.strands.items():
                seq = self.strand_sequence(n)
                lines.append(f"    {n:>6} = <{' '.join(ds)}>  →  {seq}")
        if self.reactions:
            lines.append("  Reactions:")
            for rxn, th in self.reactions:
                th_str = f"  (toehold={th}, {len(self.domain_sequence(th))} nt)" if th else ""
                lines.append(f"    {rxn}{th_str}")
        return "\n".join(lines)
