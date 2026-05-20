"""
Leaf decomposition for multi-strand design problems.

A multi-strand design objective often factorises naturally: independent
subsets of domains constrain disjoint subsets of complexes, so the global
optimum can be reached by optimising each *leaf* (connected component)
separately and only re-optimising at the joint level after each leaf has
converged (Wolfe & Pierce 2015, J. Comput. Chem. 36:255-269 §2.3).

This module exposes a thin API:

* :func:`build_strand_graph` — adjacency from an iterable of
  ``Complex``-like objects (anything exposing ``strand_names``).
* :func:`connected_components` — undirected connected components.
* :func:`decompose_assays` — split an :class:`~strider.design.assay.Assay`
  or :class:`~strider.design.assay.AssayPanel` into sub-assays whose
  domains do not overlap.

The decomposition is purely topological — it does not look at sequence
identity or structures.  Two assemblies that share *any* domain land in
the same leaf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from strider.design.assay import Assay, AssayPanel, Assembly


# ─── graph primitives ─────────────────────────────────────────────────────────


def build_strand_graph(complexes: Iterable["object"]) -> dict[str, set[str]]:
    """
    Build an undirected adjacency map: strand name → strands that co-occur
    in some complex.

    Each input object must expose either a ``strand_names`` attribute
    (sequence of names) or a ``strands`` list of strings.  Singletons
    appear as isolated nodes mapping to an empty set.
    """
    adj: dict[str, set[str]] = {}
    for cx in complexes:
        names = list(getattr(cx, "strand_names", None) or getattr(cx, "strands", []))
        for n in names:
            adj.setdefault(n, set())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if a == b:
                    continue
                adj[a].add(b)
                adj[b].add(a)
    return adj


def connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    """Return the connected components of an undirected adjacency map."""
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            stack.extend(adjacency.get(cur, ()) - seen)
        components.append(comp)
    return components


# ─── assay-level decomposition ────────────────────────────────────────────────


def decompose_assays(
    assays: "Assay | AssayPanel | Sequence[Assay]",
) -> list["Assay"]:
    """
    Split an assay (or panel) into the smallest independent sub-assays.

    Two assemblies belong to the same leaf if any of their strand-name
    sets intersect.  On-target and off-target assemblies are routed into
    the same leaf when they share strands so off-target penalties stay
    co-optimised with the structures they suppress.

    Returns a list of new :class:`Assay` instances (copies — the input is
    not mutated).  Order matches first-appearance of each leaf in the
    input assay list.
    """
    from strider.design.assay import Assay, AssayPanel

    if isinstance(assays, Assay):
        assay_list = [assays]
    elif isinstance(assays, AssayPanel):
        assay_list = list(assays.assays)
    else:
        assay_list = list(assays)

    out: list[Assay] = []
    for assay in assay_list:
        all_assemblies = list(assay.on_targets) + list(assay.off_targets)
        if not all_assemblies:
            out.append(assay)
            continue

        adj = build_strand_graph(_assembly_complex(a) for a in all_assemblies)
        comps = connected_components(adj)
        # Empty assemblies (no strands) — keep as own leaf.
        if not comps:
            out.append(assay)
            continue

        comp_index: dict[str, int] = {}
        for i, comp in enumerate(comps):
            for name in comp:
                comp_index[name] = i

        buckets_on: dict[int, list] = {i: [] for i in range(len(comps))}
        buckets_off: dict[int, list] = {i: [] for i in range(len(comps))}

        for asm in assay.on_targets:
            names = list(asm.strands)
            i = comp_index[names[0]] if names else 0
            buckets_on[i].append(asm)
        for asm in assay.off_targets:
            names = list(asm.strands)
            i = comp_index[names[0]] if names else 0
            buckets_off[i].append(asm)

        for i in range(len(comps)):
            if not buckets_on[i] and not buckets_off[i]:
                continue
            leaf = Assay(
                name=f"{assay.name}__leaf{i}",
                on_targets=buckets_on[i],
                off_targets=buckets_off[i],
                off_target_ddg_threshold=assay.off_target_ddg_threshold,
                off_target_penalty_weight=assay.off_target_penalty_weight,
            )
            out.append(leaf)

    return out


def _assembly_complex(assembly: "Assembly"):
    """Adapter so Assembly looks like a Complex for build_strand_graph."""
    return assembly.complex
