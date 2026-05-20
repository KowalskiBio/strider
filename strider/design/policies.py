"""
Mutation proposal policies for the sequence designer.

A :class:`MutationPolicy` decides *which* free position to flip on each
simulated-annealing step and *which* base to substitute.  The original
designer used a uniform-random policy, which wastes moves on already-good
positions when an objective is dominated by ensemble defect.  The
defect-weighted policy biases proposals toward positions whose target
pairing is currently poorly satisfied — empirically far faster on
defect-driven tasks (Zadeh, Wolfe & Pierce 2011, J. Comput. Chem.
32:439-452 §3.1; Wolfe & Pierce 2015, J. Comput. Chem. 36:255-269).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from strider.design.optimizer import DNA_BASES, RNA_BASES, _bases_for

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.design.constraints import HardConstraint
    from strider.design.optimizer import DomainSpec


Proposal = tuple[str, int, str]   # (domain_name, position, new_base)


class MutationPolicy:
    """Abstract proposal strategy. Subclasses implement :meth:`propose`."""

    def propose(
        self,
        seqs: dict[str, str],
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
        constraints: list["HardConstraint"] | None = None,
    ) -> Proposal | None:
        """
        Return a candidate ``(domain_name, position, new_base)`` mutation.

        Implementations may return ``None`` to signal that no acceptable
        proposal exists this step (the caller treats this as a skip).
        """
        raise NotImplementedError

    def update(
        self,
        seqs: dict[str, str],
        objective_breakdown: dict[str, float] | None = None,
    ) -> None:
        """
        Hook called by the optimizer after each accepted move so adaptive
        policies (like defect-weighted) can recompute their internal
        scoring.  Default: no-op.
        """
        return None


# ─── random ────────────────────────────────────────────────────────────────────


@dataclass
class RandomMutationPolicy(MutationPolicy):
    """
    Uniform-random base flip on a uniform-random free position.

    The original designer behaviour.  Useful as a baseline.
    """

    max_retries: int = 4

    def propose(
        self,
        seqs: dict[str, str],
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
        constraints: list["HardConstraint"] | None = None,
    ) -> Proposal | None:
        free = [n for n, d in domains.items() if not d.fixed]
        if not free:
            return None
        for _ in range(self.max_retries):
            name = rng.choice(free)
            pos = rng.randint(0, len(seqs[name]) - 1)
            old_base = seqs[name][pos]
            bases = _bases_for(domains[name].material)
            new_base = rng.choice([b for b in bases if b != old_base])
            new_seq = seqs[name][:pos] + new_base + seqs[name][pos + 1 :]
            if constraints and not all(c.check(name, new_seq) for c in constraints):
                continue
            return name, pos, new_base
        return None


# ─── defect-weighted ───────────────────────────────────────────────────────────


@dataclass
class DefectWeightedPolicy(MutationPolicy):
    """
    Sample mutation sites with probability ∝ per-residue ensemble defect.

    The policy needs an objective-specific per-residue defect vector for
    each free domain.  Callers wire it up by supplying ``defect_fn`` — a
    callable ``(seqs) -> dict[name, np.ndarray]`` returning a length-``L``
    per-residue defect vector for each domain ``name``.  Positions with
    higher defect are flipped more often; ties fall back to uniform.

    Parameters
    ----------
    defect_fn        : callable that, given the current sequences dict,
                       returns ``{domain_name: per_residue_defect}``.
    refresh_every    : recompute the defect vector every N accepted moves
                       (cheap if the McCaskill cache hits; expensive
                       otherwise).  Default ``25``.
    epsilon          : add this floor to every weight so even
                       well-satisfied positions still receive proposals
                       occasionally (avoids local minima where defect is
                       concentrated on a single base).  Default ``0.05``.
    """

    defect_fn: Callable[[dict[str, str]], dict[str, "object"]] = field(default=lambda _: {})
    refresh_every: int = 25
    epsilon: float = 0.05
    fallback: MutationPolicy = field(default_factory=RandomMutationPolicy)

    def __post_init__(self) -> None:
        self._steps_since_refresh = 0
        self._weights_cache: dict[str, list[float]] = {}
        self._last_seqs: dict[str, str] | None = None

    def _maybe_refresh(self, seqs: dict[str, str]) -> None:
        if (
            self._last_seqs is None
            or self._steps_since_refresh >= self.refresh_every
            or set(self._weights_cache) != set(seqs)
        ):
            try:
                vectors = self.defect_fn(seqs)
            except Exception:
                vectors = {}
            self._weights_cache = {
                name: [max(float(v), 0.0) + self.epsilon for v in vec]
                for name, vec in vectors.items()
            }
            self._last_seqs = dict(seqs)
            self._steps_since_refresh = 0

    def propose(
        self,
        seqs: dict[str, str],
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
        constraints: list["HardConstraint"] | None = None,
    ) -> Proposal | None:
        self._maybe_refresh(seqs)

        free = [n for n, d in domains.items() if not d.fixed]
        if not free:
            return None

        # Pick a free domain weighted by total defect; fall back to uniform.
        domain_totals = [sum(self._weights_cache.get(n, [])) or 1.0 for n in free]
        name = rng.choices(free, weights=domain_totals, k=1)[0]
        seq = seqs[name]

        weights = self._weights_cache.get(name)
        if not weights or len(weights) != len(seq):
            return self.fallback.propose(seqs, domains, rng, constraints)

        for _ in range(4):
            pos = rng.choices(range(len(seq)), weights=weights, k=1)[0]
            old_base = seq[pos]
            bases = _bases_for(domains[name].material)
            new_base = rng.choice([b for b in bases if b != old_base])
            new_seq = seq[:pos] + new_base + seq[pos + 1 :]
            if constraints and not all(c.check(name, new_seq) for c in constraints):
                continue
            return name, pos, new_base
        return self.fallback.propose(seqs, domains, rng, constraints)

    def update(
        self,
        seqs: dict[str, str],
        objective_breakdown: dict[str, float] | None = None,
    ) -> None:
        self._steps_since_refresh += 1


# ─── constraint-aware wrapper ──────────────────────────────────────────────────


@dataclass
class ConstraintAwarePolicy(MutationPolicy):
    """
    Wrap another policy and route base-flip generation through
    :meth:`HardConstraint.propose` when at least one constraint provides
    a proposer.  Falls back to the inner policy otherwise.
    """

    inner: MutationPolicy

    def propose(
        self,
        seqs: dict[str, str],
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
        constraints: list["HardConstraint"] | None = None,
    ) -> Proposal | None:
        active = [c for c in (constraints or []) if getattr(c, "proposer", None) is not None]
        if not active:
            return self.inner.propose(seqs, domains, rng, constraints)

        free = [n for n, d in domains.items() if not d.fixed]
        if not free:
            return None
        name = rng.choice(free)
        seq = seqs[name]
        pos = rng.randint(0, len(seq) - 1)
        bases = _bases_for(domains[name].material)

        for c in active:
            proposed = c.propose(name, seq, pos, rng, bases)
            if proposed is None:
                continue
            new_base = proposed
            new_seq = seq[:pos] + new_base + seq[pos + 1 :]
            if all(cc.check(name, new_seq) for cc in (constraints or [])):
                return name, pos, new_base

        return self.inner.propose(seqs, domains, rng, constraints)

    def update(
        self,
        seqs: dict[str, str],
        objective_breakdown: dict[str, float] | None = None,
    ) -> None:
        self.inner.update(seqs, objective_breakdown)


# ─── helpers ───────────────────────────────────────────────────────────────────


def per_residue_defect_from_ensemble(
    engine: "ThermoEngine",
    strand_names: list[str],
    target_structure: str,
) -> Callable[[dict[str, str]], dict[str, "object"]]:
    """
    Build a ``defect_fn`` suitable for :class:`DefectWeightedPolicy`.

    Given an engine, a list of domain names whose sequences concatenate in
    order to form the target complex, and a target dot-bracket string, the
    returned function maps a ``{name: sequence}`` dict to ``{name:
    per_residue_defect_vector}`` slices.  The defect of position ``i`` in
    the concatenated complex is ``1 − P_correct(i)`` (Zadeh, Wolfe &
    Pierce 2011 eq. 8).
    """
    import numpy as np

    from strider.structure.dot_bracket import parse_pairs

    clean_target = target_structure.replace("&", "").replace("+", "")

    def fn(seqs: dict[str, str]) -> dict[str, "object"]:
        try:
            strands = tuple(seqs[n] for n in strand_names)
        except KeyError:
            return {}
        if sum(len(s) for s in strands) != len(clean_target):
            return {}
        try:
            probs = engine.pairs(*strands)
        except Exception:
            return {}
        target_pairs: dict[int, int] = {}
        for i, j in parse_pairs(target_structure):
            target_pairs[i] = j
            target_pairs[j] = i
        n = len(clean_target)
        per_residue = np.zeros(n)
        for i in range(n):
            if i in target_pairs:
                p_correct = float(probs[i][target_pairs[i]])
            else:
                p_correct = 1.0 - float(probs[i].sum())
            per_residue[i] = max(0.0, 1.0 - p_correct)
        # Slice back to per-domain vectors in declaration order.
        out: dict[str, "object"] = {}
        offset = 0
        for name in strand_names:
            L = len(seqs[name])
            out[name] = per_residue[offset : offset + L]
            offset += L
        return out

    return fn
