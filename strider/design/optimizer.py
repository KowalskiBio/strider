"""
Sequence designer using simulated annealing with optional parallel
tempering.

Optimizes free nucleotide positions in domain sequences to minimize a
composable DesignObjective while satisfying all HardConstraints.

The default proposal strategy is :class:`MutationPolicy`-based: callers
that supply a defect-weighted policy get site sampling biased toward
high-defect positions; callers that supply ``None`` get the original
uniform-random behaviour.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from strider.design.objective import DesignObjective
    from strider.design.constraints import HardConstraint
    from strider.design.policies import MutationPolicy
    from strider.thermo.engine import ThermoEngine

DNA_BASES = list("ACGT")
RNA_BASES = list("ACGU")


@dataclass
class DomainSpec:
    """
    Specification for a single nucleic acid domain.

    Attributes
    ----------
    length    : domain length in nucleotides (inferred from sequence if provided)
    sequence  : fixed sequence string; None means the optimizer is free to choose
    material  : 'dna' or 'rna'
    fixed     : True if sequence is provided and should not be mutated
    gc_band   : optional (min_gc, max_gc) early-rejection band; mutations
                that push GC content outside the band are skipped before
                evaluating the (expensive) objective.
    """
    length: int = 0
    sequence: str | None = None   # None → free to design
    material: Literal["dna", "rna"] = "dna"
    fixed: bool = False
    gc_band: tuple[float, float] | None = None

    def __post_init__(self):
        """Infer fixed and length from sequence if sequence is provided."""
        if self.sequence is not None:
            self.fixed = True
            self.length = len(self.sequence)
        if self.length == 0 and self.sequence is None:
            raise ValueError("DomainSpec requires either length or sequence")


@dataclass
class DesignResult:
    """
    Result of a sequence design run.

    Attributes
    ----------
    sequences           : domain_name → optimized sequence string
    objective_value     : final total score (lower is better)
    objective_breakdown : per-term score contributions
    n_iterations        : simulated annealing steps run
    trial_scores        : best score from each trial (useful for convergence diagnostics)
    converged           : True if final score < 1e-4
    """
    sequences: dict[str, str]
    objective_value: float
    objective_breakdown: dict[str, float]
    n_iterations: int
    trial_scores: list[float] = field(default_factory=list)
    converged: bool = False

    def __repr__(self) -> str:
        return (
            f"DesignResult(score={self.objective_value:.4f}, "
            f"iterations={self.n_iterations}, "
            f"seqs={list(self.sequences.keys())})"
        )


class SequenceDesigner:
    """
    Simulated annealing sequence optimizer with optional parallel
    tempering and defect-weighted mutation sampling.

    Parameters
    ----------
    engine  : ThermoEngine (used by objectives)
    seed    : RNG seed for reproducibility
    """

    def __init__(
        self,
        engine: "ThermoEngine | None" = None,
        seed: int | None = None,
    ) -> None:
        self.engine = engine
        self.rng = random.Random(seed)

    def design(
        self,
        domains: dict[str, "DomainSpec"],
        objective: "DesignObjective",
        hard_constraints: list["HardConstraint"] | None = None,
        n_trials: int = 10,
        max_iterations: int = 500,
        T_start: float = 1.0,
        T_end: float = 0.01,
        verbose: bool = False,
        mutation_policy: "MutationPolicy | None" = None,
        parallel_tempering: bool = False,
        n_chains: int = 4,
        swap_every: int = 20,
        initial_sequences: dict[str, str] | None = None,
    ) -> DesignResult:
        """
        Run sequence design optimization.

        ``mutation_policy`` overrides the default uniform-random base
        flip — pass a :class:`~strider.design.policies.DefectWeightedPolicy`
        for objectives dominated by ensemble defect.  When
        ``parallel_tempering`` is True, each trial runs ``n_chains``
        replicas at a geometric temperature ladder, swapping neighbouring
        chains every ``swap_every`` steps; the lowest-temperature chain is
        used for the trial score.

        ``initial_sequences`` warm-starts free domains from a given sequence
        assignment instead of a random one (the rest are still randomized);
        this is the hand-off point used by
        :class:`~strider.design.diff_designer.DifferentiableDesigner` to polish a
        gradient-descent solution with a short SA refinement.

        Returns the best result across all trials.
        """
        constraints = hard_constraints or []
        trial_scores: list[float] = []
        best: DesignResult | None = None

        for trial in range(n_trials):
            if parallel_tempering:
                result = self._run_pt_trial(
                    domains, objective, constraints,
                    max_iterations, T_start, T_end, trial, verbose,
                    mutation_policy, n_chains, swap_every, initial_sequences,
                )
            else:
                result = self._run_trial(
                    domains, objective, constraints,
                    max_iterations, T_start, T_end, trial, verbose,
                    mutation_policy, initial_sequences,
                )
            trial_scores.append(result.objective_value)
            if best is None or result.objective_value < best.objective_value:
                best = result

            if verbose:
                print(f"  Trial {trial + 1}/{n_trials}: score={result.objective_value:.4f}")

        if best is None:
            raise RuntimeError("All trials failed hard constraints.")
        best.trial_scores = trial_scores
        return best

    # ─── internals ───────────────────────────────────────────────────────────

    def _run_trial(
        self,
        domains: dict[str, "DomainSpec"],
        objective: "DesignObjective",
        constraints: list["HardConstraint"],
        max_iterations: int,
        T_start: float,
        T_end: float,
        trial_seed: int,
        verbose: bool,
        mutation_policy: "MutationPolicy | None",
        initial_sequences: dict[str, str] | None = None,
    ) -> DesignResult:
        """Single-chain SA trial returning the best DesignResult found."""
        from strider.design.policies import RandomMutationPolicy
        rng = random.Random(self.rng.randint(0, 2**31) + trial_seed)
        policy = mutation_policy or RandomMutationPolicy()

        seqs = self._initialize(domains, rng, constraints, initial_sequences)
        current_score = objective(seqs)
        best_seqs = dict(seqs)
        best_score = current_score

        log_T_start = math.log(T_start)
        log_T_end = math.log(T_end)

        for it in range(max_iterations):
            T = math.exp(
                log_T_start + (log_T_end - log_T_start) * it / max(max_iterations - 1, 1)
            )

            proposal = policy.propose(seqs, domains, rng, constraints)
            if proposal is None:
                continue
            name, pos, new_base = proposal

            new_seq = seqs[name][:pos] + new_base + seqs[name][pos + 1 :]
            if not _passes_early_checks(name, new_seq, domains):
                continue

            new_seqs = dict(seqs)
            new_seqs[name] = new_seq

            new_score = objective(new_seqs)
            delta = new_score - current_score

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                seqs = new_seqs
                current_score = new_score
                policy.update(seqs)
                if current_score < best_score:
                    best_seqs = dict(seqs)
                    best_score = current_score

        breakdown = objective.evaluate_breakdown(best_seqs)
        return DesignResult(
            sequences=best_seqs,
            objective_value=best_score,
            objective_breakdown=breakdown,
            n_iterations=max_iterations,
            converged=(best_score < 1e-4),
        )

    def _run_pt_trial(
        self,
        domains: dict[str, "DomainSpec"],
        objective: "DesignObjective",
        constraints: list["HardConstraint"],
        max_iterations: int,
        T_start: float,
        T_end: float,
        trial_seed: int,
        verbose: bool,
        mutation_policy: "MutationPolicy | None",
        n_chains: int,
        swap_every: int,
        initial_sequences: dict[str, str] | None = None,
    ) -> DesignResult:
        """
        Parallel-tempering trial.  Each chain runs at a fixed temperature
        on a geometric ladder from T_end (chain 0, "cold") to T_start
        (chain n_chains-1, "hot").  Every ``swap_every`` steps we attempt
        a Metropolis swap between each adjacent chain pair.  Best score
        seen across all chains is returned.
        """
        from strider.design.policies import RandomMutationPolicy
        rng = random.Random(self.rng.randint(0, 2**31) + trial_seed)
        n_chains = max(2, n_chains)

        # Geometric temperature ladder.
        if n_chains == 1:
            temps = [T_end]
        else:
            ratio = (T_start / T_end) ** (1.0 / (n_chains - 1))
            temps = [T_end * ratio**i for i in range(n_chains)]

        # One independent policy + state per chain.
        chains_seqs: list[dict[str, str]] = []
        chains_scores: list[float] = []
        chains_policies: list[MutationPolicy] = []
        for _ in range(n_chains):
            s = self._initialize(domains, rng, constraints, initial_sequences)
            chains_seqs.append(s)
            chains_scores.append(objective(s))
            chains_policies.append(mutation_policy or RandomMutationPolicy())

        best_seqs = dict(chains_seqs[0])
        best_score = chains_scores[0]

        steps_per_chain = max(1, max_iterations // n_chains)
        for it in range(steps_per_chain):
            for c in range(n_chains):
                T = temps[c]
                policy = chains_policies[c]
                proposal = policy.propose(chains_seqs[c], domains, rng, constraints)
                if proposal is None:
                    continue
                name, pos, new_base = proposal
                new_seq = chains_seqs[c][name][:pos] + new_base + chains_seqs[c][name][pos + 1 :]
                if not _passes_early_checks(name, new_seq, domains):
                    continue
                new_seqs = dict(chains_seqs[c])
                new_seqs[name] = new_seq
                new_score = objective(new_seqs)
                delta = new_score - chains_scores[c]
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                    chains_seqs[c] = new_seqs
                    chains_scores[c] = new_score
                    policy.update(new_seqs)
                    if new_score < best_score:
                        best_seqs = dict(new_seqs)
                        best_score = new_score

            if swap_every > 0 and (it + 1) % swap_every == 0:
                # Attempt swaps between every adjacent pair (cold→hot).
                for c in range(n_chains - 1):
                    e1, e2 = chains_scores[c], chains_scores[c + 1]
                    t1, t2 = temps[c], temps[c + 1]
                    swap_log = (e1 - e2) * (1.0 / max(t1, 1e-10) - 1.0 / max(t2, 1e-10))
                    if swap_log >= 0 or rng.random() < math.exp(swap_log):
                        chains_seqs[c], chains_seqs[c + 1] = chains_seqs[c + 1], chains_seqs[c]
                        chains_scores[c], chains_scores[c + 1] = chains_scores[c + 1], chains_scores[c]

        breakdown = objective.evaluate_breakdown(best_seqs)
        return DesignResult(
            sequences=best_seqs,
            objective_value=best_score,
            objective_breakdown=breakdown,
            n_iterations=steps_per_chain * n_chains,
            converged=(best_score < 1e-4),
        )

    def _initialize(
        self,
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
        constraints: list["HardConstraint"] | None = None,
        initial_sequences: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Generate starting sequences for free domains; use fixed sequences
        as-is.  A free domain present in ``initial_sequences`` is warm-started
        from that sequence (the gradient-designer hand-off); otherwise the
        optimizer retries up to 64 times to land in a hard-constraint-feasible
        random start before giving up and emitting an unconstrained random
        sequence (the SA loop will re-reject it).
        """
        seqs: dict[str, str] = {}
        cs = constraints or []
        warm = initial_sequences or {}
        for name, spec in domains.items():
            if spec.sequence is not None:
                seqs[name] = spec.sequence.upper().replace("U", "T")
                continue
            if name in warm and warm[name]:
                seqs[name] = warm[name].upper().replace("U", "T")
                continue
            bases = _bases_for(spec.material)
            attempt = ""
            for _ in range(64):
                attempt = "".join(rng.choice(bases) for _ in range(spec.length))
                if all(c.check(name, attempt) for c in cs):
                    break
            seqs[name] = attempt
        return seqs


def _bases_for(material: str) -> list[str]:
    """Return the alphabet (ACGU for RNA, ACGT for DNA) as a list."""
    return RNA_BASES if material == "rna" else DNA_BASES


def _passes_early_checks(name: str, seq: str, domains: dict[str, "DomainSpec"]) -> bool:
    """
    Cheap pre-objective filters: skip the expensive objective if the
    domain has declared a ``gc_band`` and the candidate sequence leaves
    it.  Other early-rejection rules (length floors, reachability) can
    grow into this hook later.
    """
    spec = domains.get(name)
    if spec is None or spec.gc_band is None or not seq:
        return True
    gc = sum(1 for b in seq.upper() if b in "GC") / len(seq)
    lo, hi = spec.gc_band
    return lo <= gc <= hi
