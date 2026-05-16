"""
Sequence designer using simulated annealing.

Optimizes free nucleotide positions in domain sequences to minimize a
composable DesignObjective while satisfying all HardConstraints.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from strider.design.objective import DesignObjective
    from strider.design.constraints import HardConstraint
    from strider.thermo.engine import ThermoEngine

DNA_BASES = list("ACGT")
RNA_BASES = list("ACGU")


@dataclass
class DomainSpec:
    """
    Specification for a single nucleic acid domain.

    Attributes
    ----------
    length   : domain length in nucleotides (inferred from sequence if provided)
    sequence : fixed sequence string; None means the optimizer is free to choose
    material : 'dna' or 'rna'
    fixed    : True if sequence is provided and should not be mutated
    """
    length: int = 0
    sequence: str | None = None   # None → free to design
    material: Literal["dna", "rna"] = "dna"
    fixed: bool = False

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
    Simulated annealing sequence optimizer.

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
    ) -> DesignResult:
        """
        Run sequence design optimization.

        Returns the best result across all trials.
        """
        constraints = hard_constraints or []
        trial_scores: list[float] = []
        best: DesignResult | None = None

        for trial in range(n_trials):
            result = self._run_trial(
                domains, objective, constraints,
                max_iterations, T_start, T_end, trial, verbose,
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
    ) -> DesignResult:
        """Run a single simulated annealing trial and return the best DesignResult found."""
        rng = random.Random(self.rng.randint(0, 2**31) + trial_seed)

        # Initialize sequences
        seqs = self._initialize(domains, rng)

        current_score = objective(seqs)
        best_seqs = dict(seqs)
        best_score = current_score

        log_T_start = math.log(T_start)
        log_T_end = math.log(T_end)

        for it in range(max_iterations):
            T = math.exp(
                log_T_start + (log_T_end - log_T_start) * it / max(max_iterations - 1, 1)
            )

            # Propose mutation: flip one base in one free domain
            name = rng.choice([n for n, d in domains.items() if not d.fixed])
            pos = rng.randint(0, len(seqs[name]) - 1)
            old_base = seqs[name][pos]
            bases = _bases_for(domains[name].material)
            new_base = rng.choice([b for b in bases if b != old_base])

            new_seq = seqs[name][:pos] + new_base + seqs[name][pos + 1:]
            new_seqs = dict(seqs)
            new_seqs[name] = new_seq

            # Check hard constraints
            if not all(c.check(name, new_seq) for c in constraints):
                continue

            new_score = objective(new_seqs)
            delta = new_score - current_score

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                seqs = new_seqs
                current_score = new_score
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

    def _initialize(
        self,
        domains: dict[str, "DomainSpec"],
        rng: random.Random,
    ) -> dict[str, str]:
        """Generate random starting sequences for free domains; use fixed sequences as-is."""
        seqs: dict[str, str] = {}
        for name, spec in domains.items():
            if spec.sequence is not None:
                seqs[name] = spec.sequence.upper().replace("U", "T")
            else:
                bases = _bases_for(spec.material)
                seqs[name] = "".join(rng.choice(bases) for _ in range(spec.length))
        return seqs


def _bases_for(material: str) -> list[str]:
    """Return the alphabet (ACGU for RNA, ACGT for DNA) as a list."""
    return RNA_BASES if material == "rna" else DNA_BASES
