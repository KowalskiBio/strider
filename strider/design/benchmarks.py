"""
Canonical defect-based design benchmark tasks.

The tasks here are simplified versions of the standard hairpin / duplex /
3-arm-junction problems used in the literature to compare ensemble-defect
sequence-design tools (see Zadeh, Wolfe & Pierce 2011, J. Comput. Chem.
32:439-452 and Wolfe & Pierce 2015, J. Comput. Chem. 36:255-269).

Each :class:`BenchmarkTask` packages the domains, the on-target
:class:`~strider.design.assay.Assembly`, and the target dot-bracket so
that the runner can compose a :class:`~strider.design.objective.DesignObjective`
on the fly.  Off-target species are added only where they are essential
(e.g. avoiding homodimers for self-complementary strands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.design.optimizer import DomainSpec


@dataclass(frozen=True)
class BenchmarkTask:
    """A single design benchmark task."""

    name: str
    material: str
    domains: dict[str, int]            # name → length
    on_target_strands: list[str]       # ordered list of domain names forming the complex
    target_structure: str              # dot-bracket (no separators)
    description: str = ""
    expected_floor: float = 0.10       # rough best-case normalised defect

    def domain_specs(self) -> "dict[str, DomainSpec]":
        from strider.design.optimizer import DomainSpec
        return {
            n: DomainSpec(length=L, material=self.material)
            for n, L in self.domains.items()
        }


def standard_tasks() -> list[BenchmarkTask]:
    """Return the suite of canonical benchmark tasks bundled with strider."""
    return [
        BenchmarkTask(
            name="hairpin-12",
            material="dna",
            domains={"H": 12},
            on_target_strands=["H"],
            target_structure="((((....))))",
            description="12-nt hairpin with a 4-bp stem and a 4-nt loop.",
            expected_floor=0.06,
        ),
        BenchmarkTask(
            name="hairpin-20",
            material="dna",
            domains={"H": 20},
            on_target_strands=["H"],
            target_structure="((((((........))))))",
            description="20-nt hairpin with a 6-bp stem and an 8-nt loop.",
            expected_floor=0.04,
        ),
        BenchmarkTask(
            name="duplex-12",
            material="dna",
            domains={"A": 12, "B": 12},
            on_target_strands=["A", "B"],
            target_structure="(" * 12 + ")" * 12,
            description="12-nt blunt-end duplex (two independent strands).",
            expected_floor=0.10,
        ),
    ]


@dataclass
class BenchmarkResult:
    """Outcome of running a single task."""

    task: BenchmarkTask
    final_defect: float
    final_sequences: dict[str, str]
    iterations: int
    wall_time: float
    trial_scores: list[float] = field(default_factory=list)

    def summary(self) -> str:
        seqs = ", ".join(f"{k}={v}" for k, v in self.final_sequences.items())
        return (
            f"{self.task.name:<14}  defect={self.final_defect:.4f}  "
            f"floor={self.task.expected_floor:.2f}  "
            f"iters={self.iterations}  wall={self.wall_time:.1f}s  [{seqs}]"
        )


def run_task(
    task: BenchmarkTask,
    engine,
    *,
    n_trials: int = 3,
    max_iterations: int = 2000,
    seed: int = 0,
    parallel_tempering: bool = True,
) -> BenchmarkResult:
    """Run a benchmark task on a configured :class:`ThermoEngine`."""
    import time

    from strider.design.objective import DesignObjective
    from strider.design.optimizer import SequenceDesigner
    from strider.design.policies import (
        DefectWeightedPolicy, per_residue_defect_from_ensemble,
    )

    obj = DesignObjective.ensemble_defect(
        engine, task.on_target_strands, task.target_structure, normalize=True,
    )
    defect_fn = per_residue_defect_from_ensemble(
        engine, task.on_target_strands, task.target_structure,
    )
    policy = DefectWeightedPolicy(defect_fn=defect_fn)
    designer = SequenceDesigner(engine, seed=seed)

    t0 = time.perf_counter()
    result = designer.design(
        domains=task.domain_specs(),
        objective=obj,
        n_trials=n_trials,
        max_iterations=max_iterations,
        mutation_policy=policy,
        parallel_tempering=parallel_tempering,
    )
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        task=task,
        final_defect=result.objective_value,
        final_sequences=result.sequences,
        iterations=result.n_iterations,
        wall_time=wall,
        trial_scores=result.trial_scores,
    )


def run_all(
    engine=None,
    *,
    n_trials: int = 3,
    max_iterations: int = 2000,
    seed: int = 0,
    parallel_tempering: bool = True,
) -> list[BenchmarkResult]:
    """Run every task in :func:`standard_tasks` and return their results."""
    if engine is None:
        from strider.thermo.engine import ThermoEngine
        engine = ThermoEngine(material="dna")

    return [
        run_task(
            task, engine,
            n_trials=n_trials, max_iterations=max_iterations,
            seed=seed, parallel_tempering=parallel_tempering,
        )
        for task in standard_tasks()
    ]
