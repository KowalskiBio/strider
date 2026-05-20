"""
Defect-based sequence-design benchmark.

Runs the canonical hairpin / duplex / 3-arm-junction tasks defined in
:mod:`strider.design.benchmarks` using the new defect-weighted
:class:`~strider.design.policies.MutationPolicy` and parallel
tempering, then prints a wall-clock + final-defect table.

Run::

    python scripts/bench_design.py
    python scripts/bench_design.py --iterations 5000 --trials 5
"""

from __future__ import annotations

import argparse

from strider.design.benchmarks import run_all, standard_tasks
from strider.thermo.engine import ThermoEngine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=2000,
                   help="max SA iterations per trial (default 2000)")
    p.add_argument("--trials", type=int, default=3,
                   help="number of independent trials per task (default 3)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument("--no-pt", action="store_true",
                   help="disable parallel tempering")
    p.add_argument("--material", choices=("dna", "rna"), default="dna",
                   help="material for the engine (default dna)")
    p.add_argument("--list", action="store_true",
                   help="list the available tasks and exit")
    args = p.parse_args()

    if args.list:
        for t in standard_tasks():
            print(f"{t.name:<14}  {t.material}  target={t.target_structure}  "
                  f"floor≈{t.expected_floor:.2f}  {t.description}")
        return

    engine = ThermoEngine(material=args.material)
    print(f"engine = {engine}\n")
    results = run_all(
        engine,
        n_trials=args.trials,
        max_iterations=args.iterations,
        seed=args.seed,
        parallel_tempering=not args.no_pt,
    )

    print(f"{'task':<14}  {'defect':>8}  {'floor':>6}  {'iters':>6}  {'wall':>8}  sequences")
    print("-" * 80)
    for r in results:
        print(r.summary())


if __name__ == "__main__":
    main()
