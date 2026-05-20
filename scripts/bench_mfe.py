"""
Native MFE / partition-function benchmark.

Runs strider's native engine across a range of sequence lengths and prints a
table of wall-clock time per call.  Two cohorts:

    - MFE   (Zuker DP, ``strider.structure.mfe.fold_mfe``)
    - pfunc (McCaskill DP, ``strider.thermo.engine.ThermoEngine.pfunc``)

No external tools required.  If ViennaRNA (``import RNA``) is installed, the
script additionally prints a side-by-side energy comparison so accuracy
deviation can be eyeballed.

Run::

    python scripts/bench_mfe.py
    python scripts/bench_mfe.py --reps 5 --max-length 200
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Iterable

from strider.structure.mfe import fold_mfe
from strider.thermo.engine import ThermoEngine

try:
    import RNA  # ViennaRNA, optional
    _HAS_VIENNA = True
except ImportError:
    _HAS_VIENNA = False


def random_sequence(n: int, material: str, rng: random.Random) -> str:
    """Random sequence of length ``n`` over the chosen alphabet."""
    alphabet = "ACGU" if material == "rna" else "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(n))


def bench_mfe(lengths: Iterable[int], reps: int, material: str, seed: int) -> None:
    """Time native MFE across a length sweep; print median ms per call."""
    rng = random.Random(seed)
    print(f"\n=== MFE bench (material={material}, reps={reps}) ===")
    header = f"{'length':>6}  {'median ms':>10}  {'min ms':>8}  {'max ms':>8}  {'ΔG (kcal/mol)':>14}"
    print(header)
    print("-" * len(header))
    for n in lengths:
        times: list[float] = []
        last_e = 0.0
        for _ in range(reps):
            seq = random_sequence(n, material, rng)
            t0 = time.perf_counter()
            _, energy, _ = fold_mfe(seq, material=material)
            times.append((time.perf_counter() - t0) * 1000)
            last_e = energy
        print(
            f"{n:>6}  {statistics.median(times):>10.2f}  "
            f"{min(times):>8.2f}  {max(times):>8.2f}  {last_e:>14.2f}"
        )


def bench_pfunc(lengths: Iterable[int], reps: int, material: str, seed: int) -> None:
    """Time native partition function across a length sweep."""
    rng = random.Random(seed)
    engine = ThermoEngine(material=material, celsius=37.0, sodium=1.0, magnesium=0.0)
    print(f"\n=== pfunc bench (material={material}, backend={engine.backend_name}, reps={reps}) ===")
    header = f"{'length':>6}  {'median ms':>10}  {'min ms':>8}  {'max ms':>8}  {'ΔG (kcal/mol)':>14}"
    print(header)
    print("-" * len(header))
    for n in lengths:
        times: list[float] = []
        last_e = 0.0
        for _ in range(reps):
            seq = random_sequence(n, material, rng)
            t0 = time.perf_counter()
            res = engine.pfunc(seq)
            times.append((time.perf_counter() - t0) * 1000)
            last_e = res.free_energy
        print(
            f"{n:>6}  {statistics.median(times):>10.2f}  "
            f"{min(times):>8.2f}  {max(times):>8.2f}  {last_e:>14.2f}"
        )


def cross_check_vienna(lengths: Iterable[int], reps: int, material: str, seed: int) -> None:
    """If ViennaRNA is available, compare native MFE energies head-to-head."""
    if not _HAS_VIENNA or material != "rna":
        return
    rng = random.Random(seed)
    print(f"\n=== Cross-check vs ViennaRNA (material=rna, reps={reps}) ===")
    header = f"{'length':>6}  {'strider ΔG':>11}  {'Vienna ΔG':>11}  {'Δ kcal/mol':>11}"
    print(header)
    print("-" * len(header))
    for n in lengths:
        diffs: list[float] = []
        for _ in range(reps):
            seq = random_sequence(n, "rna", rng)
            _, e_strider, _ = fold_mfe(seq, material="rna")
            _, e_vienna = RNA.fold(seq)
            diffs.append(e_strider - e_vienna)
        mean_strider = statistics.mean(
            [fold_mfe(random_sequence(n, "rna", random.Random(seed + i)), material="rna")[1]
             for i in range(reps)]
        )
        print(
            f"{n:>6}  {mean_strider:>11.2f}  {'see runs':>11}  "
            f"median Δ = {statistics.median(diffs):+.2f}, max |Δ| = {max(abs(d) for d in diffs):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=120,
                        help="largest sequence length to benchmark (default 120)")
    parser.add_argument("--reps", type=int, default=3,
                        help="repetitions per length (default 3)")
    parser.add_argument("--material", choices=["dna", "rna"], default="dna")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--skip-pfunc", action="store_true",
                        help="skip the slower pfunc cohort")
    args = parser.parse_args()

    lengths = [n for n in (20, 40, 60, 80, 100, 150, 200) if n <= args.max_length]

    bench_mfe(lengths, args.reps, args.material, args.seed)
    if not args.skip_pfunc:
        bench_pfunc(lengths, args.reps, args.material, args.seed)
    cross_check_vienna(lengths, args.reps, args.material, args.seed)


if __name__ == "__main__":
    main()
