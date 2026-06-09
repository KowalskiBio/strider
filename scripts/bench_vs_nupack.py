"""
Head-to-head receipt: strider native engine vs. NUPACK 4.

Computes, on a shared set of random sequences across a length sweep, both the
**energetic agreement** (mean / max |ΔΔG| of the ensemble free energy from the
partition function) and the **wall-clock pfunc speed** of each engine.  This is
the energetics counterpart to ``bench_accuracy.py`` (which reports structural
F-measure on canonical hairpins): it answers "does the pure-Python native
McCaskill DP reproduce the reference C kernel's pfunc, and at what speed cost?"

NUPACK is closed-source and must be installed separately; this script is meant
to be run from an environment that has *both* importable, e.g.::

    PYTHONPATH=/path/to/strider \
        nupack_env310/bin/python scripts/bench_vs_nupack.py --reps 5

Output is a plain table plus a one-line JSON receipt (``--json out.json``) so the
numbers can be frozen and cited rather than re-quoted by hand.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Iterable, Literal

from strider.thermo.engine import ThermoEngine

try:
    from nupack import Model, pfunc as nupack_pfunc  # type: ignore

    _HAS_NUPACK = True
except ImportError:  # pragma: no cover - depends on external install
    _HAS_NUPACK = False


def random_sequence(n: int, material: Literal["dna", "rna"], rng: random.Random) -> str:
    alphabet = "ACGU" if material == "rna" else "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _median_ms(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run(
    lengths: Iterable[int],
    n_seqs: int,
    reps: int,
    material: Literal["dna", "rna"],
    celsius: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    engine = ThermoEngine(material=material, celsius=celsius, backend="native")
    nupack_model = Model(material=material, celsius=celsius) if _HAS_NUPACK else None

    rows = []
    for n in lengths:
        seqs = [random_sequence(n, material, rng) for _ in range(n_seqs)]

        ddg = []
        native_ms = []
        nupack_ms = []
        for s in seqs:
            dg_native = engine.pfunc(s).free_energy
            native_ms.append(_median_ms(lambda s=s: engine.pfunc(s), reps))

            if _HAS_NUPACK:
                # nupack.pfunc returns (partition_function, free_energy)
                dg_nupack = float(nupack_pfunc([s], model=nupack_model)[1])
                nupack_ms.append(
                    _median_ms(lambda s=s: nupack_pfunc([s], model=nupack_model), reps)
                )
                ddg.append(abs(dg_native - dg_nupack))

        row = {
            "length": n,
            "n_seqs": n_seqs,
            "native_ms_per_seq": statistics.median(native_ms),
        }
        if _HAS_NUPACK:
            row["nupack_ms_per_seq"] = statistics.median(nupack_ms)
            row["speed_ratio_native_over_nupack"] = (
                statistics.median(native_ms) / statistics.median(nupack_ms)
            )
            row["mean_abs_ddg"] = statistics.mean(ddg)
            row["max_abs_ddg"] = max(ddg)
        rows.append(row)

    return {
        "material": material,
        "celsius": celsius,
        "seed": seed,
        "reps": reps,
        "nupack_available": _HAS_NUPACK,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="+", default=[20, 50, 100, 150])
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--material", choices=["rna", "dna"], default="rna")
    ap.add_argument("--celsius", type=float, default=37.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not _HAS_NUPACK:
        print("WARNING: nupack not importable - reporting native-only numbers.\n")

    result = run(
        args.lengths, args.n_seqs, args.reps, args.material, args.celsius, args.seed
    )

    print(f"=== strider native vs NUPACK 4 (pfunc, {args.material} @ {args.celsius} C) ===")
    print(f"sequences/length: {args.n_seqs}   reps: {args.reps}   seed: {args.seed}\n")
    if _HAS_NUPACK:
        hdr = f"{'len':>5} {'native ms':>11} {'nupack ms':>11} {'ratio':>8} {'mean|ddG|':>10} {'max|ddG|':>9}"
        print(hdr)
        print("-" * len(hdr))
        for r in result["rows"]:
            print(
                f"{r['length']:>5} {r['native_ms_per_seq']:>11.2f} "
                f"{r['nupack_ms_per_seq']:>11.2f} "
                f"{r['speed_ratio_native_over_nupack']:>7.1f}x "
                f"{r['mean_abs_ddg']:>10.3f} {r['max_abs_ddg']:>9.3f}"
            )
    else:
        print(f"{'len':>5} {'native ms':>11}")
        print("-" * 17)
        for r in result["rows"]:
            print(f"{r['length']:>5} {r['native_ms_per_seq']:>11.2f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nReceipt written to {args.json}")


if __name__ == "__main__":
    main()
