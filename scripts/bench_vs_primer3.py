"""
Cross-implementation check: strider's unimolecular hairpin Tm vs primer3's
``calc_hairpin`` on a panel of DNA stem-loops, at a matched buffer.

This is a *compatibility* receipt between two independent nearest-neighbor
implementations, not a deviation from ground truth -- and primer3's
``calc_hairpin`` is emphatically not a ground truth here. It is an approximate
alignment routine, not a folding dynamic program: on a typical panel it
*under-extends* the stem (leaves Watson--Crick terminal pairs unpaired) on
roughly half of sequences, where strider's Zuker MFE closes them. The script
therefore separates two sources of disagreement:

  * **structure** -- the two engines predict a different fold (different number
    of base pairs). primer3 leaving valid terminal pairs open is the common
    case; strider's fold is then the more complete one.
  * **energetics + salt** -- restricted to sequences where the two agree on the
    fold, the residual Tm reflects the nearest-neighbor parameter tables and the
    salt model. At 1 M Na+ (no salt correction) this isolates the parameters;
    the growth of the residual at lower salt isolates the salt model (strider
    uses a per-base-pair dG correction; primer3 a SantaLucia entropy term).

A hairpin Tm is additionally hypersensitive to the dH/dS bookkeeping (a
few-percent shift moves Tm by tens of degrees), so absolute values are
calibratable against experiment rather than exact; this script quantifies how
far two engines sit apart and *why*.

Conditions are matched explicitly: strider's salt model is given
``sodium_M = mv_conc/1000`` and ``magnesium_M = dv_conc/1000``; primer3 gets the
same monovalent/divalent concentrations with ``dntp_conc = 0``.

Run: python scripts/bench_vs_primer3.py [--mv 50] [--dv 0] [--random N] [--seed 0]
     # parameter-only view (no salt correction):
     python scripts/bench_vs_primer3.py --mv 1000 --random 60 --seed 7
Requires the optional dependency ``primer3-py`` (pip install primer3-py).
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys

from strider import hairpin_thermo

# Canonical DNA stem-loops: GC- and AT-rich stems, tri/tetra/penta loops, and a
# few molecular-beacon-style probes. Fixed so the receipt is reproducible.
PANEL = [
    "CGCGAAAAAGCGCG",
    "GCGCTTTTTTGCGC",
    "GGGGAAAACCCC",
    "GGGCGCGTTTTTCGCGCCC",
    "GCCGCCAAAAGGCGGC",
    "AGCTGCAAAAGCAGCT",
    "CCCAAAGGGTTTTTCCCTTTGGG",
    "GGGAAACCCTTTTTGGGTTTCCC",
    "CTTTCAACACTGTTGCAGTAA",
    "GGATCGAAAAAGATCC",
    "CACGCAGAAAACTGCGTG",
    "TGGCGACGTTTTCGTCGCCA",
]


def random_stem_loop(rng: random.Random) -> str:
    stem_len = rng.randint(5, 9)
    loop_len = rng.randint(4, 7)
    stem = "".join(rng.choice("ACGT") for _ in range(stem_len))
    rc = stem.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    loop = "".join(rng.choice("ACGT") for _ in range(loop_len))
    return stem + loop + rc


def primer3_pairs(result) -> int | None:
    """Number of base pairs in primer3's predicted hairpin (count of '/')."""
    try:
        return result.ascii_structure_lines[0].count("/")
    except Exception:  # older primer3-py without ascii structure
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mv", type=float, default=50.0, help="monovalent (Na+) mM")
    ap.add_argument("--dv", type=float, default=0.0, help="divalent (Mg2+) mM")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="append N random stem-loops to the fixed panel")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import primer3
    except ImportError:
        print("primer3-py not installed; `pip install primer3-py` to run this "
              "benchmark.", file=sys.stderr)
        return 1

    seqs = list(PANEL)
    if args.random:
        rng = random.Random(args.seed)
        seqs += [random_stem_loop(rng) for _ in range(args.random)]

    na, mg = args.mv / 1000.0, args.dv / 1000.0
    print(f"# strider hairpin Tm vs primer3 calc_hairpin  "
          f"(mv={args.mv:g} mM, dv={args.dv:g} mM)\n")
    print(f"{'sequence':26} {'sd_bp':>5} {'p3_bp':>5} {'strider':>8} "
          f"{'primer3':>8} {'Δ':>7}  note")
    print("-" * 78)

    same, diff, under = [], [], 0
    for s in seqs:
        th = hairpin_thermo(s, sodium_M=na, magnesium_M=mg)
        r = primer3.calc_hairpin(s, mv_conc=args.mv, dv_conc=args.dv,
                                 dntp_conc=0.0, output_structure=True)
        if not r.structure_found:
            continue
        p3p = primer3_pairs(r)
        d = th.tm_celsius - r.tm
        note = ""
        if p3p is not None and p3p != th.n_pairs:
            note = f"struct differ (p3 {'under' if p3p < th.n_pairs else 'over'}-extends)"
            if p3p < th.n_pairs:
                under += 1
            diff.append(d)
        elif p3p is not None:
            same.append(d)
        print(f"{s:26} {th.n_pairs:5d} "
              f"{('?' if p3p is None else p3p):>5} {th.tm_celsius:8.1f} "
              f"{r.tm:8.1f} {d:+7.1f}  {note}")

    def summarize(name, xs):
        if not xs:
            print(f"  {name:26} n=0")
            return
        a = [abs(x) for x in xs]
        print(f"  {name:26} n={len(xs):3d}  mean Δ={statistics.mean(xs):+5.2f}  "
              f"mean|Δ|={statistics.mean(a):4.2f}  max|Δ|={max(a):5.2f} °C")

    folded = len(same) + len(diff)
    print("-" * 78)
    print(f"folded by both: {folded}   primer3 under-extends stem: {under} "
          f"({100 * under / folded:.0f}% of folded)\n")
    summarize("same fold (params+salt):", same)
    summarize("different fold (heuristic):", diff)
    print("\nΔ = strider − primer3. primer3 calc_hairpin is an approximate "
          "routine, not a folding\nDP; where the folds differ it is usually "
          "primer3 leaving valid pairs open. Run at\n--mv 1000 to remove the "
          "salt correction and isolate the parameter residual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
