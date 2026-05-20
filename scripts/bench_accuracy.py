"""
Accuracy + timing benchmark for strider.

Produces the receipts that back the "competes with NUPACK on the canonical
working set" claim:

  1. Structure-prediction sensitivity / PPV / F-measure on canonical
     hairpins from the primary RNA / DNA thermodynamics literature
     (Cheong 1990, Heus & Pardi 1991, Antao 1991, Mathews 1999,
     SantaLucia 2004, Lu 2006).
  2. TMSD kinetics — head-to-head with the Zhang & Winfree 2009 Fig. 4
     reference table.
  3. Wall-clock MFE / pfunc timing across a sequence-length sweep, with
     ViennaRNA side-by-side numbers when ``import RNA`` succeeds.

Run::

    python scripts/bench_accuracy.py
    python scripts/bench_accuracy.py --lengths 20 50 100 200 --reps 5
    python scripts/bench_accuracy.py --no-vienna       # skip ViennaRNA comparison
    python scripts/bench_accuracy.py --section structure
"""

from __future__ import annotations

import argparse

from strider.benchmarks.runners import (
    run_structure_benchmark,
    run_timing_benchmark,
    run_tmsd_benchmark,
    _has_vienna,
)


def _format_structure(report) -> str:
    lines = ["=== Secondary-structure prediction ==="]
    n = len(report.per_sequence)
    lines.append(
        f"References:       {n} canonical hairpins (Cheong/Heus/Antao/"
        f"Mathews/SantaLucia/Lu primary literature)"
    )
    lines.append(f"Mean sensitivity: {report.mean_sensitivity:.3f}")
    lines.append(f"Mean PPV:         {report.mean_ppv:.3f}")
    lines.append(f"Mean F-measure:   {report.mean_f_measure:.3f}")
    lines.append(f"Exact match:      {report.n_exact}/{n}")
    if report.mean_abs_dG_vienna_diff is not None:
        lines.append(
            f"Mean |ΔG native − ViennaRNA|: "
            f"{report.mean_abs_dG_vienna_diff:.2f} kcal/mol"
        )
    lines.append("")
    lines.append(f"{'name':<40} {'F':>5}  {'ΔG (kcal/mol)':>14}  {'vienna':>8}")
    lines.append("-" * 76)
    for s in report.per_sequence:
        vienna = (
            f"{s.dG_vienna:6.2f}" if s.dG_vienna is not None else "      "
        )
        lines.append(
            f"{s.ref.name:<40} {s.metrics.f_measure:>5.2f}  "
            f"{s.dG_native:>14.2f}  {vienna:>8}"
        )
    return "\n".join(lines)


def _format_tmsd(report) -> str:
    lines = ["", "=== TMSD kinetics — Zhang & Winfree (2009) Fig. 4 ==="]
    lines.append(
        f"Mean relative error vs Fig. 4 lookup: {report.mean_rel_error:.1%}"
    )
    lines.append(
        f"Max relative error:                    {report.max_rel_error:.1%}"
    )
    lines.append(
        f"37 °C Arrhenius extrapolation monotonic in toehold length: "
        f"{report.arrhenius_monotonic}"
    )
    lines.append("")
    lines.append(f"{'toehold (nt)':>12}  {'kf (25 °C)':>14}  {'reference':>14}  {'rel err':>9}")
    lines.append("-" * 56)
    for p in report.points_25C:
        lines.append(
            f"{p.toehold_nt:>12}  {p.kf_predicted:>14.3e}  "
            f"{p.kf_reference:>14.3e}  {p.relative_error:>8.1%}"
        )
    return "\n".join(lines)


def _format_timing(report) -> str:
    lines = ["", "=== Timing (median of N reps, milliseconds) ==="]
    has_vienna = any(r.vienna_mfe_median_ms is not None for r in report.rows)
    if has_vienna:
        header = (
            f"{'length':>7}  {'native MFE':>11}  {'native pfunc':>13}  "
            f"{'vienna MFE':>11}  {'vienna pfunc':>13}"
        )
    else:
        header = f"{'length':>7}  {'native MFE':>11}  {'native pfunc':>13}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.rows:
        base = (
            f"{r.length:>7}  {r.mfe_median_ms:>9.1f}ms  "
            f"{r.pfunc_median_ms:>11.1f}ms"
        )
        if has_vienna:
            vmfe = (
                f"{r.vienna_mfe_median_ms:>9.1f}ms"
                if r.vienna_mfe_median_ms is not None else "        --"
            )
            vpf = (
                f"{r.vienna_pfunc_median_ms:>11.1f}ms"
                if r.vienna_pfunc_median_ms is not None else "          --"
            )
            base = f"{base}  {vmfe}  {vpf}"
        lines.append(base)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--section", choices=("all", "structure", "tmsd", "timing"),
        default="all",
        help="Which section to run (default: all).",
    )
    p.add_argument(
        "--lengths", nargs="+", type=int, default=[20, 50, 100],
        help="Sequence lengths for the timing sweep.",
    )
    p.add_argument(
        "--reps", type=int, default=3,
        help="Reps per length for the timing sweep.",
    )
    p.add_argument(
        "--material", choices=("dna", "rna"), default="dna",
        help="Material for the timing sweep.",
    )
    p.add_argument(
        "--no-vienna", action="store_true",
        help="Skip ViennaRNA head-to-head even if RNA is importable.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    include_vienna = not args.no_vienna
    if include_vienna and not _has_vienna():
        print(
            "(note: ViennaRNA not importable — running native-only; "
            "install with `pip install strider-dna[vienna]` for head-to-head.)"
        )
        include_vienna = False

    if args.section in ("all", "structure"):
        print(_format_structure(run_structure_benchmark(include_vienna=include_vienna)))

    if args.section in ("all", "tmsd"):
        print(_format_tmsd(run_tmsd_benchmark()))

    if args.section in ("all", "timing"):
        print(_format_timing(run_timing_benchmark(
            lengths=args.lengths, reps=args.reps,
            material=args.material, seed=args.seed,
            include_vienna=include_vienna,
        )))


if __name__ == "__main__":
    main()
