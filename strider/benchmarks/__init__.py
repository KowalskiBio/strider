"""
Benchmark suite for strider — receipts on accuracy and timing.

Three pieces of evidence the suite produces:

1. **Structure-prediction accuracy** vs. canonical hairpins from the
   primary RNA / DNA thermodynamics literature (Cheong 1990, Heus & Pardi
   1991, Antao 1991, Mathews 1999, SantaLucia 2004, Lu 2006).  Reports
   sensitivity / PPV / F-measure on each reference, plus head-to-head
   ΔG agreement with ViennaRNA when installed.
2. **TMSD kinetics** — validates :func:`strider.kinetics.tmsd.toehold_kf`
   against Zhang & Winfree (2009) Fig. 4 at 25 °C and checks that the
   Arrhenius extrapolation at 37 °C remains monotonic in toehold length.
3. **Wall-clock timing** of MFE / pfunc on a length sweep with optional
   ViennaRNA side-by-side numbers.

The runners are pure-Python, deterministic, and live in-process so the
benchmark can run inside the test suite, on CI, or from
``scripts/bench_accuracy.py``.
"""

from strider.benchmarks.accuracy import (
    StructureMetrics,
    dot_bracket_metrics,
    pair_set_metrics,
    relative_error,
    mean_abs_dG_diff,
)
from strider.benchmarks.structure_refs import (
    StructureRef, REFERENCES, get_references,
)
from strider.benchmarks.runners import (
    StructureBenchmark, StructurePerSequence,
    TMSDBenchmark, TMSDPoint,
    TimingBenchmark, TimingRow,
    run_structure_benchmark,
    run_tmsd_benchmark,
    run_timing_benchmark,
)

__all__ = [
    "StructureMetrics", "dot_bracket_metrics", "pair_set_metrics",
    "relative_error", "mean_abs_dG_diff",
    "StructureRef", "REFERENCES", "get_references",
    "StructureBenchmark", "StructurePerSequence",
    "TMSDBenchmark", "TMSDPoint",
    "TimingBenchmark", "TimingRow",
    "run_structure_benchmark", "run_tmsd_benchmark", "run_timing_benchmark",
]
