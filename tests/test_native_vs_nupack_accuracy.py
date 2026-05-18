"""
Native-backend accuracy benchmark against NUPACK.

These tests quantify the systematic free-energy discrepancy between strider's
native McCaskill DP and NUPACK's C++ engine.  They are *not* strict equality
checks — they pin the current accuracy envelope so regressions can be detected.

Findings (May 2026):

* Strider's native backend systematically over-stabilizes by ~0.15–0.50 kcal/mol
  relative to NUPACK on hairpins with flanking single-stranded regions.
* The discrepancy grows with stem length (≈0.05 kcal/mol per stem base-pair).
* Root cause is the **external-loop dangle / terminal-mismatch model**.  Probe
  experiments (``scratch/tm_formula_test.py``) show NUPACK applies 5'- and
  3'-dangle bonuses *additively* when both flanking bases are unpaired, whereas
  strider's ``_apply_coaxial_external`` uses a Boltzmann stacking-ensemble
  decoration (None / D5 / D3 / TM as mutually exclusive alternatives).
* Closing the gap requires a structural rewrite of the external-loop DP plus
  re-derivation of the dangle key formats from the NUPACK C++ source.  This
  is the open work flagged as "task 5" in the May 2026 roadmap.

Skipped automatically when NUPACK is unavailable.
"""

import math
import pytest

nupack = pytest.importorskip("nupack", reason="NUPACK not installed")


CASES = [
    # (sequence, max_acceptable_abs_error_kcal_per_mol)
    ("GCGCAAAAGCGC",       0.5),
    ("AGCGCAAAAGCGCA",     0.6),
    ("GCATGCATGC",         0.3),
    ("ATATATATATATATAT",   0.4),
    ("GCGCGCAAAAGCGCGC",   0.6),
]


@pytest.mark.parametrize("seq,tol", CASES)
def test_native_within_tolerance_of_nupack(seq, tol):
    """Native ΔG must be within ``tol`` kcal/mol of NUPACK ΔG."""
    from strider.thermo.engine import ThermoEngine
    eng_n = ThermoEngine(material="dna", celsius=37.0, backend="nupack")
    eng_s = ThermoEngine(material="dna", celsius=37.0, backend="native")
    gn = eng_n.pfunc(seq).free_energy
    gs = eng_s.pfunc(seq).free_energy
    err = abs(gs - gn)
    assert err < tol, (
        f"native {gs:.3f} vs NUPACK {gn:.3f} = error {gs-gn:+.3f} kcal/mol "
        f"(exceeds {tol})"
    )


def test_systematic_bias_documented():
    """Native consistently OVER-stabilizes vs NUPACK (negative bias)."""
    from strider.thermo.engine import ThermoEngine
    eng_n = ThermoEngine(material="dna", celsius=37.0, backend="nupack")
    eng_s = ThermoEngine(material="dna", celsius=37.0, backend="native")

    diffs = []
    for seq, _ in CASES:
        gn = eng_n.pfunc(seq).free_energy
        gs = eng_s.pfunc(seq).free_energy
        diffs.append(gs - gn)
    mean_diff = sum(diffs) / len(diffs)
    # The bias is currently negative (strider over-stabilizes).  If this flips
    # sign, the external-loop fix is in and the test should be updated.
    assert mean_diff < 0, f"unexpected: mean bias {mean_diff:+.3f} is non-negative"
