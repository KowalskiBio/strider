"""
Native-backend accuracy benchmark against NUPACK.

At physiological salt (Na⁺=0.137 M, Mg²⁺=0.01 M, the engine default), strider's
native McCaskill DP now matches NUPACK pfunc to within ~0.05 kcal/mol for the
hairpin cases below.  The remaining residual (~0.03 kcal/mol on GCATGCATGC) is
unrelated to salt and not yet characterized.

Historical note: an earlier diagnosis attributed a ~0.15–0.50 kcal/mol
over-stabilization to the external-loop dangle / terminal_mismatch model.
That was wrong — ``STK_TM_DELTA`` is algebraically ``STK_D5_DELTA · STK_D3_DELTA``,
so the Boltzmann decoration form and NUPACK's additive form are identical.
The actual cause was the missing per-base-pair salt correction in the native
ensemble DP, now applied via :func:`strider.thermo.salt.dg_per_bp_salt`
(empirical fit: ΔG_per_bp = −0.114·ln([Na⁺] + 3.4·√[Mg²⁺]), kcal/mol).

Skipped automatically when NUPACK is unavailable.
"""

import math
import pytest

nupack = pytest.importorskip("nupack", reason="NUPACK not installed")


CASES = [
    # (sequence, max_acceptable_abs_error_kcal_per_mol)
    ("GCGCAAAAGCGC",       0.05),
    ("AGCGCAAAAGCGCA",     0.05),
    ("GCATGCATGC",         0.05),
    ("ATATATATATATATAT",   0.05),
    ("GCGCGCAAAAGCGCGC",   0.05),
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


def test_mean_bias_near_zero():
    """Mean native−NUPACK bias must be near zero (was -0.33 before salt fix)."""
    from strider.thermo.engine import ThermoEngine
    eng_n = ThermoEngine(material="dna", celsius=37.0, backend="nupack")
    eng_s = ThermoEngine(material="dna", celsius=37.0, backend="native")

    diffs = []
    for seq, _ in CASES:
        gn = eng_n.pfunc(seq).free_energy
        gs = eng_s.pfunc(seq).free_energy
        diffs.append(gs - gn)
    mean_diff = sum(diffs) / len(diffs)
    assert abs(mean_diff) < 0.02, f"mean bias {mean_diff:+.3f} kcal/mol exceeds 0.02"
