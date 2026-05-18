"""
Round-trip validation: strider (+ mantis) vs. NUPACK.

These tests verify that:

  1. ``solve_equilibrium`` agrees with ``nupack.tube_analysis`` to within
     numerical tolerance when given the same partition functions.
  2. ``strider.equilibrium_from_engine`` (using the NUPACK backend for ΔG
     calculation) reproduces NUPACK's tube concentrations on a real DNA
     mixture.
  3. The long-time limit of a mantis ODE simulation (rates derived from
     detailed balance against the same ΔΔG values) approaches the
     thermodynamic equilibrium computed in test 2 — i.e., kinetics → thermo.

Skipped automatically when nupack or mantis are not importable.  Run under
``nupack_env`` to exercise the full suite.
"""

import math
import pytest

nupack = pytest.importorskip("nupack", reason="NUPACK not installed in this Python")


R = 1.987e-3   # kcal / (mol · K)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _nupack_tube_concentrations(
    sequences: dict[str, str],
    totals: dict[str, float],
    *,
    max_size: int = 2,
    celsius: float = 37.0,
    sodium: float = 0.137,
    magnesium: float = 0.01,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (complex_concentrations_M, complex_pfunc_dG_kcal_per_mol)."""
    strands = {n: nupack.Strand(seq, name=n) for n, seq in sequences.items()}
    tube = nupack.Tube(
        {strands[n]: totals[n] for n in sequences},
        complexes=nupack.SetSpec(max_size=max_size),
        name="t",
    )
    model = nupack.Model(
        material="dna", celsius=celsius, sodium=sodium, magnesium=magnesium,
    )
    res = nupack.tube_analysis(tubes=[tube], model=model, compute=["pfunc"])
    tube_res = list(res.tubes.values())[0]
    concs: dict[str, float] = {}
    dgs: dict[str, float] = {}
    for cx, c in tube_res.complex_concentrations.items():
        # NUPACK names complexes like '(A+B)' — strip parens, swap '+' for '_'
        name = cx.name.strip("()").replace("+", "_")
        concs[name] = float(c)
        dgs[name] = float(res.complexes[cx].free_energy)
    return concs, dgs


def _strider_complex_name(strand_list: list[str]) -> str:
    """Match NUPACK's naming convention: sorted strand names joined with '_'."""
    return "_".join(sorted(strand_list))


# ─── tests ────────────────────────────────────────────────────────────────────

class TestSolverAgainstNUPACK:
    """Stand-alone solver test: feed NUPACK pfunc values into strider's solver."""

    @pytest.mark.parametrize(
        "totals",
        [
            {"A": 1e-7, "B": 1e-7},
            {"A": 5e-8, "B": 2e-7},
            {"A": 1e-6, "B": 1e-9},  # large stoichiometric excess
        ],
    )
    def test_two_strand_mix_matches_nupack(self, totals):
        from strider import solve_equilibrium
        from strider.equilibrium import water_molarity
        seqs = {"A": "GCGCGCAAAA", "B": "TTTTGCGCGC"}

        # 1. Get NUPACK's per-complex pfunc + equilibrium concentrations
        np_conc, np_dg = _nupack_tube_concentrations(seqs, totals)

        # 2. Feed exact same pfunc values into strider's solver, telling it
        #    that NUPACK uses water-molarity standard state.
        complexes = {
            "A":   (["A"],          np_dg.get("A", 0.0)),
            "B":   (["B"],          np_dg.get("B", 0.0)),
            "A_A": (["A", "A"],     np_dg["A_A"]),
            "A_B": (["A", "B"],     np_dg["A_B"]),
            "B_B": (["B", "B"],     np_dg["B_B"]),
        }
        res = solve_equilibrium(
            complexes, totals, celsius=37.0,
            standard_state_M=water_molarity(37.0),
        )
        assert res.converged

        # NUPACK's reported pfunc already includes the σ correction for
        # homomeric complexes (Q_displayed is the species-level Q), so the
        # solver doesn't reapply it.  All five complex concentrations match
        # NUPACK to ~2 %.
        for name in ("A_B", "A", "B", "A_A", "B_B"):
            ours = res.concentrations[name]
            theirs = np_conc[name]
            rel = abs(ours - theirs) / max(theirs, 1e-15)
            assert rel < 2e-2, (
                f"{name}: strider {ours:.3e} vs NUPACK {theirs:.3e} (rel {rel:.2%})"
            )

    def test_three_strand_mix_matches_nupack(self):
        """Verify k=3 complex enumeration and σ correction against NUPACK."""
        from strider import solve_equilibrium
        from strider.equilibrium import water_molarity
        seqs = {"A": "GCGCGCAAAA", "B": "TTTTGCGCGC", "C": "GCATATGC"}
        totals = {"A": 1e-7, "B": 1e-7, "C": 1e-7}

        np_conc, np_dg = _nupack_tube_concentrations(seqs, totals, max_size=3)

        # Build complex dict from NUPACK enumeration (uses NUPACK's exact ΔG).
        from itertools import combinations_with_replacement
        complexes = {}
        for k in range(1, 4):
            for combo in combinations_with_replacement(["A", "B", "C"], k):
                cname = "_".join(combo)
                if cname in np_dg:
                    complexes[cname] = (list(combo), np_dg[cname])

        res = solve_equilibrium(
            complexes, totals, celsius=37.0,
            standard_state_M=water_molarity(37.0),
        )
        assert res.converged

        # Compare every complex we enumerated.  NUPACK additionally enumerates
        # distinct cyclic orderings of trimers (e.g. C_A_B vs C_B_A); our
        # combinations_with_replacement only emits the lexicographic
        # representative.  Those are dominated by the monomer/dimer terms in
        # this regime (< 10 fM) so the comparison is meaningful.
        for name in complexes:
            expected = np_conc.get(name, 0.0)
            if expected < 1e-12:
                continue
            ours = res.concentrations.get(name, 0.0)
            rel = abs(ours - expected) / max(expected, 1e-15)
            assert rel < 5e-2, (
                f"{name}: strider {ours:.3e} vs NUPACK {expected:.3e} "
                f"(rel {rel:.2%})"
            )


class TestEngineAgainstNUPACK:
    """Round-trip: use strider's NUPACK backend for ΔG, solve, compare to NUPACK."""

    def test_equilibrium_from_engine_matches_nupack(self):
        from strider import equilibrium_from_engine
        from strider.thermo.engine import ThermoEngine

        seqs = {"A": "GCGCGCAAAA", "B": "TTTTGCGCGC"}
        totals = {"A": 1e-7, "B": 1e-7}

        np_conc, _ = _nupack_tube_concentrations(seqs, totals)

        engine = ThermoEngine(
            material="dna", celsius=37.0, sodium=0.137, magnesium=0.01,
            backend="nupack",
        )
        ours = equilibrium_from_engine(engine, seqs, totals, max_size=2)
        assert ours.converged

        for name in ("A_B", "A", "B"):
            o = ours.concentrations[name]
            n = np_conc[name]
            rel = abs(o - n) / max(n, 1e-15)
            assert rel < 5e-2, f"{name}: strider {o:.3e} vs NUPACK {n:.3e} ({rel:.2%})"


class TestKineticsToThermoConsistency:
    """Mantis ODE long-time limit should approach the thermodynamic equilibrium."""

    def test_cha_steady_state_approaches_equilibrium(self):
        mantis = pytest.importorskip("mantis", reason="mantis not installed")
        from strider import CircuitBridge, solve_equilibrium
        from strider.thermo.engine import ThermoEngine

        # Simple two-strand reversible binding — the kinetic steady state must
        # match the equilibrium up to numerical tolerance.
        engine = ThermoEngine(
            material="dna", celsius=37.0, sodium=0.137, magnesium=0.01,
            backend="nupack",
        )
        seqs = {"A": "GCGCGCAAAA", "B": "TTTTGCGCGC"}

        bridge = CircuitBridge(
            reactions=["A + B <-> AB"],
            sequences=seqs,
            engine=engine,
        )

        # Equilibrium reference — engine.pfunc returns 1 M-standard ΔG for
        # every backend, so solve_equilibrium can use its default standard.
        g_a = engine.pfunc(seqs["A"]).free_energy
        g_b = engine.pfunc(seqs["B"]).free_energy
        g_ab = engine.pfunc(seqs["A"], seqs["B"]).free_energy
        eq = solve_equilibrium(
            complexes={
                "A":  (["A"], g_a),
                "B":  (["B"], g_b),
                "AB": (["A", "B"], g_ab),
            },
            totals={"A": 1e-7, "B": 1e-7},
            celsius=37.0,
        )

        # Kinetic long-time limit
        rn = bridge.to_crnetwork()
        sim = rn.simulate(
            {"A": 1e-7, "B": 1e-7, "AB": 0.0}, (0.0, 1e5),
        )
        final = sim.final()

        # Match within 5% for the dominant complex
        rel = abs(final["AB"] - eq.concentrations["AB"]) / eq.concentrations["AB"]
        assert rel < 0.05, (
            f"kinetic [AB]={final['AB']:.3e} vs equilibrium [AB]="
            f"{eq.concentrations['AB']:.3e} (rel {rel:.2%})"
        )
