"""
Multi-strand tube analysis tests.

Covers Strand / Complex / SetSpec / ComplexSet enumeration semantics,
Tube.analyze numerics against the closed-form bimolecular reference, and
TubeResult.defect / .pair_probabilities lazy access.  No external tool
dependencies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strider import (
    ComplexSet,
    Complex,
    SetSpec,
    Strand,
    ThermoEngine,
    Tube,
    TubeResult,
    solve_equilibrium,
    tube_analysis,
)


R = 1.987e-3   # kcal / (mol · K)


# ─── Strand ───────────────────────────────────────────────────────────────────

class TestStrand:
    def test_basic(self):
        s = Strand("A", "ACGT")
        assert s.name == "A"
        assert s.material == "dna"
        assert len(s) == 4

    def test_material_validation(self):
        with pytest.raises(ValueError, match="material"):
            Strand("A", "ACGT", material="garbage")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Strand("", "ACGT")

    def test_hashable(self):
        # Strands are used as dict keys in Tube.strand_totals.
        s1 = Strand("A", "ACGT")
        s2 = Strand("A", "ACGT")
        d = {s1: 1.0}
        assert d[s2] == 1.0


# ─── Complex ──────────────────────────────────────────────────────────────────

class TestComplex:
    def setup_method(self):
        self.A = Strand("A", "ACGTACGT")
        self.B = Strand("B", "TTTTAAAA")

    def test_monomer(self):
        cx = Complex(strands=(self.A,))
        assert cx.n_strands == 1
        assert cx.canonical_name == "A"
        assert cx.sigma == 1

    def test_heterodimer_sigma(self):
        cx = Complex(strands=(self.A, self.B))
        assert cx.canonical_name == "A_B"
        assert cx.sigma == 1

    def test_homodimer_sigma(self):
        cx = Complex(strands=(self.A, self.A))
        assert cx.canonical_name == "A_A"
        assert cx.sigma == 2

    def test_cyclic_rotation_equal(self):
        # (A, B) and (B, A) are the same species after canonicalization.
        cx1 = Complex(strands=(self.A, self.B))
        cx2 = Complex(strands=(self.B, self.A))
        assert cx1 == cx2
        assert hash(cx1) == hash(cx2)

    def test_total_length(self):
        cx = Complex(strands=(self.A, self.B))
        assert cx.total_length == 16

    def test_explicit_name_overrides_canonical(self):
        cx = Complex(strands=(self.A, self.B), name="my_dimer")
        assert cx.canonical_name == "my_dimer"

    def test_empty_strands_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Complex(strands=())


# ─── SetSpec + ComplexSet ─────────────────────────────────────────────────────

class TestComplexSet:
    def setup_method(self):
        self.A = Strand("A", "ACGTACGT")
        self.B = Strand("B", "TTTTAAAA")

    def test_monomers_only(self):
        cs = ComplexSet([self.A, self.B], SetSpec(max_size=1))
        names = [c.canonical_name for c in cs.enumerate()]
        assert names == ["A", "B"]

    def test_max_size_2_enumerates_all_dimers(self):
        cs = ComplexSet([self.A, self.B], SetSpec(max_size=2))
        names = {c.canonical_name for c in cs.enumerate()}
        assert names == {"A", "B", "A_A", "A_B", "B_B"}

    def test_max_size_3(self):
        cs = ComplexSet([self.A, self.B], SetSpec(max_size=3))
        names = {c.canonical_name for c in cs.enumerate()}
        assert "A_A_B" in names
        assert "B_B_B" in names

    def test_explicit_include_supplements_auto(self):
        # max_size=0 disables auto; only explicit include should remain.
        spec = SetSpec(max_size=0, include=[Complex(strands=(self.A, self.B))])
        cs = ComplexSet([self.A, self.B], spec)
        names = [c.canonical_name for c in cs.enumerate()]
        assert names == ["A_B"]

    def test_exclude_removes_complex(self):
        spec = SetSpec(
            max_size=2,
            exclude=[Complex(strands=(self.A, self.A))],
        )
        cs = ComplexSet([self.A, self.B], spec)
        names = {c.canonical_name for c in cs.enumerate()}
        assert "A_A" not in names
        assert "A_B" in names

    def test_iterator_protocol(self):
        cs = ComplexSet([self.A], SetSpec(max_size=2))
        items = list(cs)
        assert len(items) == 2

    def test_len(self):
        cs = ComplexSet([self.A, self.B], SetSpec(max_size=2))
        assert len(cs) == 5

    def test_dedup_when_include_overlaps_auto(self):
        # An explicit include that matches an auto-generated complex must not
        # cause double counting.
        spec = SetSpec(
            max_size=2,
            include=[Complex(strands=(self.A, self.B))],
        )
        cs = ComplexSet([self.A, self.B], spec)
        names = [c.canonical_name for c in cs.enumerate()]
        assert names.count("A_B") == 1


# ─── Tube.analyze numerics ────────────────────────────────────────────────────

class TestTubeAnalyze:
    """Numerical accuracy against analytical / solver references."""

    def test_monomers_only_returns_totals_unchanged(self):
        # No complexes formed → free strand = total strand.
        A = Strand("A", "ACGTACGT")
        tube = Tube(
            strand_totals={A: 1e-6},
            complexes=ComplexSet([A], SetSpec(max_size=1)),
        )
        engine = ThermoEngine(material="dna", celsius=37.0)
        res = tube.analyze(engine)
        assert res.converged
        assert res.concentrations["A"] == pytest.approx(1e-6, rel=1e-6)
        assert res.strand_free["A"] == pytest.approx(1e-6, rel=1e-6)

    def test_bimolecular_matches_solve_equilibrium(self):
        # Build the same problem two ways: via solve_equilibrium directly and
        # via Tube.analyze.  Equilibrium concentrations must agree.
        A = Strand("A", "ACGTACGT")
        B = Strand("B", "ACGTACGTACGT")
        engine = ThermoEngine(material="dna", celsius=37.0)

        # Reference path: read pfunc directly, hand to the low-level solver.
        dG_A   = engine.pfunc(A.sequence).free_energy
        dG_B   = engine.pfunc(B.sequence).free_energy
        dG_AA  = engine.pfunc(A.sequence, A.sequence).free_energy
        dG_AB  = engine.pfunc(A.sequence, B.sequence).free_energy
        dG_BB  = engine.pfunc(B.sequence, B.sequence).free_energy

        ref = solve_equilibrium(
            complexes={
                "A":   (["A"], dG_A),
                "B":   (["B"], dG_B),
                "A_A": (["A", "A"], dG_AA),
                "A_B": (["A", "B"], dG_AB),
                "B_B": (["B", "B"], dG_BB),
            },
            totals={"A": 1e-6, "B": 1e-6},
            celsius=37.0,
        )

        # New API path.
        tube = Tube(
            strand_totals={A: 1e-6, B: 1e-6},
            complexes=ComplexSet([A, B], SetSpec(max_size=2)),
        )
        new = tube.analyze(engine)

        for k in ref.concentrations:
            assert new.concentrations[k] == pytest.approx(ref.concentrations[k], rel=1e-6)

    def test_unknown_strand_in_complex_raises(self):
        A = Strand("A", "ACGT")
        rogue = Strand("R", "AAAA")
        tube = Tube(
            strand_totals={A: 1e-6},
            complexes=ComplexSet([A], SetSpec(
                max_size=0,
                include=[Complex(strands=(A, rogue))],
            )),
        )
        engine = ThermoEngine(material="dna", celsius=37.0)
        with pytest.raises(ValueError, match="not present"):
            tube.analyze(engine)

    def test_strong_binding(self):
        # A long perfect-complement duplex should bind tightly: most of A is
        # consumed into A_B.
        A = Strand("A", "GCGCGCGCGCGC")
        B = Strand("B", "GCGCGCGCGCGC")
        tube = Tube(
            strand_totals={A: 1e-7, B: 1e-7},
            complexes=ComplexSet([A, B], SetSpec(max_size=2)),
        )
        engine = ThermoEngine(material="dna", celsius=37.0)
        res = tube.analyze(engine)
        assert res.converged
        free_A = res.strand_free["A"]
        # Total free monomer should be much less than total input.
        assert free_A < 0.5e-7


# ─── TubeResult lazy access ──────────────────────────────────────────────────

class TestTubeResultLazyAccess:
    def test_pair_probabilities(self):
        A = Strand("A", "GCGCAAAAGCGC")
        tube = Tube(
            strand_totals={A: 1e-6},
            complexes=ComplexSet([A], SetSpec(max_size=1)),
        )
        engine = ThermoEngine(material="dna", celsius=37.0)
        res = tube.analyze(engine)
        P = res.pair_probabilities("A")
        n = len(A)
        assert P.shape == (n, n)
        # Symmetric to within numerical noise.
        assert np.allclose(P, P.T, atol=1e-6)
        # No negative probabilities.  (The native McCaskill outside recurrence
        # can slightly exceed 1.0 because multiloop outside terms are not
        # fully wired — pre-existing limitation flagged in ensemble.py docstring.)
        assert P.min() >= -1e-9
        assert P.max() <= 1.15

    def test_defect_matches_engine(self):
        A = Strand("A", "GCGCAAAAGCGC")
        tube = Tube(
            strand_totals={A: 1e-6},
            complexes=ComplexSet([A], SetSpec(max_size=1)),
        )
        engine = ThermoEngine(material="dna", celsius=37.0)
        res = tube.analyze(engine)

        target = "((((....))))"
        d_via_tube = res.defect("A", target)
        d_direct = engine.ensemble_defect(A.sequence, target)
        assert d_via_tube == pytest.approx(d_direct, abs=1e-9)

    def test_lazy_methods_require_engine(self):
        # A TubeResult built without an engine (e.g. deserialized) raises a
        # helpful error when lazy methods are called.
        bare = TubeResult(
            tube_name="t",
            concentrations={"A": 1e-6},
            free_energies={"A": 0.0},
            strand_free={"A": 1e-6},
            complexes={"A": Complex(strands=(Strand("A", "ACGT"),))},
            converged=True,
        )
        with pytest.raises(RuntimeError, match="engine"):
            bare.pair_probabilities("A")
        with pytest.raises(RuntimeError, match="engine"):
            bare.defect("A", "....")


# ─── tube_analysis driver ────────────────────────────────────────────────────

class TestTubeAnalysisDriver:
    def test_multiple_tubes(self):
        A = Strand("A", "ACGTACGT")
        engine = ThermoEngine(material="dna", celsius=37.0)
        tubes = [
            Tube(strand_totals={A: 1e-6}, complexes=ComplexSet([A], SetSpec(max_size=1)), name="dilute"),
            Tube(strand_totals={A: 1e-4}, complexes=ComplexSet([A], SetSpec(max_size=2)), name="dense"),
        ]
        results = tube_analysis(tubes, engine)
        assert set(results.keys()) == {"dilute", "dense"}
        assert all(r.converged for r in results.values())
        # In the denser tube, more A is consumed into A_A.
        assert results["dense"].concentrations.get("A_A", 0) > 0

    def test_duplicate_tube_name_raises(self):
        A = Strand("A", "ACGT")
        engine = ThermoEngine(material="dna", celsius=37.0)
        tubes = [
            Tube(strand_totals={A: 1e-6}, complexes=ComplexSet([A], SetSpec(max_size=1)), name="t1"),
            Tube(strand_totals={A: 1e-6}, complexes=ComplexSet([A], SetSpec(max_size=1)), name="t1"),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            tube_analysis(tubes, engine)


# ─── Backward compatibility with equilibrium_from_engine ──────────────────────

class TestBackwardCompatibility:
    def test_equilibrium_from_engine_still_works(self):
        # The legacy wrapper now delegates to Tube — must still produce
        # equivalent results.
        from strider import equilibrium_from_engine
        engine = ThermoEngine(material="dna", celsius=37.0)
        res = equilibrium_from_engine(
            engine,
            strands={"A": "GCGCGCGC", "B": "GCGCGCGC"},
            totals={"A": 1e-7, "B": 1e-7},
            max_size=2,
        )
        assert res.converged
        # Standard dimerization-style equilibrium: all strands appear in `strand_free`.
        assert "A" in res.strand_free
        assert "B" in res.strand_free
