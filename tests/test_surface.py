"""Tests for the surface-tethered biophysics layer (strider.surface)."""
import numpy as np
import pytest

from strider import (
    SurfaceModel, SurfaceParams, PrussianBlueLabel, ReadoutChain,
    LabelModel, SurfaceCorrection, tether_dg, double_layer_local_salt,
    debye_length_m,
)
from strider.surface.transducer import captured_count, shoup_szabo_f


# ─── label addressability ──────────────────────────────────────────────────────

class TestPrussianBlueLabel:
    def test_small_particle_fully_addressable(self):
        # At fM-relevant sizes the K+ front penetrates the whole particle.
        lbl = PrussianBlueLabel(diameter_nm=40.0)
        assert lbl.f_addressable() == pytest.approx(1.0, abs=1e-6)

    def test_addressability_collapses_for_large_or_slow(self):
        big = PrussianBlueLabel(diameter_nm=2000.0)            # 2 µm particle
        slow = PrussianBlueLabel(diameter_nm=40.0, counterion_d_m2_s=1e-18)
        assert big.f_addressable() < 0.2
        assert slow.f_addressable() < 0.5

    def test_signal_scales_with_volume(self):
        small = PrussianBlueLabel(diameter_nm=20.0, f_addressable_override=1.0)
        large = PrussianBlueLabel(diameter_nm=40.0, f_addressable_override=1.0)
        # Z ∝ R³ ⇒ 40 nm carries ~8× the charge of 20 nm.
        assert large.signal_per_event() / small.signal_per_event() == pytest.approx(8.0, rel=1e-3)

    def test_override_bypasses_submodel(self):
        lbl = PrussianBlueLabel(diameter_nm=2000.0, f_addressable_override=0.3)
        assert lbl.f_addressable() == 0.3


class TestReadoutChain:
    def test_floor_matches_lmp91000_esp32(self):
        ro = ReadoutChain()
        # (3.0/2^12)/√64 / 350kΩ ≈ 0.26 nA
        assert ro.current_floor_A() == pytest.approx(0.26e-9, rel=0.05)
        assert ro.charge_floor_C() == pytest.approx(ro.current_floor_A() * ro.dpv_pulse_s)


# ─── capture physics ───────────────────────────────────────────────────────────

class TestCapture:
    def test_shoup_szabo_limits(self):
        # f(τ)→1.0 as τ→∞ (steady-state 4·D·a flux); diverges as τ→0 (Cottrell).
        assert shoup_szabo_f(1e9) == pytest.approx(1.0, abs=1e-3)
        assert shoup_szabo_f(1e-6) > 100.0

    def test_capture_linear_at_low_analyte(self):
        p = SurfaceParams()
        t = np.linspace(0.0, p.incubation_s, 200)
        lo = captured_count(t, np.full_like(t, 1e-15), p)
        hi = captured_count(t, np.full_like(t, 2e-15), p)
        # well below N_max ⇒ doubling concentration doubles capture.
        assert hi / lo == pytest.approx(2.0, rel=1e-3)
        assert lo < 0.01 * p.max_capture_sites()

    def test_capture_saturates_at_uloq(self):
        p = SurfaceParams()
        t = np.linspace(0.0, p.incubation_s, 200)
        n = captured_count(t, np.full_like(t, 1e-3), p)   # huge analyte
        assert n == pytest.approx(p.max_capture_sites(), rel=1e-3)


class TestSurfaceModel:
    def test_capture_fraction_matches_urotrace(self):
        # 40 nm PBNP ×1, 3 mm electrode, 90 min, 50 µL → ~6.7% capture,
        # detectable (urotrace PROGRESS §6b).
        p = SurfaceParams(incubation_s=90 * 60)
        m = SurfaceModel(p)
        t = np.linspace(0.0, 90 * 60, 400)
        # dimer grows from 0 to ~6.6 fM over the incubation (≈ the fM-regime
        # cascade trace); capture fraction is geometry-set, ~6.7%.
        dimer = 6.6e-15 * (t / t[-1])
        r = m.transduce(t, dimer)
        assert 0.05 < r.capture_fraction < 0.09
        assert r.detectable
        assert r.peak_current_A > p.readout.current_floor_A()

    def test_lod_returns_lowest_detectable(self):
        p = SurfaceParams(incubation_s=90 * 60)
        m = SurfaceModel(p)
        t = np.linspace(0.0, 90 * 60, 400)
        ref = 6.6e-15 * (t / t[-1])                    # reference ramp at 1 fM trigger

        def make_trace(c):                             # linear scaling
            return t, ref * (c / 1e-15)

        triggers = np.logspace(-18, -12, 31)
        lod = m.lod(make_trace, triggers)
        assert lod is not None
        # below LOD is not detectable; at/above is
        assert not m.transduce(*make_trace(lod / 10)).detectable
        assert m.transduce(*make_trace(lod)).detectable

    def test_custom_label_plugs_in(self):
        # A non-Prussian-Blue label only needs signal_per_event().
        class FixedChargeLabel(LabelModel):
            def signal_per_event(self):
                return 1e-15        # 1 fC per captured event

        p = SurfaceParams(incubation_s=90 * 60, label=FixedChargeLabel())
        m = SurfaceModel(p)
        t = np.linspace(0.0, 90 * 60, 200)
        r = m.transduce(t, np.full_like(t, 1e-13))
        assert r.peak_charge_C == pytest.approx(r.n_captured * 1e-15, rel=1e-9)


# ─── surface thermodynamics ────────────────────────────────────────────────────

class TestSurfaceThermo:
    def test_tether_penalty_sign_and_table(self):
        assert tether_dg(None) == 0.0
        assert tether_dg("c6") > 0.0
        assert tether_dg("peg18") > tether_dg("c6")
        # ideal-chain term relieved by a longer linker
        assert tether_dg("c6", n_segments=2) > tether_dg("c6", n_segments=18)

    def test_double_layer_enhances_local_salt(self):
        bulk = 0.137
        local = double_layer_local_salt(bulk, probe_density_per_m2=1e16)
        assert local > bulk                              # counterion accumulation
        # denser, more charged monolayer ⇒ more enhancement
        denser = double_layer_local_salt(bulk, 1e16, charge_per_probe_e=-5.0)
        assert denser > local
        # no charge or no density ⇒ unchanged
        assert double_layer_local_salt(bulk, 0.0) == bulk
        assert double_layer_local_salt(bulk, 1e16, charge_per_probe_e=0.0) == bulk

    def test_debye_length_decreases_with_salt(self):
        assert debye_length_m(0.01) > debye_length_m(0.1)
        assert debye_length_m(0.0) == float("inf")

    def test_surface_correction_callable_and_offsets(self):
        seq = "ACGTACGTACGTACGTACGT"
        sc = SurfaceCorrection(probe_density_per_m2=1e16, charge_per_probe_e=-5.0,
                               spacer="c6", n_segments=6)
        # callable returns the per-strand salt offset (stabilizing here)
        assert sc(seq) == sc.salt_offset(seq)
        assert sc.salt_offset(seq) <= 0.0
        assert sc.tether_offset() > 0.0
        # vanishes as the layer vanishes / no spacer
        none = SurfaceCorrection(probe_density_per_m2=0.0, spacer=None)
        assert none.salt_offset(seq) == pytest.approx(0.0, abs=1e-12)
        assert none.tether_offset() == 0.0

    def test_correction_plugs_into_engine(self):
        from strider.thermo.engine import ThermoEngine
        sc = SurfaceCorrection(probe_density_per_m2=1e16, charge_per_probe_e=-10.0)
        eng = ThermoEngine(material="dna", celsius=37.0, backend="native",
                           correction_model=sc)
        plain = ThermoEngine(material="dna", celsius=37.0, backend="native")
        seq = "GCGCATGCATGCATGCGC"
        # surface engine shifts the free energy by the salt offset
        delta = eng.pfunc(seq).free_energy - plain.pfunc(seq).free_energy
        assert delta == pytest.approx(sc.salt_offset(seq), abs=1e-6)
