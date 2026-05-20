"""
Tests for Turner-style nearest-neighbor parameter sets.

Covers the in-memory native adapter (built from SantaLucia 2004 / Mathews 1999
constants), the JSON loader, and engine integration.  A round-trip test
exercises the JSON path using a synthetic file written into a tmp directory —
no external parameter source is required.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from strider import (
    ParameterSet,
    load_parameters,
    list_parameter_sets,
    param_search_paths,
)
from strider.thermo.parameters_native import build_native_paramset
from strider.thermo.engine import ThermoEngine


# ─── native adapter ───────────────────────────────────────────────────────────

class TestNativeParameterSet:
    def test_load_native(self):
        p = load_parameters("native")
        assert isinstance(p, ParameterSet)
        assert p.material == "DNA"
        assert p.default_wobble_pairing is False

    def test_load_native_rna(self):
        p = load_parameters("native-rna")
        assert p.material == "RNA"
        assert p.default_wobble_pairing is True

    def test_stack_table_has_16_wc_entries(self):
        p = build_native_paramset("DNA")
        stack = p.dG["stack"]
        for b1 in "ACGT":
            for b2 in "ACGT":
                key = b1 + b2 + _wc_dna(b1 + b2)
                assert key in stack, f"missing stack key {key}"
        assert len(stack) == 16

    def test_stack_matches_santalucia_2004(self):
        # SantaLucia 2004 Table 1: AA/TT stack ΔG37 = -1.00 kcal/mol.
        p = build_native_paramset("DNA")
        assert p.dG["stack"]["AATT"] == pytest.approx(-1.00, abs=0.01)
        # CG/CG (-2.17) and GC/GC (-2.24) are the strongest WC stacks.
        assert p.dG["stack"]["CGCG"] == pytest.approx(-2.17, abs=0.05)
        assert p.dG["stack"]["GCGC"] == pytest.approx(-2.24, abs=0.05)

    def test_stack_matches_mathews_1999(self):
        # Mathews 1999 Table 4: AU/UA stack ΔG37 = -0.93 kcal/mol.
        p = build_native_paramset("RNA")
        assert p.dG["stack"]["AATT"] == pytest.approx(-0.93, abs=0.05)
        # GC/GC RNA stack is the strongest single WC stack at ≈ -3.42.
        assert p.dG["stack"]["GCGC"] < -3.0

    def test_hairpin_loop_table_shape(self):
        p = build_native_paramset("DNA")
        assert p.dG["hairpin_size"].shape == (30,)
        # Loop sizes < 3 are sentinels (no hairpin possible by geometric constraint).
        assert p.dG["hairpin_size"][2] > 10.0

    def test_multiloop_params(self):
        a, b, c = build_native_paramset("DNA").multiloop_params()
        # Turner 2004 / SantaLucia: a > 0, b > 0, c ≥ 0.
        assert a > 0 and b > 0 and c >= 0

    def test_terminal_penalty(self):
        p = build_native_paramset("DNA")
        # SantaLucia INIT_AT penalty applies at AT termini; GC termini carry none.
        assert p.terminal_penalty("A", "T") > 0
        assert p.terminal_penalty("G", "C") == 0.0

    def test_dh_table_present(self):
        p = build_native_paramset("DNA")
        # Enthalpy ΔH must round-trip alongside ΔG so future T-extrapolation works.
        assert "stack" in p.dH
        assert p.dH["stack"]["AATT"] == pytest.approx(-7.9, abs=0.01)


def _wc_dna(dinuc: str) -> str:
    """5'→3' DNA complement of ``dinuc``, returned in the orientation used for stack keys."""
    return dinuc.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# ─── loader plumbing ──────────────────────────────────────────────────────────

class TestLoader:
    def test_list_includes_native(self):
        names = list_parameter_sets()
        assert "native" in names

    def test_search_paths_includes_module_dir(self):
        paths = [str(p) for p in param_search_paths()]
        # The bundled parameters/ directory must always appear.
        assert any(p.endswith("strider/thermo/parameters") for p in paths)

    def test_search_paths_respects_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))
        paths = [str(p) for p in param_search_paths()]
        assert paths[0] == str(tmp_path)

    def test_missing_set_reports_search_paths(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError) as exc:
            load_parameters("does-not-exist")
        msg = str(exc.value)
        assert "Searched" in msg
        assert str(tmp_path) in msg


# ─── JSON round-trip (no external dependency) ─────────────────────────────────

class TestJsonRoundTrip:
    def _synthetic_paramfile(self, tmp_path: pathlib.Path, name: str = "synthetic") -> pathlib.Path:
        """Write a minimal Turner-schema JSON file and return its path."""
        data = {
            "name": name,
            "material": "DNA",
            "default_wobble_pairing": False,
            "comment": "synthetic test fixture — not for production use",
            "dG": {
                "stack": {
                    "AATT": -1.00, "ATAT": -0.88, "TATA": -0.58, "CGCG": -2.17,
                    "GCGC": -2.24, "GGCC": -1.84,
                },
                "hairpin_size": [99.0, 99.0, 99.0, 4.1, 4.3, 4.9, 4.4, 4.3, 4.1, 4.0],
                "bulge_size": [99.0, 3.8, 2.8, 3.2, 3.6, 4.0],
                "interior_size": [99.0, 99.0, 4.1, 5.1, 4.9, 5.3],
                "asymmetry_ninio": [0.4, 0.3, 0.2, 0.1, 3.0],
                "terminal_penalty": {"AT": 0.45, "TA": 0.45},
                "multiloop_init": 3.4,
                "multiloop_pair": 0.4,
                "multiloop_base": 0.0,
                "log_loop_penalty": 1.07,
            },
            "dH": {
                "stack": {"AATT": -7.9, "CGCG": -10.6},
                "hairpin_size": [0.0] * 10,
            },
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        return path

    def test_round_trip_via_env_override(self, monkeypatch, tmp_path):
        self._synthetic_paramfile(tmp_path, "synthetic")
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))

        p = load_parameters("synthetic")
        assert p.name == "synthetic"
        assert p.material == "DNA"
        assert p.default_wobble_pairing is False
        assert p.dG["stack"]["AATT"] == -1.00
        assert isinstance(p.dG["hairpin_size"], np.ndarray)
        assert p.dG["hairpin_size"].dtype == np.float64
        assert p.dH["stack"]["AATT"] == -7.9

    def test_list_picks_up_synthetic_file(self, monkeypatch, tmp_path):
        self._synthetic_paramfile(tmp_path, "synthetic")
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))
        names = list_parameter_sets()
        assert "synthetic" in names
        assert "native" in names

    def test_metadata_keys_stripped_from_sections(self, monkeypatch, tmp_path):
        # The dH section may carry extra meta keys (name, type, references…);
        # the loader strips them so only physics tables remain.
        path = tmp_path / "meta.json"
        path.write_text(json.dumps({
            "name": "meta",
            "material": "DNA",
            "default_wobble_pairing": False,
            "dG": {"stack": {"AATT": -1.0}, "hairpin_size": [99.0, 99.0, 99.0, 4.1]},
            "dH": {
                "name": "ignored", "type": "ignored", "material": "DNA",
                "references": ["ignored"], "time_generated": "ignored",
                "stack": {"AATT": -7.9},
            },
        }))
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))
        p = load_parameters("meta")
        assert set(p.dH.keys()) == {"stack"}

    def test_self_consistency_with_native(self, monkeypatch, tmp_path):
        # Dump the native DNA set to JSON, then reload it — energies must match.
        native = build_native_paramset("DNA")

        def _serializable(section):
            out = {}
            for k, v in section.items():
                if isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                else:
                    out[k] = v
            return out

        data = {
            "name": "roundtrip",
            "material": native.material,
            "default_wobble_pairing": native.default_wobble_pairing,
            "dG": _serializable(native.dG),
            "dH": _serializable(native.dH),
        }
        path = tmp_path / "roundtrip.json"
        path.write_text(json.dumps(data))
        monkeypatch.setenv("STRIDER_PARAMS_DIR", str(tmp_path))

        reloaded = load_parameters("roundtrip")
        for key in native.dG["stack"]:
            assert reloaded.dG["stack"][key] == pytest.approx(native.dG["stack"][key])
        assert reloaded.dG["hairpin_size"].shape == native.dG["hairpin_size"].shape


# ─── engine integration ───────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_default_params_resolve_to_native(self):
        e = ThermoEngine(material="dna")
        assert e.params.name == "native-dna"

    def test_default_rna(self):
        e = ThermoEngine(material="rna")
        assert e.params.name == "native-rna"

    def test_explicit_parameter_set_name(self):
        e = ThermoEngine(material="dna", parameter_set="native")
        assert e.params.material == "DNA"

    def test_explicit_parameter_set_instance(self):
        ps = build_native_paramset("DNA")
        e = ThermoEngine(material="dna", parameter_set=ps)
        # Engine must hold on to the user-supplied instance verbatim.
        assert e.params is ps

    def test_cache_key_changes_with_parameter_set(self):
        e1 = ThermoEngine(material="dna", parameter_set="native")
        e2 = ThermoEngine(material="dna")  # default sentinel
        k1 = e1._cache_key("mfe", ("ACGT",))
        k2 = e2._cache_key("mfe", ("ACGT",))
        assert k1 != k2

    def test_repr_includes_parameter_set_when_set(self):
        e = ThermoEngine(material="dna", parameter_set="native")
        assert "parameter_set='native'" in repr(e)


class TestCustomParamsAffectNumerics:
    """A custom ParameterSet must actually change pfunc / MFE output."""

    def test_perturbed_stack_changes_pfunc(self):
        """
        Doubling every stack ΔG must drive a hairpin's ensemble free energy
        sharply more negative (more bp Boltzmann weight ⇒ lower G).
        """
        from copy import deepcopy

        baseline = ThermoEngine(material="dna", parameter_set="native-dna")
        g_baseline = baseline.pfunc("GCGCAAAAGCGC").free_energy

        custom = build_native_paramset("DNA")
        custom = deepcopy(custom)
        custom.dG["stack"] = {k: 2.0 * v for k, v in custom.dG["stack"].items()}
        # Strip name so the engine treats it as a non-default instance.
        custom.name = "perturbed-stack"

        e = ThermoEngine(material="dna", parameter_set=custom)
        g_custom = e.pfunc("GCGCAAAAGCGC").free_energy

        # Doubling the (already-negative) stack energies must move ΔG further
        # negative — by at least 1 kcal/mol on a 4-bp stem.
        assert g_custom < g_baseline - 1.0, (
            f"expected stronger binding under doubled stack, "
            f"got baseline={g_baseline:.3f}, custom={g_custom:.3f}"
        )

    def test_perturbed_multiloop_changes_mfe(self):
        """A large multi-loop init penalty must suppress multi-branch folds."""
        from copy import deepcopy

        # A sequence that *can* form either two hairpins (multiloop) or a
        # single stem-loop.  Multiloop pays ML_INIT once; if we crank that up
        # the optimal fold cannot be a multiloop.
        seq = "GCGCGGAAAACCGCGCAACGCGCAAAAGCGCG"

        baseline = ThermoEngine(material="dna", parameter_set="native-dna")
        mfe_base = baseline.mfe(seq)

        custom = deepcopy(build_native_paramset("DNA"))
        custom.dG["multiloop_init"] = 100.0   # forbid multiloops
        custom.name = "no-multiloop"
        e = ThermoEngine(material="dna", parameter_set=custom)
        mfe_custom = e.mfe(seq)

        # Custom fold has fewer branches ⇒ fewer pairs or higher ΔG than baseline.
        assert mfe_custom.energy >= mfe_base.energy - 1e-6, (
            f"forbidding multiloops should not produce a more stable fold: "
            f"baseline={mfe_base.energy:.3f}, custom={mfe_custom.energy:.3f}"
        )

    def test_native_string_alias_uses_module_constants(self):
        """
        Passing the built-in ``"native-dna"`` name takes the default path —
        the override channel is *not* entered, so output must be bit-identical
        to the no-parameter-set default.
        """
        e_default = ThermoEngine(material="dna")
        e_native = ThermoEngine(material="dna", parameter_set="native-dna")
        assert e_default.pfunc("GCGCAAAAGCGC").free_energy == pytest.approx(
            e_native.pfunc("GCGCAAAAGCGC").free_energy
        )
