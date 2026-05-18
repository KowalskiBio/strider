"""Tests for the strider CLI."""
import io
import json
import os
import tempfile
from contextlib import redirect_stdout

import pytest

from strider.cli import main, build_parser, _read_sequence


class TestSequenceParsing:
    def test_inline_sequence(self):
        assert _read_sequence("gcgcatgc") == "GCGCATGC"

    def test_file_sequence(self, tmp_path):
        f = tmp_path / "seq.txt"
        f.write_text("gcg\ncatgc\n")
        assert _read_sequence(f"@{f}") == "GCGCATGC"


class TestParser:
    def test_help(self):
        parser = build_parser()
        # Just ensure all subcommands are registered without raising.
        actions = {a.dest for a in parser._actions}
        assert "command" in actions


def _run(args) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(args)
    assert rc == 0
    return buf.getvalue()


class TestFold:
    def test_fold_basic(self):
        out = _run(["fold", "GCGCAAAAGCGC"])
        assert "((((....))))" in out
        assert "ΔG" in out

    def test_fold_json(self):
        out = _run(["fold", "--json", "GCGCAAAAGCGC"])
        data = json.loads(out)
        assert data["sequence"] == "GCGCAAAAGCGC"
        assert data["structure"] == "((((....))))"
        assert data["energy_kcal_per_mol"] < 0


class TestPfunc:
    def test_pfunc_basic(self):
        out = _run(["pfunc", "GCGCAAAAGCGC"])
        assert "ΔG_ens" in out

    def test_pfunc_json(self):
        out = _run(["pfunc", "--json", "GCGCAAAAGCGC"])
        data = json.loads(out)
        assert data["free_energy_kcal_per_mol"] < 0
        assert data["partition_function"] > 1.0

    def test_pfunc_multistrand(self):
        out = _run(["pfunc", "GCGCATGC", "GCATGCGC"])
        assert "ΔG_ens" in out


class TestDuplex:
    def test_duplex_auto_complement(self):
        out = _run(["duplex", "GCGCATGC"])
        assert "Tm" in out
        assert "ΔG_duplex" in out

    def test_duplex_json(self):
        out = _run(["duplex", "--json", "GCGCATGC", "GCATGCGC"])
        data = json.loads(out)
        assert "duplex_dg_kcal_per_mol" in data
        assert "melting_temperature_celsius" in data


class TestMelt:
    def test_melt_basic(self):
        out = _run(["melt", "GCGCATGCATGC"])
        assert "Tm" in out

    def test_melt_json(self):
        out = _run(["melt", "--json", "GCGCATGCATGC"])
        data = json.loads(out)
        assert isinstance(data["tm_celsius"], float)


class TestCotranscriptional:
    def test_cotx_basic(self):
        out = _run(["cotx", "GGGAAACCCAAAGGG", "--min-length", "5"])
        # Should show one line per prefix length 5..15
        lines = [l for l in out.splitlines() if l.strip()]
        # 11 prefix snapshots (5..15) plus optional rearrangement line
        assert len(lines) >= 11

    def test_cotx_json(self):
        out = _run(["cotx", "--json", "GGGAAACCC", "--min-length", "5"])
        data = json.loads(out)
        assert "prefixes" in data
        assert data["prefixes"][-1]["length"] == len("GGGAAACCC")
