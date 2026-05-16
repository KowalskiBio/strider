"""Export format tests."""
import pytest
from strider.export.formats import to_vienna, to_ct, to_bpseq, to_fasta, to_oxdna


SEQ = "GCATGCATGC"
STRUCT = "(((....)))"


class TestFormats:
    def test_vienna(self):
        out = to_vienna(SEQ, STRUCT, name="test")
        assert out.startswith(">test\n")
        assert SEQ in out
        assert STRUCT in out

    def test_ct_has_n_rows(self):
        out = to_ct(SEQ, STRUCT, name="test")
        lines = out.strip().split("\n")
        # 1 header + n rows
        assert len(lines) == len(SEQ) + 1

    def test_bpseq_has_n_rows(self):
        out = to_bpseq(SEQ, STRUCT)
        data_lines = [l for l in out.strip().split("\n") if not l.startswith("#")]
        assert len(data_lines) == len(SEQ)

    def test_fasta(self):
        out = to_fasta(SEQ, name="seq1", description="test")
        assert out.startswith(">seq1 test\n")
        assert SEQ in out

    def test_oxdna_topology(self):
        out = to_oxdna(SEQ)
        lines = out.strip().split("\n")
        # First line: N 1
        assert lines[0] == f"{len(SEQ)} 1"
        assert len(lines) == len(SEQ) + 1

    def test_ct_paired_positions(self):
        out = to_ct(SEQ, STRUCT)
        lines = out.strip().split("\n")[1:]  # skip header
        # Position 0 (1-indexed=1) should be paired with position 9 (1-indexed=10)
        first_row = lines[0].split("\t")
        assert int(first_row[4]) == 10  # paired with position 10

    def test_bpseq_unpaired_is_zero(self):
        out = to_bpseq(SEQ, "." * len(SEQ))
        data_lines = [l for l in out.strip().split("\n") if not l.startswith("#")]
        for line in data_lines:
            parts = line.split()
            assert parts[2] == "0", f"Unpaired position should be 0, got {parts[2]}"
