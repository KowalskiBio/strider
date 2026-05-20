"""
Nearest-neighbor parameter sets.

The data model is a structured collection of Turner-style energy tables
covering the building blocks of nucleic-acid secondary structure:

    stack              base-pair stacking (16 Watson–Crick + optional wobble)
    hairpin_size       loop initiation by unpaired-loop length
    hairpin_mismatch   first-mismatch contribution at the hairpin loop closing pair
    hairpin_triloop    sequence-specific 3-nt loop bonuses
    hairpin_tetraloop  sequence-specific 4-nt loop bonuses
    interior_size      internal loop initiation by total loop length
    interior_mismatch  mismatch contribution at each closing pair of an interior loop
    interior_1_1       all 1×1 internal loops
    interior_1_2       all 1×2 internal loops
    interior_2_2       all 2×2 internal loops
    bulge_size         bulge loop initiation by length
    multiloop_init     multi-branch loop init constant ``a``
    multiloop_pair     per-branch coefficient ``b``
    multiloop_base     per-unpaired-base coefficient ``c``
    asymmetry_ninio    five-term Ninio asymmetric-loop penalty
    terminal_penalty   AU/GU (RNA) and AT (DNA) end-pair penalty
    terminal_mismatch  terminal-mismatch stabilization at duplex ends
    dangle_5, dangle_3 dangling-end contributions
    coaxial_stack      coaxial stacking between adjacent helices

References for the underlying physics:
    SantaLucia & Hicks (2004) Annu. Rev. Biophys. Biomol. Struct. 33:415-440
    SantaLucia (1998) PNAS 95:1460-1465
    Mathews, Sabina, Zuker & Turner (1999) J. Mol. Biol. 288:911-940
    Turner & Mathews (2010) Nucleic Acids Res. 38:D280-D282
    Bommarito, Peyret & SantaLucia (2000) Nucleic Acids Res. 28:1929-1934
    Lu, Turner & Mathews (2006) Nucleic Acids Res. 34:4912-4924

File format
-----------
On disk, parameter sets are plain JSON with two top-level subtrees, ``dG`` and
``dH``, that share the same keys (the list above).  Values are either Python
scalars, length-N lists of floats (size-indexed tables), or string-keyed
dicts (e.g. ``"AATT"`` for a stack — top dinucleotide then bottom
dinucleotide read 5'→3').  This is the same shape consumed by ViennaRNA,
RNAstructure, and other secondary-structure tools.

Loading
-------
:func:`load_parameters` resolves a name by searching, in order:

    1. ``$STRIDER_PARAMS_DIR`` (user override)
    2. the ``parameters/`` directory shipped with this module

The built-in name ``"native"`` (also ``"native-dna"`` / ``"native-rna"``) is
synthesized in memory from strider's nearest-neighbor constants in
:mod:`strider.thermo.nn_dna` and :mod:`strider.thermo.nn_rna`; it does not
require any JSON file.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ParameterSet",
    "load_parameters",
    "list_parameter_sets",
    "param_search_paths",
]


def _dict_get(table: Any, key: str) -> float | None:
    """Safe lookup helper for the advanced-table accessors."""
    if not isinstance(table, dict):
        return None
    val = table.get(key)
    return float(val) if val is not None else None


# ─── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ParameterSet:
    """
    Nearest-neighbor parameter set in Turner-style JSON schema.

    ``dG`` and ``dH`` are dicts with the same keys.  Each value is either a
    Python scalar, a 1-D :class:`numpy.ndarray` (length-N loop tables), or a
    ``dict`` keyed by a short DNA/RNA tuple string (e.g. ``"AATT"`` for a
    stack, ``"AAT"`` for a dangle).

    Use :func:`load_parameters` to construct from a JSON file, or
    :func:`strider.thermo.parameters_native.build_native_paramset` to assemble
    one in memory from strider's built-in constants.
    """
    name: str
    material: str                          # "DNA" or "RNA"
    default_wobble_pairing: bool
    dG: dict[str, Any] = field(default_factory=dict)
    dH: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    comment: str = ""

    # ─── convenience accessors ────────────────────────────────────────────────

    def stack(self, top5: str, top3: str, bot5: str, bot3: str) -> float:
        """
        ΔG of the 2-bp stack with closing pair (top5·bot3) and inner pair (top3·bot5).

        Lookup key is ``top5 + top3 + bot5 + bot3`` (top strand 5'→3' followed by
        bottom strand 5'→3', so the duplex visualisation is::

            5'- top5  top3 -3'
            3'- bot3  bot5 -5'

        i.e. key letters proceed clockwise around the stack starting top-left.
        """
        return float(self.dG["stack"][top5 + top3 + bot5 + bot3])

    def hairpin_loop(self, n: int) -> float:
        """Hairpin loop ΔG for an unpaired-loop length of ``n`` nt."""
        arr = self.dG["hairpin_size"]
        if n < len(arr):
            return float(arr[n])
        return float(arr[-1]) + self.dG.get("log_loop_penalty", 1.07) * float(np.log(n / (len(arr) - 1)))

    def bulge_loop(self, n: int) -> float:
        """Bulge loop ΔG for a bulge of ``n`` nt."""
        arr = self.dG["bulge_size"]
        if n < len(arr):
            return float(arr[n])
        return float(arr[-1]) + self.dG.get("log_loop_penalty", 1.07) * float(np.log(n / (len(arr) - 1)))

    def interior_loop(self, n: int) -> float:
        """Generic (size-only) interior loop ΔG for total loop length ``n`` nt."""
        arr = self.dG["interior_size"]
        if n < len(arr):
            return float(arr[n])
        return float(arr[-1]) + self.dG.get("log_loop_penalty", 1.07) * float(np.log(n / (len(arr) - 1)))

    def terminal_penalty(self, base5: str, base3: str) -> float:
        """End-base AU/GU/AT penalty applied at each terminal pair of a duplex (0 for GC/CG)."""
        tp = self.dG.get("terminal_penalty", {})
        return float(tp.get(base5 + base3, 0.0))

    def multiloop_params(self) -> tuple[float, float, float]:
        """(a, b, c) coefficients for the linear multiloop model: a + b·branches + c·unpaired."""
        return (
            float(self.dG.get("multiloop_init", 0.0)),
            float(self.dG.get("multiloop_pair", 0.0)),
            float(self.dG.get("multiloop_base", 0.0)),
        )

    # ─── advanced-table accessors ────────────────────────────────────────────
    #
    # Every accessor is a thin convenience wrapper around ``self.dG.get(...)``.
    # Returning ``None`` (rather than raising) signals "this parameter set
    # does not define an override for that table" — callers fall back to the
    # module-level constant in that case.

    def dangle_5(self, key: str) -> float | None:
        """5' dangling-end ΔG.  Key = ``base5 + close5 + close3`` (Turner 2004 §6)."""
        return _dict_get(self.dG.get("dangle_5"), key)

    def dangle_3(self, key: str) -> float | None:
        """3' dangling-end ΔG.  Key = ``close5 + close3 + base3``."""
        return _dict_get(self.dG.get("dangle_3"), key)

    def terminal_mismatch(self, key: str) -> float | None:
        """Terminal-mismatch ΔG at a helix end (Turner 2004 §7)."""
        return _dict_get(self.dG.get("terminal_mismatch"), key)

    def hairpin_mismatch(self, key: str) -> float | None:
        """First-mismatch contribution at a hairpin loop's closing pair."""
        return _dict_get(self.dG.get("hairpin_mismatch"), key)

    def interior_mismatch(self, key: str) -> float | None:
        """Mismatch contribution at each closing pair of an interior loop."""
        return _dict_get(self.dG.get("interior_mismatch"), key)

    def interior_1_1(self, key: str) -> float | None:
        """ΔG of a 1×1 internal loop (sequence-specific)."""
        return _dict_get(self.dG.get("interior_1_1"), key)

    def interior_1_2(self, key: str) -> float | None:
        """ΔG of a 1×2 internal loop (sequence-specific)."""
        return _dict_get(self.dG.get("interior_1_2"), key)

    def interior_2_2(self, key: str) -> float | None:
        """ΔG of a 2×2 internal loop (sequence-specific)."""
        return _dict_get(self.dG.get("interior_2_2"), key)

    def hairpin_triloop(self, key: str) -> float | None:
        """Sequence-specific 3-nt hairpin loop bonus (5-char key)."""
        return _dict_get(self.dG.get("hairpin_triloop"), key)

    def hairpin_tetraloop(self, key: str) -> float | None:
        """Sequence-specific 4-nt hairpin loop bonus (6-char key)."""
        return _dict_get(self.dG.get("hairpin_tetraloop"), key)

    def coaxial_stack(self, key: str) -> float | None:
        """Coaxial-stacking ΔG between adjacent helices (Walter et al. 1994)."""
        return _dict_get(self.dG.get("coaxial_stack"), key)

    # ─── inspection ───────────────────────────────────────────────────────────

    def keys(self) -> list[str]:
        """All available ΔG sub-table names."""
        return sorted(self.dG.keys())

    def has(self, key: str) -> bool:
        """Whether sub-table ``key`` is present in this parameter set."""
        return key in self.dG

    def __repr__(self) -> str:
        return (
            f"ParameterSet(name={self.name!r}, material={self.material!r}, "
            f"wobble={self.default_wobble_pairing}, keys={len(self.dG)})"
        )


# ─── loader ───────────────────────────────────────────────────────────────────

_NATIVE_NAMES = {"native", "native-dna", "native-rna"}


def param_search_paths() -> list[pathlib.Path]:
    """
    Return the ordered list of directories that :func:`load_parameters` searches
    for JSON parameter files.

    Order:
      1. ``$STRIDER_PARAMS_DIR``  (if set)
      2. strider's own ``parameters/`` directory
    """
    paths: list[pathlib.Path] = []
    env = os.environ.get("STRIDER_PARAMS_DIR")
    if env:
        paths.append(pathlib.Path(env))

    here = pathlib.Path(__file__).resolve().parent
    paths.append(here / "parameters")

    return paths


def list_parameter_sets() -> list[str]:
    """
    Names of every parameter set currently resolvable in this environment.

    Always includes the in-memory ``"native"`` adapter; additional names come
    from any ``<name>.json`` files found in :func:`param_search_paths`.
    """
    found = {"native"}
    for d in param_search_paths():
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix == ".json":
                found.add(f.stem)
    return sorted(found)


def load_parameters(name: str = "native") -> ParameterSet:
    """
    Load a parameter set by name.

    Parameters
    ----------
    name : str
        ``"native"``, ``"native-dna"``, or ``"native-rna"`` for strider's
        built-in adapter, or the basename of any ``<name>.json`` file in one
        of :func:`param_search_paths`.

    Raises
    ------
    FileNotFoundError
        if ``name`` cannot be located.  The error lists every directory that
        was searched.
    """
    if name in _NATIVE_NAMES:
        from strider.thermo.parameters_native import build_native_paramset
        material = "RNA" if name == "native-rna" else "DNA"
        return build_native_paramset(material)

    searched: list[str] = []
    for d in param_search_paths():
        searched.append(str(d))
        if not d.is_dir():
            continue
        candidate = d / f"{name}.json"
        if candidate.is_file():
            return _load_from_path(candidate)

    raise FileNotFoundError(
        f"Could not find parameter set {name!r}.\n"
        f"Searched:\n  - " + "\n  - ".join(searched) + "\n\n"
        f"To resolve:\n"
        f"  • Place a Turner-format JSON file at one of the searched paths.\n"
        f"  • OR set $STRIDER_PARAMS_DIR to a directory containing "
        f"{name}.json.\n"
        f"  • OR use parameter_set='native' (always available)."
    )


def _load_from_path(path: pathlib.Path) -> ParameterSet:
    """Parse a Turner-format JSON file at ``path`` into a :class:`ParameterSet`."""
    with open(path) as fh:
        raw = json.load(fh)

    dG = _normalize_section(raw.get("dG", {}))
    dH = _normalize_section(raw.get("dH", {}))

    return ParameterSet(
        name=str(raw.get("name", path.stem)),
        material=str(raw.get("material", "DNA")),
        default_wobble_pairing=bool(raw.get("default_wobble_pairing", False)),
        dG=dG,
        dH=dH,
        source_path=str(path),
        comment=str(raw.get("comment", "")),
    )


def _normalize_section(section: dict[str, Any]) -> dict[str, Any]:
    """
    Convert any list values in a parameter section into numpy arrays; leave
    scalars and dicts untouched.  Metadata-only keys are dropped so the result
    contains only physics tables.
    """
    out: dict[str, Any] = {}
    _metadata = {"name", "type", "material", "references", "time_generated"}
    for k, v in section.items():
        if k in _metadata:
            continue
        if isinstance(v, list):
            out[k] = np.asarray(v, dtype=float)
        else:
            out[k] = v
    return out
