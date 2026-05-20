"""
Export a strider :class:`ParameterSet` to a Turner-style JSON file.

Practical workflow for users who want a custom set
--------------------------------------------------

1. Export the native baseline::

       python scripts/export_paramset.py --name native-dna --out my-dna.json

2. Edit the resulting JSON in place — change individual entries in any
   sub-table (stack, dangle_5, interior_1_1, ...).
3. Drop the file in ``$STRIDER_PARAMS_DIR`` (or in
   ``strider/thermo/parameters/`` for in-tree distribution).
4. Load it as::

       engine = ThermoEngine(material="dna", parameter_set="my-dna")

The ``parameters_native`` adapter populates *every* sub-table consumed by
the energy DP, so a JSON exported here is a complete, self-contained
parameter set with no implicit fallback to module constants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from strider.thermo.parameters import load_parameters, param_search_paths
from strider.thermo.parameters_native import build_native_paramset


def _serialise(value: Any) -> Any:
    """Convert numpy / nested-dict values to JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return [float(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--name", default="native-dna",
        help='ParameterSet name to export (default "native-dna"). '
             'Use "native-rna" for the RNA baseline, or any name resolvable '
             'by `load_parameters`.',
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Destination JSON path.  Defaults to ./<name>.json",
    )
    p.add_argument(
        "--rebuild-native", action="store_true",
        help="Skip the loader and rebuild a native set in memory from the "
             "module constants (useful for regenerating the baseline).",
    )
    args = p.parse_args()

    if args.rebuild_native:
        if args.name.endswith("rna"):
            ps = build_native_paramset("RNA")
        else:
            ps = build_native_paramset("DNA")
    else:
        ps = load_parameters(args.name)

    data = {
        "name": ps.name,
        "material": ps.material,
        "default_wobble_pairing": ps.default_wobble_pairing,
        "comment": ps.comment,
        "dG": _serialise(ps.dG),
        "dH": _serialise(ps.dH),
    }

    out = args.out or Path(f"{ps.name}.json")
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print("loader search paths:")
    for path in param_search_paths():
        print(f"  {path}")


if __name__ == "__main__":
    main()
