"""
Optional NUPACK compatibility shim.

If a user has NUPACK installed, this module routes strider's ThermoEngine
calls to NUPACK for higher-accuracy results. strider works without NUPACK;
this shim is purely additive.

Usage:
    engine = ThermoEngine(backend="nupack")
    # All calculations now use NUPACK internally.
"""

from __future__ import annotations


def is_nupack_available() -> bool:
    """Return True if the nupack package is importable."""
    try:
        import nupack  # noqa: F401
        return True
    except ImportError:
        return False


def nupack_mfe(strands: list[str], celsius: float = 37.0, material: str = "dna") -> tuple[str, float]:
    """Thin wrapper around nupack.mfe()."""
    import nupack
    model = nupack.Model(material=material, celsius=celsius, sodium=0.137, magnesium=0.01)
    result = nupack.mfe(strands, model)
    if result:
        return str(result[0].structure), float(result[0].energy)
    return "." * sum(len(s) for s in strands), 0.0


def nupack_pfunc(strands: list[str], celsius: float = 37.0, material: str = "dna") -> tuple[float, float]:
    """Thin wrapper around nupack.pfunc(). Returns (ensemble_dG, log_partition_function)."""
    import nupack
    import math
    model = nupack.Model(material=material, celsius=celsius, sodium=0.137, magnesium=0.01)
    pf, dG = nupack.pfunc(strands, model)
    return float(dG), float(pf)
