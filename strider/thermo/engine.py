"""
ThermoEngine — central dispatch for all thermodynamic calculations.

Backend selection order (automatic):
    nupack  (if installed, highest accuracy)
    vienna  (if ViennaRNA installed, GPL, open source)
    native  (built-in NN implementation, always available)
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from strider.sweep.cache import DiskCache
    from strider.thermo.modified import ModificationSite

BackendName = Literal["auto", "native", "vienna", "nupack"]

R = 1.987e-3  # kcal / (mol · K)

COMPLEMENT_DNA = str.maketrans("ACGT", "TGCA")


@dataclass
class MFEResult:
    energy: float           # kcal/mol
    structure: str          # dot-bracket
    base_pairs: list[tuple[int, int]] = field(default_factory=list)
    sequence: str = ""


@dataclass
class PFuncResult:
    free_energy: float          # ensemble ΔG (kcal/mol)
    partition_function: float   # dimensionless Q
    pair_probs: np.ndarray      # shape (n, n)


class ThermoEngine:
    """
    Central thermodynamic engine.

    Parameters
    ----------
    material : 'dna' or 'rna'
    celsius  : temperature in Celsius
    sodium   : [Na+] in molar
    magnesium: [Mg2+] in molar
    backend  : 'auto' | 'native' | 'vienna' | 'nupack'
    cache    : optional DiskCache for persistent memoization
    correction_model : optional callable(sequence) -> float for ML corrections
    """

    def __init__(
        self,
        material: Literal["dna", "rna"] = "dna",
        celsius: float = 37.0,
        sodium: float = 0.137,
        magnesium: float = 0.01,
        backend: BackendName = "auto",
        cache: "DiskCache | None" = None,
        correction_model: Callable[[str], float] | None = None,
    ) -> None:
        self.material = material
        self.celsius = celsius
        self.sodium = sodium
        self.magnesium = magnesium
        self.cache = cache
        self.correction_model = correction_model
        self._backend = self._resolve_backend(backend)

    # ─── public API ──────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """The active backend name: 'native', 'vienna', or 'nupack'."""
        return self._backend

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return a list of backend names importable in the current environment."""
        backends = ["native"]
        try:
            import RNA  # noqa: F401
            backends.append("vienna")
        except ImportError:
            pass
        try:
            import nupack  # noqa: F401
            backends.append("nupack")
        except ImportError:
            pass
        return backends

    def mfe(self, *sequences: str) -> MFEResult:
        """Minimum free energy structure for one or more strands."""
        key = self._cache_key("mfe", sequences)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        result = self._mfe_dispatch(sequences)
        if self.cache:
            self.cache.set(key, result)
        return result

    def pfunc(self, *sequences: str) -> PFuncResult:
        """Ensemble free energy and pair probability matrix."""
        key = self._cache_key("pfunc", sequences)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        result = self._pfunc_dispatch(sequences)
        if self.correction_model is not None:
            combined = "".join(sequences)
            result = PFuncResult(
                result.free_energy + self.correction_model(combined),
                result.partition_function,
                result.pair_probs,
            )
        if self.cache:
            self.cache.set(key, result)
        return result

    def pairs(self, *sequences: str) -> np.ndarray:
        """Pair-probability matrix P[i,j] for the given (multi-)strand complex."""
        return self.pfunc(*sequences).pair_probs

    def ensemble_defect(
        self,
        sequences: str | tuple[str, ...],
        target_structure: str,
        normalize: bool = True,
    ) -> float:
        """
        Ensemble defect of a target dot-bracket structure for the given complex.

        Defect = Σ_i (1 − P_correct(i)), where
          - if position i is unpaired in the target, P_correct(i) = 1 − Σ_j P(i,j)
          - if position i pairs with j in the target, P_correct(i) = P(i,j)

        If ``normalize`` is True (default), the defect is divided by sequence
        length so the value lies in [0, 1].
        """
        from strider.structure.dot_bracket import parse_pairs
        if isinstance(sequences, str):
            seqs = (sequences,)
        else:
            seqs = tuple(sequences)
        clean_target = target_structure.replace("&", "").replace("+", "")
        n = sum(len(s) for s in seqs)
        if len(clean_target) != n:
            raise ValueError(
                f"target structure length {len(clean_target)} != total sequence length {n}"
            )

        probs = self.pairs(*seqs)
        target_pairs = dict()
        for i, j in parse_pairs(target_structure):
            target_pairs[i] = j
            target_pairs[j] = i

        defect = 0.0
        for i in range(n):
            if i in target_pairs:
                j = target_pairs[i]
                p_correct = float(probs[i][j])
            else:
                p_correct = 1.0 - float(probs[i].sum())
            defect += max(0.0, 1.0 - p_correct)

        return defect / n if normalize else defect

    def duplex_dg(self, seq1: str, seq2: str | None = None) -> float:
        """
        ΔG (kcal/mol) of hybridization.

        seq2=None → hairpin (intramolecular folding of seq1).
        """
        if seq2 is None:
            return self.pfunc(seq1).free_energy
        return self._duplex_dg_native(seq1, seq2)

    def ddg(
        self,
        reactants: list[str | list[str]],
        products: list[str | list[str]],
    ) -> float:
        """
        ΔΔG = Σ G(products) - Σ G(reactants) (kcal/mol).

        Each element of reactants/products is either:
          - a single sequence string → compute pfunc of that strand alone
          - a list of sequences → compute pfunc of that multi-strand complex
        """
        def g(item):
            if isinstance(item, str):
                return self.pfunc(item).free_energy
            return self.pfunc(*item).free_energy

        g_react = sum(g(r) for r in reactants)
        g_prod = sum(g(p) for p in products)
        return g_prod - g_react

    def toehold_accessibility(
        self,
        sequence: str,
        toehold_positions: slice | list[int],
    ) -> float:
        """
        Fraction of ensemble where all toehold positions are unpaired.
        """
        result = self.pfunc(sequence)
        pair_probs = result.pair_probs
        n = len(sequence)

        if isinstance(toehold_positions, slice):
            positions = list(range(*toehold_positions.indices(n)))
        else:
            positions = list(toehold_positions)

        if not positions:
            return 1.0

        # Probability position i is unpaired = 1 - Σ_j P(i,j)
        probs_unpaired = [1.0 - pair_probs[i].sum() for i in positions]
        # Joint probability (lower bound, assuming independence)
        return float(np.prod(probs_unpaired))

    def mfe_batch(
        self,
        strand_groups: list[tuple[str, ...]],
        n_workers: int = 1,
    ) -> list[MFEResult]:
        """Parallelized batch MFE computation."""
        if n_workers <= 1 or len(strand_groups) < 4:
            return [self.mfe(*grp) for grp in strand_groups]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(self.mfe, *grp) for grp in strand_groups]
            return [f.result() for f in futures]

    def melting_temperature(
        self,
        seq: str,
        strand_conc_M: float = 250e-9,
    ) -> float:
        """Melting temperature (°C) for the given sequence."""
        from strider.thermo.nn_dna import melting_temperature as _mt_dna
        from strider.thermo.nn_rna import duplex_dg_rna
        if self.material == "dna":
            return _mt_dna(seq, strand_conc_M, self.sodium, self.magnesium)
        # RNA: approximate
        dH = -80.0  # placeholder; full RNA Tm needs Turner tables
        return dH

    # ─── dispatch ────────────────────────────────────────────────────────────

    def _mfe_dispatch(self, sequences: tuple[str, ...]) -> MFEResult:
        """Route MFE calculation to the active backend."""
        if self._backend == "nupack":
            return self._mfe_nupack(sequences)
        if self._backend == "vienna":
            return self._mfe_vienna(sequences)
        return self._mfe_native(sequences)

    def _pfunc_dispatch(self, sequences: tuple[str, ...]) -> PFuncResult:
        """Route partition function calculation to the active backend."""
        if self._backend == "nupack":
            return self._pfunc_nupack(sequences)
        if self._backend == "vienna":
            return self._pfunc_vienna(sequences)
        return self._pfunc_native(sequences)

    # ─── native backend ───────────────────────────────────────────────────────

    def _mfe_native(self, sequences: tuple[str, ...]) -> MFEResult:
        """MFE via the built-in Zuker-style DP (strider.structure.mfe)."""
        from strider.structure.mfe import fold_mfe
        seq = _concat(sequences)
        structure, energy, pairs = fold_mfe(seq, self.celsius, self.material)
        return MFEResult(energy=energy, structure=structure, base_pairs=pairs, sequence=seq)

    def _pfunc_native(self, sequences: tuple[str, ...]) -> PFuncResult:
        """Partition function via the built-in McCaskill DP (single- or multi-strand)."""
        if len(sequences) == 1:
            from strider.thermo.ensemble import ensemble_dg
            dG, probs = ensemble_dg(sequences[0], self.celsius, self.material)
        else:
            # Multi-strand: nick-aware McCaskill DP on concatenated sequence.
            # Returns ensemble ΔG of the complex (not the binding ΔG).
            # engine.ddg() subtracts individual strand energies to get ΔΔG.
            from strider.thermo.ensemble import multistrand_pairs
            dG, probs = multistrand_pairs(list(sequences), self.celsius, self.material)

        Z = math.exp(-dG / (R * (self.celsius + 273.15)))
        return PFuncResult(free_energy=dG, partition_function=Z, pair_probs=probs)

    def _duplex_dg_native(self, seq1: str, seq2: str) -> float:
        """Dispatch duplex ΔG to the correct NN table (DNA/DNA, RNA/RNA, or DNA:RNA hybrid)."""
        from strider.thermo.nn_dna import duplex_dg
        from strider.thermo.nn_rna import duplex_dg_rna
        from strider.thermo.nn_dna_rna import hybrid_duplex_dg

        s1 = seq1.upper()
        s2 = seq2.upper()
        has_u1 = "U" in s1
        has_u2 = "U" in s2

        if has_u1 or has_u2:
            if has_u1 and has_u2:
                return duplex_dg_rna(s1, self.celsius, self.sodium)
            # hybrid
            dna = s2 if has_u1 else s1
            return hybrid_duplex_dg(dna, self.celsius, self.sodium)
        return duplex_dg(s1, s2, self.celsius, self.sodium)

    # ─── vienna backend ───────────────────────────────────────────────────────

    def _mfe_vienna(self, sequences: tuple[str, ...]) -> MFEResult:
        """MFE via ViennaRNA RNA.fold() or RNA.cofold() for multi-strand input."""
        from strider.thermo import vienna_backend as vb
        from strider.structure.dot_bracket import parse_pairs
        seq = _concat(sequences) if len(sequences) == 1 else sequences[0] + "&" + sequences[-1]
        if len(sequences) > 1:
            structure, energy = vb.co_fold(sequences[0], sequences[-1], self.celsius)
        else:
            structure, energy = vb.fold(sequences[0], self.celsius)
        pairs = parse_pairs(structure.replace("&", ""))
        return MFEResult(energy=energy, structure=structure, base_pairs=pairs, sequence=seq)

    def _pfunc_vienna(self, sequences: tuple[str, ...]) -> PFuncResult:
        """Partition function via ViennaRNA pf_fold()."""
        from strider.thermo import vienna_backend as vb
        seq = sequences[0] if len(sequences) == 1 else sequences[0] + "&" + sequences[-1]
        dG, probs = vb.pf_fold(sequences[0], self.celsius)
        Z = math.exp(-dG / (R * (self.celsius + 273.15)))
        return PFuncResult(free_energy=dG, partition_function=Z, pair_probs=probs)

    # ─── nupack backend ───────────────────────────────────────────────────────

    def _mfe_nupack(self, sequences: tuple[str, ...]) -> MFEResult:
        """MFE via NUPACK nupack.mfe()."""
        import nupack
        from strider.structure.dot_bracket import parse_pairs
        model = self._nupack_model()
        result = nupack.mfe(list(sequences), model)
        if result:
            e = float(result[0].energy)
            s = str(result[0].structure)
            pairs = parse_pairs(s)
            return MFEResult(energy=e, structure=s, base_pairs=pairs, sequence="".join(sequences))
        return MFEResult(energy=0.0, structure="." * sum(len(s) for s in sequences))

    def _pfunc_nupack(self, sequences: tuple[str, ...]) -> PFuncResult:
        """
        Partition function via NUPACK ``nupack.pfunc()``, with pair probabilities
        via ``nupack.pairs()``.

        NUPACK reports ΔG at a water-molarity standard state.  We shift to a
        1 M reference (the convention used by strider's native backend and by
        SantaLucia / Turner nearest-neighbor parameters) so that engine.ddg(),
        equilibrium solvers, and detailed-balance kinetics all see consistent
        units regardless of the backend.
        """
        import nupack
        from strider.equilibrium import water_molarity
        model = self._nupack_model()
        pf, dG = nupack.pfunc(list(sequences), model)

        # Standard-state conversion: ΔG_1M = ΔG_c0 + (N − 1) · RT · ln(c0)
        n_strands = len(sequences)
        if n_strands > 1:
            c0 = water_molarity(self.celsius)
            dG = float(dG) + (n_strands - 1) * R * (self.celsius + 273.15) * math.log(c0)

        try:
            pm = nupack.pairs(list(sequences), model)
            probs = np.asarray(pm.to_array()) if hasattr(pm, "to_array") else np.asarray(pm)
        except Exception:
            n = sum(len(s) for s in sequences)
            probs = np.zeros((n, n))
        return PFuncResult(free_energy=float(dG), partition_function=float(pf), pair_probs=probs)

    def _nupack_model(self):
        """Build a nupack.Model from the engine's material/temperature/salt settings."""
        import nupack
        return nupack.Model(
            material=self.material,
            celsius=self.celsius,
            sodium=self.sodium,
            magnesium=self.magnesium,
        )

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _resolve_backend(self, backend: BackendName) -> str:
        """Resolve 'auto' to the best available backend; pass explicit names through unchanged."""
        if backend != "auto":
            return backend
        available = self.available_backends()
        for preferred in ("nupack", "vienna", "native"):
            if preferred in available:
                return preferred
        return "native"

    def _cache_key(self, op: str, sequences: tuple[str, ...]) -> str:
        """Build a SHA-256 cache key from the operation, conditions, and sequence content."""
        raw = f"{op}|{self.material}|{self.celsius}|{self.sodium}|{self.magnesium}|{'|'.join(sequences)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"ThermoEngine(material={self.material!r}, celsius={self.celsius}, "
            f"sodium={self.sodium}, magnesium={self.magnesium}, backend={self._backend!r})"
        )


def _concat(sequences: tuple[str, ...]) -> str:
    """Concatenate a tuple of sequence strings into a single string."""
    return "".join(sequences)
