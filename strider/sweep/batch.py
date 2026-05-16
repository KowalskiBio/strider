"""
Vectorized parameter sweeps for thermodynamic calculations.

ParameterSweep runs a function over a grid of parameters,
caching results and parallelizing with ProcessPoolExecutor.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.sweep.cache import DiskCache


@dataclass
class SweepResult:
    """Results of a parameter sweep."""
    axes: dict[str, np.ndarray]
    values: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self):
        """Return a tidy pandas DataFrame with one column per axis parameter and a 'value' column."""
        import pandas as pd
        rows = []
        for idx in np.ndindex(*self.values.shape):
            row = {name: self.axes[name][i] for name, i in zip(self.axes.keys(), idx)}
            row["value"] = self.values[idx]
            rows.append(row)
        return pd.DataFrame(rows)

    def optimum(self) -> dict:
        """Return the parameter dict corresponding to the global minimum across the grid."""
        flat_idx = int(np.argmin(self.values.ravel()))
        multi_idx = np.unravel_index(flat_idx, self.values.shape)
        return {
            name: float(self.axes[name][i])
            for name, i in zip(self.axes.keys(), multi_idx)
        }

    def plot(self, ax=None, xlabel: str | None = None, ylabel: str = "Score"):
        """Plot the sweep result: line plot for 1-D axes, filled contour for 2-D axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        names = list(self.axes.keys())
        if len(names) == 1:
            ax.plot(self.axes[names[0]], self.values.ravel(), marker="o")
            ax.set_xlabel(xlabel or names[0])
            ax.set_ylabel(ylabel)
        elif len(names) == 2:
            X, Y = np.meshgrid(self.axes[names[0]], self.axes[names[1]], indexing="ij")
            c = ax.contourf(X, Y, self.values, levels=20, cmap="viridis")
            plt.colorbar(c, ax=ax, label=ylabel)
            ax.set_xlabel(xlabel or names[0])
            ax.set_ylabel(names[1])
        return ax


class ParameterSweep:
    """
    Vectorized parameter sweeps with caching and parallelism.

    Parameters
    ----------
    engine    : ThermoEngine (used for thermodynamic calls)
    cache     : optional DiskCache for memoization across sessions
    n_workers : number of parallel workers (1 = sequential)
    """

    def __init__(
        self,
        engine: "ThermoEngine",
        cache: "DiskCache | None" = None,
        n_workers: int = 1,
    ) -> None:
        self.engine = engine
        self.cache = cache
        self.n_workers = n_workers

    def toehold_sweep(
        self,
        hairpin_seq: str,
        toehold_lengths: list[int],
        target_strand: str,
    ) -> SweepResult:
        """
        Sweep toehold length → (kf, ΔΔG, accessibility).

        Returns SweepResult with values = kf at each toehold length.
        """
        from strider.kinetics.tmsd import toehold_kf, rates_from_ddg

        kf_vals = []
        for nt in toehold_lengths:
            toehold = hairpin_seq[:nt]
            kf = toehold_kf(nt, self.engine.material, self.engine.celsius)
            kf_vals.append(kf)

        return SweepResult(
            axes={"toehold_length": np.array(toehold_lengths)},
            values=np.array(kf_vals),
            metadata={"unit": "M^-1 s^-1", "hairpin": hairpin_seq[:20]},
        )

    def temperature_sweep(
        self,
        sequences: dict[str, str],
        temperatures: list[float],
    ) -> SweepResult:
        """
        Sweep temperature → duplex ΔG for each sequence pair.

        Returns SweepResult with values = dict of ΔG per sequence name.
        """
        from strider.thermo.engine import ThermoEngine

        dg_matrix = np.zeros((len(sequences), len(temperatures)))
        names = list(sequences.keys())

        for j, T in enumerate(temperatures):
            eng = ThermoEngine(
                material=self.engine.material,
                celsius=T,
                sodium=self.engine.sodium,
                magnesium=self.engine.magnesium,
                backend=self.engine.backend_name,
                cache=self.cache,
            )
            for i, (name, seq) in enumerate(sequences.items()):
                dg_matrix[i, j] = eng.pfunc(seq).free_energy

        return SweepResult(
            axes={"temperature_C": np.array(temperatures)},
            values=dg_matrix,
            metadata={"strand_names": names},
        )

    def grid_sweep(
        self,
        axes: dict[str, list],
        fn: Callable[[dict], float],
        flatten: bool = True,
    ) -> SweepResult:
        """
        N-dimensional grid sweep.

        axes : {param_name: [value1, value2, ...]}
        fn   : fn({param_name: value, ...}) -> float
        """
        ax_names = list(axes.keys())
        ax_arrays = [np.array(axes[n]) for n in ax_names]
        shape = tuple(len(v) for v in ax_arrays)
        values = np.zeros(shape)

        tasks = list(cartesian_product(*[enumerate(v) for v in ax_arrays]))

        if self.n_workers > 1 and len(tasks) > 4:
            results = self._parallel_sweep(tasks, ax_names, fn, shape)
            values = results
        else:
            for combo in tasks:
                idx = tuple(i for i, _ in combo)
                params = {n: float(v) for n, (_, v) in zip(ax_names, combo)}
                key = _cache_key(fn, params)
                if self.cache:
                    cached = self.cache.get(key)
                    if cached is not None:
                        values[idx] = cached
                        continue
                val = fn(params)
                if self.cache:
                    self.cache.set(key, val)
                values[idx] = val

        return SweepResult(
            axes={n: a for n, a in zip(ax_names, ax_arrays)},
            values=values,
        )

    def _parallel_sweep(self, tasks, ax_names, fn, shape):
        """Execute the grid sweep in parallel using ProcessPoolExecutor and collect results."""
        values = np.zeros(shape)
        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {}
            for combo in tasks:
                idx = tuple(i for i, _ in combo)
                params = {n: float(v) for n, (_, v) in zip(ax_names, combo)}
                futures[ex.submit(fn, params)] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    values[idx] = fut.result()
                except Exception:
                    values[idx] = float("nan")
        return values


def _cache_key(fn: Callable, params: dict) -> str:
    """Build a deterministic SHA-256 key from a function name and its parameter dict."""
    from strider.sweep.cache import DiskCache
    return DiskCache.make_key(fn.__name__, sorted(params.items()))
