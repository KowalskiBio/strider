"""
Arc diagram visualization for secondary structures.

Draws sequence along horizontal axis with semicircular arcs connecting
base-paired positions. Arcs are colored by pair type or probability.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def arc_diagram(
    sequence: str,
    structure: str | None = None,
    pair_probs: "np.ndarray | None" = None,
    engine=None,
    ax=None,
    color_by: str = "type",
    title: str | None = None,
    min_prob: float = 0.1,
):
    """
    Draw an arc diagram for a secondary structure.

    Parameters
    ----------
    sequence   : nucleotide sequence
    structure  : dot-bracket string (computed via engine if None)
    pair_probs : (n,n) probability matrix (overrides structure if given)
    engine     : ThermoEngine (used only if structure and pair_probs are None)
    color_by   : 'type' (GC/AT/GU) or 'probability'
    min_prob   : minimum pair probability to draw (when using pair_probs)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    seq = sequence.upper()
    n = len(seq)

    if ax is None:
        fig_w = max(8, n // 3)
        _, ax = plt.subplots(figsize=(fig_w, 4))

    # Resolve pairs
    arcs: list[tuple[int, int, float]] = []  # (i, j, weight)
    if pair_probs is not None:
        for i in range(n):
            for j in range(i + 4, n):
                p = float(pair_probs[i, j])
                if p >= min_prob:
                    arcs.append((i, j, p))
    else:
        if structure is None and engine is not None:
            result = engine.mfe(sequence)
            structure = result.structure
        if structure:
            from strider.structure.dot_bracket import parse_pairs
            pairs = parse_pairs(structure)
            arcs = [(i, j, 1.0) for i, j in pairs]

    # Draw backbone
    ax.hlines(0, 0, n - 1, color="gray", linewidth=1.5, zorder=1)

    # Draw nucleotides
    colors_nt = {"A": "#e8a838", "T": "#3878c8", "C": "#38a838", "G": "#c83838",
                 "U": "#3878c8"}
    for i, base in enumerate(seq):
        c = colors_nt.get(base, "gray")
        ax.plot(i, 0, "o", color=c, markersize=8, zorder=3)
        ax.text(i, -0.15, base, ha="center", va="top", fontsize=7, color=c)

    # Draw arcs
    cmap = plt.cm.Blues if color_by == "probability" else None
    pair_colors = {"GC": "#d44", "CG": "#d44", "AT": "#44d", "TA": "#44d",
                   "GU": "#da4", "UG": "#da4"}

    for i, j, weight in arcs:
        cx = (i + j) / 2
        radius = (j - i) / 2
        theta = [math.pi * k / 100 for k in range(101)]
        xs = [cx + radius * math.cos(t) for t in theta]
        ys = [radius * math.sin(t) for t in theta]

        if color_by == "probability":
            color = cmap(weight)
        else:
            pair = seq[i] + seq[j]
            color = pair_colors.get(pair, "#888")

        ax.plot(xs, ys, color=color, linewidth=1.0 + weight, alpha=0.7, zorder=2)

    ax.set_xlim(-1, n)
    ax.set_ylim(-0.4, max((j - i) / 2 for _, j, _ in arcs) * 1.1 if arcs else 1)
    ax.set_xlabel("Position")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    if title:
        ax.set_title(title)
    elif structure:
        ax.set_title(f"Arc diagram  |  {structure[:60]}{'...' if len(structure) > 60 else ''}")

    return ax
