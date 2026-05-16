"""
CHA circuit topology diagram.

Draws hairpin structures (stem-loop cartoons) connected by labeled arrows
showing the CHA cascade flow.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.bridge.mantis_bridge import CHABridge


def cha_circuit(
    bridge: "CHABridge | None" = None,
    ddg_values: dict | None = None,
    rates: dict | None = None,
    ax=None,
    show_ddg: bool = True,
    show_rates: bool = False,
    title: str = "CHA Circuit",
):
    """
    Draw a schematic CHA circuit diagram.

    bridge    : CHABridge (extracts DDG and rates automatically)
    ddg_values: override dict {"R1": float, "R2": float, "R3": float, "leakage": float}
    rates     : override rate dict for labeling
    show_ddg  : annotate arrows with ΔΔG values
    show_rates: annotate arrows with kinetic rates
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if bridge is not None:
        ddg = bridge.ddg_pathway
        if show_rates:
            r = bridge.rates
    elif ddg_values is not None:
        ddg = ddg_values
        r = rates or {}
    else:
        ddg = {}
        r = {}

    # Layout: miRNA → H1 → H1·H2 → H1·H2·CP (detection)
    state_x = [0.5, 2.5, 4.5, 6.5, 8.5]
    state_y = [2.5, 2.5, 2.5, 2.5, 2.5]
    state_labels = ["miR-21\n(target)", "H1\n(hairpin)", "miR·H1\n(complex)",
                    "H1·H2\n+ miR-21", "H1·H2·CP\n(signal)"]
    state_colors = ["#ef5350", "#42a5f5", "#ab47bc", "#26a69a", "#ffa726"]

    for x, y, label, color in zip(state_x, state_y, state_labels, state_colors):
        _draw_node(ax, x, y, label, color)

    # Arrows with ΔΔG labels
    arrow_specs = [
        (0, 1, "R1", "Initiation"),
        (1, 2, "R1", "H1 opening"),
        (2, 3, "R2", "Propagation"),
        (3, 4, "R3", "Detection"),
    ]

    for src_idx, tgt_idx, ddg_key, step_label in arrow_specs:
        x1, y1 = state_x[src_idx], state_y[src_idx]
        x2, y2 = state_x[tgt_idx], state_y[tgt_idx]
        ax.annotate(
            "",
            xy=(x2 - 0.35, y2),
            xytext=(x1 + 0.35, y1),
            arrowprops=dict(arrowstyle="->", color="#455a64", lw=1.5),
        )
        mid_x = (x1 + x2) / 2
        if show_ddg and ddg_key in ddg:
            ax.text(mid_x, y1 + 0.55, f"ΔΔG={ddg[ddg_key]:.1f}", ha="center",
                    fontsize=8, color="#455a64")
        ax.text(mid_x, y1 + 0.25, step_label, ha="center", fontsize=7,
                color="gray", style="italic")

    # Leakage arrow (H1 + H2 → H1H2 directly, bypass)
    ax.annotate(
        "",
        xy=(state_x[3], state_y[3] - 0.8),
        xytext=(state_x[1], state_y[1] - 0.8),
        arrowprops=dict(arrowstyle="->", color="#ef9a9a", lw=1.2, linestyle="dashed"),
    )
    mid_leak = (state_x[1] + state_x[3]) / 2
    if "leakage" in ddg:
        ax.text(mid_leak, state_y[1] - 1.2,
                f"Leakage: ΔΔG={ddg['leakage']:.1f}", ha="center",
                fontsize=7, color="#ef9a9a")

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    return ax


def _draw_node(ax, x, y, label, color, radius=0.3):
    import matplotlib.pyplot as plt
    circle = plt.Circle((x, y), radius, color=color, zorder=3, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y - 0.6, label, ha="center", va="top", fontsize=8, zorder=4)
