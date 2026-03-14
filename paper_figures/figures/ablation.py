"""
ablation.py — Figure 3: Ablation study violin plots (4 variants × 30 seeds).

Two-subplot figure comparing PR-AUC and Dice across multitask, cls_only,
seg_only, and unweighted_losses variants. Significance symbols are placed
directly above each non-reference violin (no stacked brackets).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle
from data_loader import PaperData

VARIANTS_ORDER = ["multitask", "cls_only", "seg_only", "unweighted_losses"]
_GAP   = 0.010   # gap between violin top and significance symbol
_YPAD  = 0.055   # extra y-axis headroom above the highest annotation


def _pvalue_label(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _annotate_above(ax, x: int, y_top: float, p: float) -> float:
    """Place significance symbol above violin; return the y coordinate used."""
    label = _pvalue_label(p)
    y = y_top + _GAP
    ax.text(x, y, label,
            ha="center", va="bottom",
            fontsize=IEEEStyle.FONT_SIZE_ANNOT + 1,
            color="black", fontweight="bold" if label != "ns" else "normal")
    return y


class AblationFigure(FigureGenerator):
    """Violin plots for ablation study with statistical significance annotations."""

    def __init__(self, data: PaperData) -> None:
        self._data = data

    def _get_stat(self, comparison: str, metric: str) -> float | None:
        """Return Holm-corrected Wilcoxon p-value for a given comparison/metric."""
        row = self._data.stat_tests[
            (self._data.stat_tests["comparison"] == comparison) &
            (self._data.stat_tests["metric"] == metric)
        ]
        return float(row["wilcoxon_p_holm"].values[0]) if len(row) else None

    def generate(self) -> Figure:
        df = self._data.blind_test

        metrics   = ["test_pr_auc",   "test_dice_global"]
        ylabels   = ["PR-AUC",        "Dice (global)"]
        stat_keys = ["test_pr_auc",   "test_dice_global"]

        fig, axes = plt.subplots(1, 2, figsize=IEEEStyle.FIGSIZE_DOUBLE_COL)

        for ax, metric, ylabel in zip(axes, metrics, ylabels):

            data_per_variant = [
                df.loc[df["variant"] == v, metric].values
                for v in VARIANTS_ORDER
            ]
            colors = [IEEEStyle.VARIANT_COLORS[v] for v in VARIANTS_ORDER]
            labels = [IEEEStyle.VARIANT_LABELS[v] for v in VARIANTS_ORDER]

            parts = ax.violinplot(
                data_per_variant,
                positions=range(len(VARIANTS_ORDER)),
                showmedians=True,
                showextrema=True,
                widths=0.6,
            )

            # Colour each violin individually
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.65)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.5)
            for part_name in ("cmedians", "cmaxes", "cmins", "cbars"):
                parts[part_name].set_linewidth(0.8)
                parts[part_name].set_color("black")

            # Overlay individual seed dots
            for i, vals in enumerate(data_per_variant):
                jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter, vals,
                    s=4, color=colors[i], alpha=0.55, linewidths=0, zorder=3,
                )

            # Significance symbols + track highest annotation y
            ann_ys = []
            for j, other in enumerate(VARIANTS_ORDER):
                if other == "multitask":
                    continue
                comparison = f"multitask vs {other}"
                p = self._get_stat(comparison, stat_keys[metrics.index(metric)])
                if p is not None:
                    y_top = data_per_variant[j].max()
                    ann_ys.append(_annotate_above(ax, j, y_top, p))

            # Extend y-axis so annotations never touch the top spine
            if ann_ys:
                y_ceil = max(ann_ys)
                cur_bot, cur_top = ax.get_ylim()
                ax.set_ylim(cur_bot, max(cur_top, y_ceil + _YPAD))

            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(VARIANTS_ORDER)))
            ax.set_xticklabels(labels, rotation=20, ha="right",
                               fontsize=IEEEStyle.FONT_SIZE_TICK)
            ax.set_xlim(-0.6, len(VARIANTS_ORDER) - 0.4)

        # ── Legend 1: significance key (lower-left of LEFT subplot) ─────────
        sig_entries = [
            mlines.Line2D([], [], color="none", label="*** (p < 0.001)"),
            mlines.Line2D([], [], color="none", label="ns  (p \u2265 0.05)"),
        ]
        leg1 = axes[0].legend(
            handles=sig_entries,
            loc="lower left", ncol=1,
            prop={"family": "monospace", "size": IEEEStyle.FONT_SIZE_LEGEND},
            framealpha=1.0, edgecolor="black",
            title="vs. Multitask", title_fontsize=IEEEStyle.FONT_SIZE_LEGEND,
            handlelength=0, handletextpad=0.3,
        )
        leg1.get_frame().set_linewidth(0.5)

        # ── Legend 2: variant colours (lower-right of RIGHT subplot) ─────────
        color_patches = [
            mpatches.Patch(color=IEEEStyle.VARIANT_COLORS[v],
                           label=IEEEStyle.VARIANT_LABELS[v])
            for v in VARIANTS_ORDER
        ]
        leg2 = axes[1].legend(
            handles=color_patches,
            loc="lower right", ncol=1,
            fontsize=IEEEStyle.FONT_SIZE_LEGEND,
            framealpha=1.0, edgecolor="black",
            title="Variant", title_fontsize=IEEEStyle.FONT_SIZE_LEGEND,
        )
        leg2.get_frame().set_linewidth(0.5)

        fig.tight_layout()
        return fig
