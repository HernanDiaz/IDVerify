"""
scalar_vs_pareto.py — Figure 5: Multi-objective Pareto vs. scalar grid search.

Box plots comparing PR-AUC and Dice across 10 outer folds for three methods:
multi-objective Pareto HPO, scalar grid (best by PR-AUC), scalar grid (best by Dice).
Significance symbols placed directly above each non-reference box (no brackets).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle
from data_loader import PaperData

METHOD_LABELS = {
    "multiobjective": "Pareto (ours)",
    "by_prauc":       "Scalar (best PR-AUC)",
    "by_dice":        "Scalar (best Dice)",
}
METHOD_COLORS = {
    "multiobjective": IEEEStyle.COLORS["blue"],
    "by_prauc":       IEEEStyle.COLORS["orange"],
    "by_dice":        IEEEStyle.COLORS["green"],
}
METHODS_ORDER = ["multiobjective", "by_prauc", "by_dice"]

_GAP  = 0.008   # gap between box top and significance symbol
_YPAD = 0.045   # extra headroom above highest annotation


def _pvalue_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _annotate_above(ax, x: int, y_top: float, p: float) -> float:
    """Place significance symbol above box; return y coordinate used."""
    label = _pvalue_label(p)
    y = y_top + _GAP
    ax.text(x, y, label,
            ha="center", va="bottom",
            fontsize=IEEEStyle.FONT_SIZE_ANNOT + 1,
            color="black", fontweight="bold" if label != "ns" else "normal")
    return y


class ScalarVsPareto(FigureGenerator):
    """Box plots comparing Pareto HPO vs. scalar grid search on 10 outer folds."""

    def __init__(self, data: PaperData) -> None:
        self._data = data

    def _get_method_data(self, method: str, metric: str) -> np.ndarray:
        """Return per-fold metric values for a given method."""
        if method == "multiobjective":
            return self._data.nested_cv[metric].values
        df = self._data.scalar_selected
        crit = "by_prauc" if method == "by_prauc" else "by_dice"
        return df[df["selection_criterion"] == crit][metric].values

    def _get_pvalue(self, comparison: str, metric: str) -> float | None:
        row = self._data.scalar_stats[
            (self._data.scalar_stats["comparison"] == comparison) &
            (self._data.scalar_stats["metric"] == metric)
        ]
        return float(row["wilcoxon_p_holm"].values[0]) if len(row) else None

    def generate(self) -> Figure:
        metrics  = ["outer_test_pr_auc",  "outer_test_dice_global"]
        csv_keys = ["test_pr_auc",        "test_dice_global"]
        ylabels  = ["PR-AUC",             "Dice (global)"]

        fig, axes = plt.subplots(1, 2, figsize=IEEEStyle.FIGSIZE_DOUBLE_COL)

        for ax, metric, csv_key, ylabel in zip(axes, metrics, csv_keys, ylabels):

            data_per_method = {
                m: self._get_method_data(m, metric if m == "multiobjective" else csv_key)
                for m in METHODS_ORDER
            }

            positions = np.arange(len(METHODS_ORDER))
            bp = ax.boxplot(
                [data_per_method[m] for m in METHODS_ORDER],
                positions=positions,
                widths=0.45,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )

            for patch, m in zip(bp["boxes"], METHODS_ORDER):
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_alpha(0.70)
                patch.set_linewidth(0.6)

            # Individual fold dots
            for i, m in enumerate(METHODS_ORDER):
                vals   = data_per_method[m]
                jitter = np.random.default_rng(i + 42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter, vals,
                    s=8, color=METHOD_COLORS[m], alpha=0.65, linewidths=0, zorder=3,
                )

            # Significance symbols above each non-Pareto box
            ann_ys = []
            for other in ["by_prauc", "by_dice"]:
                comparison = f"multiobjective vs {other}"
                p = self._get_pvalue(comparison, csv_key)
                if p is not None:
                    j = METHODS_ORDER.index(other)
                    # whisker top = max of data (fliers excluded from whisker)
                    y_top = data_per_method[other].max()
                    ann_ys.append(_annotate_above(ax, j, y_top, p))

            # Extend y-axis so annotations never touch the top spine
            if ann_ys:
                y_ceil = max(ann_ys)
                cur_bot, cur_top = ax.get_ylim()
                ax.set_ylim(cur_bot, max(cur_top, y_ceil + _YPAD))

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [METHOD_LABELS[m] for m in METHODS_ORDER],
                fontsize=IEEEStyle.FONT_SIZE_TICK,
                rotation=15, ha="right",
            )
            ax.set_ylabel(ylabel)
            ax.set_xlim(-0.6, len(METHODS_ORDER) - 0.4)

        # ── Legend 1: method colours (upper-left of LEFT subplot) ────────────
        color_patches = [
            mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
            for m in METHODS_ORDER
        ]
        leg1 = axes[0].legend(
            handles=color_patches,
            loc="upper left", ncol=1,
            fontsize=IEEEStyle.FONT_SIZE_LEGEND,
            framealpha=1.0, edgecolor="black",
            title="Method", title_fontsize=IEEEStyle.FONT_SIZE_LEGEND,
        )
        leg1.get_frame().set_linewidth(0.5)

        # ── Legend 2: significance key (upper-right of RIGHT subplot) ────────
        sig_entries = [
            mlines.Line2D([], [], color="none", label="ns  (p \u2265 0.05)"),
        ]
        leg2 = axes[1].legend(
            handles=sig_entries,
            loc="upper right", ncol=1,
            prop={"family": "monospace", "size": IEEEStyle.FONT_SIZE_LEGEND},
            framealpha=1.0, edgecolor="black",
            title="vs. Pareto", title_fontsize=IEEEStyle.FONT_SIZE_LEGEND,
            handlelength=0, handletextpad=0.3,
        )
        leg2.get_frame().set_linewidth(0.5)

        fig.tight_layout()
        return fig
