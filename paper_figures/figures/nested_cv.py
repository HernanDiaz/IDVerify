"""
nested_cv.py — Figure 2: Nested CV results per outer fold.

Grouped bar chart showing PR-AUC and Dice for each of the 10 outer folds,
with horizontal dashed lines indicating the mean across folds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle
from data_loader import PaperData


class NestedCVFigure(FigureGenerator):
    """Bar chart of outer-fold test metrics (PR-AUC and Dice) from nested CV."""

    def __init__(self, data: PaperData) -> None:
        self._data = data

    def generate(self) -> Figure:
        df = self._data.nested_cv.sort_values("outer_fold")

        folds     = df["outer_fold"].values
        pr_auc    = df["outer_test_pr_auc"].values
        dice      = df["outer_test_dice_global"].values

        x     = np.arange(len(folds))
        width = 0.35

        fig, ax = plt.subplots(figsize=IEEEStyle.FIGSIZE_DOUBLE_COL)

        bars1 = ax.bar(
            x - width / 2, pr_auc, width,
            label="PR-AUC",
            color=IEEEStyle.COLORS["blue"],
            alpha=0.85,
            linewidth=0.5,
            edgecolor="white",
        )
        bars2 = ax.bar(
            x + width / 2, dice, width,
            label="Dice",
            color=IEEEStyle.COLORS["red"],
            alpha=0.85,
            linewidth=0.5,
            edgecolor="white",
        )

        # Mean lines
        mean_pr  = pr_auc.mean()
        mean_dice = dice.mean()
        std_pr   = pr_auc.std()
        std_dice = dice.std()

        ax.axhline(mean_pr,   color=IEEEStyle.COLORS["blue"],   linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label=f"PR-AUC mean={mean_pr:.4f}±{std_pr:.4f}")
        ax.axhline(mean_dice, color=IEEEStyle.COLORS["red"], linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label=f"Dice mean={mean_dice:.4f}±{std_dice:.4f}")

        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Metric value")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in folds])
        ax.set_ylim(0.75, 1.02)
        ax.legend(loc="lower left", framealpha=1.0,
                  fontsize=IEEEStyle.FONT_SIZE_LEGEND)

        fig.tight_layout()
        return fig
