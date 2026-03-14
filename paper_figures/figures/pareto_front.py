"""
pareto_front.py — Figure 1: Pareto front scatter plot (PR-AUC vs Dice).

Shows all Optuna trials as background (grey) and highlights the 51 Pareto-optimal
trials in blue, with the ideal point (1, 1) marked for reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle
from data_loader import PaperData


def _compute_global_pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of globally Pareto-optimal points (maximisation of both axes).

    A point i is dominated if there exists j such that x[j] >= x[i] AND y[j] >= y[i]
    with at least one strict inequality.
    """
    points = np.column_stack([x, y])
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        # check if any other point dominates i
        dominates_i = (
            np.all(points >= points[i], axis=1) &
            np.any(points > points[i],  axis=1)
        )
        dominates_i[i] = False
        if dominates_i.any():
            is_optimal[i] = False
    return is_optimal


class ParetoFrontFigure(FigureGenerator):
    """
    Scatter plot of validation PR-AUC vs Dice for all HPO trials.

    Pareto-optimal trials are highlighted; a step-line traces the Pareto frontier.
    """

    def __init__(self, data: PaperData) -> None:
        self._data = data

    def generate(self) -> Figure:
        pareto = self._data.pareto_front
        all_t  = self._data.all_trials

        fig, ax = plt.subplots(figsize=IEEEStyle.FIGSIZE_DOUBLE_COL)

        # ── Background: all trials ───────────────────────────────────────────
        ax.scatter(
            all_t["val_cls_prauc"],
            all_t["val_mask_dice_global"],
            s=8,
            color=IEEEStyle.COLORS["light_gray"],
            alpha=0.5,
            linewidths=0,
            label=f"All trials ($n$={len(all_t)})",
            zorder=2,
        )

        # ── Foreground: Pareto-optimal trials ────────────────────────────────
        ax.scatter(
            pareto["val_cls_prauc"],
            pareto["val_mask_dice_global"],
            s=22,
            color=IEEEStyle.COLORS["blue"],
            alpha=0.85,
            linewidths=0.3,
            edgecolors="white",
            label=f"Pareto-optimal ($n$={len(pareto)})",
            zorder=4,
        )

        # ── Global Pareto frontier step-line (computed from all 500 trials) ──
        # The per-fold Pareto fronts (blue dots) were computed independently
        # per fold. For a correct visual envelope we compute the global
        # non-dominated front from all trials and draw that as the boundary.
        global_mask = _compute_global_pareto(
            all_t["val_cls_prauc"].values,
            all_t["val_mask_dice_global"].values,
        )
        global_front = all_t[global_mask].sort_values("val_cls_prauc")
        # Step-line: for maximisation the staircase goes lower-right → upper-left
        ax.step(
            np.append(global_front["val_cls_prauc"].values, 1.0),
            np.append(global_front["val_mask_dice_global"].values[0],
                      global_front["val_mask_dice_global"].values),
            where="pre",
            color=IEEEStyle.COLORS["gray"],
            linewidth=0.9,
            linestyle="--",
            alpha=0.7,
            zorder=3,
            label=f"Global Pareto frontier ($n$={global_mask.sum()})",
        )

        # ── Ideal point (1, 1): dashed crosshair — standard in MO literature ─
        # A star marker is informal; dashed reference lines to the corner are
        # the canonical way to mark the ideal point in IEEE/optimization papers.
        x_min = max(0.0, all_t["val_cls_prauc"].min() - 0.02)
        y_min = max(0.0, all_t["val_mask_dice_global"].min() - 0.02)
        ax.plot([1.0, 1.0], [y_min, 1.0],
                color=IEEEStyle.COLORS["red"], linewidth=0.8,
                linestyle=":", alpha=0.7, zorder=3)
        ax.plot([x_min, 1.0], [1.0, 1.0],
                color=IEEEStyle.COLORS["red"], linewidth=0.8,
                linestyle=":", alpha=0.7, zorder=3)
        ax.scatter([1.0], [1.0], s=30, marker="D",
                   color=IEEEStyle.COLORS["red"], zorder=5,
                   label="Ideal point $(1,1)$")

        # ── Distance-to-ideal annotation (placed well above the point) ───────
        best  = pareto.loc[pareto["distance_to_ideal"].idxmin()]
        y_top = all_t["val_mask_dice_global"].max()
        ax.annotate(
            f"Best selected ($d$={best['distance_to_ideal']:.3f})",
            xy=(best["val_cls_prauc"], best["val_mask_dice_global"]),
            xytext=(best["val_cls_prauc"], y_top + 0.06),
            fontsize=IEEEStyle.FONT_SIZE_ANNOT,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="black",
                            connectionstyle="arc3,rad=0.0"),
            color="black",
        )

        ax.set_xlabel("Validation PR-AUC")
        ax.set_ylabel("Validation Dice (global)")
        ax.set_xlim(left=x_min)
        ax.set_ylim(bottom=y_min)
        ax.legend(loc="upper left", framealpha=1.0, bbox_to_anchor=(0.0, 0.93))

        fig.tight_layout()
        return fig
