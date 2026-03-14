"""
challenge.py — Figure 4: DocVerify vs. DeepID 2025 Challenge participants.

Horizontal bar chart comparing detection F1@0.5 (Track 1) and localization
F1 per-image (Track 2) across published challenge results and DocVerify.
Evaluation note: challenge results are on the official test set; DocVerify
results are on an internal holdout (15% of FantasyID).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle
from data_loader import PaperData

# Published challenge results (Korshunov et al., ICCVW 2025, Tables 2–3)
# Format: (display_name, f1_det, f1_loc, training_data_note)
# None = did not participate in that track — only top 4 finishers retained
CHALLENGE_SYSTEMS = [
    ("Sunlight (1st)",        0.991,  0.784, "60K+ external imgs"),
    ("Incode (2nd)",          0.868,  None,  "100K proprietary imgs"),
    ("AG / EdgeDoc (3rd)",    0.958,  0.686, "FantasyID only"),
    ("UAM-Biometrics (4th)",  0.712,  0.620, "FantasyID only"),
]


class ChallengeFigure(FigureGenerator):
    """
    Horizontal grouped bar chart comparing DocVerify against challenge participants.
    """

    def __init__(self, data: PaperData) -> None:
        self._data = data

    def _get_docverify_stats(self) -> dict:
        """Extract DocVerify mean ± std from challenge_summary."""
        df  = self._data.challenge_summary
        det = df[df["metric"] == "det_f1_thr05"].iloc[0]
        loc = df[df["metric"] == "loc_f1_perimage"].iloc[0]
        return {
            "det_mean": det["mean"], "det_std": det["std"],
            "loc_mean": loc["mean"], "loc_std": loc["std"],
        }

    def generate(self) -> Figure:
        stats = self._get_docverify_stats()

        fig, axes = plt.subplots(1, 2, figsize=IEEEStyle.FIGSIZE_DOUBLE_TALL,
                                 sharey=False)

        track_configs = [
            {
                "ax":    axes[0],
                "title": "Track 1 — Detection F1@0.5",
                "key":   "f1_det",
                "dv_mean": stats["det_mean"],
                "dv_std":  stats["det_std"],
            },
            {
                "ax":    axes[1],
                "title": "Track 2 — Localization F1 (per-image)",
                "key":   "f1_loc",
                "dv_mean": stats["loc_mean"],
                "dv_std":  stats["loc_std"],
            },
        ]

        for cfg in track_configs:
            ax    = cfg["ax"]
            key   = cfg["key"]
            idx   = 0 if key == "f1_det" else 1

            # Filter systems with a score for this track
            systems_with_score = [
                (name, row[idx], note)
                for name, *row, note in CHALLENGE_SYSTEMS
                if row[idx] is not None
            ]

            # Add DocVerify at the end
            systems_with_score.append(
                ("DocVerify (ours)†", cfg["dv_mean"], "FantasyID only")
            )

            # Sort descending by score
            systems_with_score.sort(key=lambda x: x[1], reverse=True)

            names  = [s[0] for s in systems_with_score]
            scores = [s[1] for s in systems_with_score]
            colors = [
                IEEEStyle.COLORS["blue"] if "ours" in n else IEEEStyle.COLORS["light_gray"]
                for n in names
            ]
            edge_colors = [
                IEEEStyle.COLORS["blue"] if "ours" in n else IEEEStyle.COLORS["gray"]
                for n in names
            ]

            y_pos = np.arange(len(names))
            bars  = ax.barh(y_pos, scores, color=colors, edgecolor=edge_colors,
                            linewidth=0.6, height=0.6)

            # Error bar for DocVerify only
            dv_idx = next(i for i, n in enumerate(names) if "ours" in n)
            ax.barh(
                y_pos[dv_idx], scores[dv_idx],
                xerr=cfg["dv_std"],
                color="none", edgecolor="none",
                ecolor=IEEEStyle.COLORS["blue"],
                capsize=IEEEStyle.CAP_SIZE,
                linewidth=0.9,
                height=0.6,
            )

            # Value labels — placed after error bar end for DocVerify, after bar for others
            for i, (bar, score) in enumerate(zip(bars, scores)):
                is_dv = "ours" in names[i]
                x_label = score + (cfg["dv_std"] if is_dv else 0) + 0.008
                ax.text(x_label, i, f"{score:.3f}",
                        va="center", fontsize=IEEEStyle.FONT_SIZE_ANNOT)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=IEEEStyle.FONT_SIZE_TICK)
            ax.set_xlabel("F1 score")
            ax.set_title(cfg["title"], fontsize=IEEEStyle.FONT_SIZE_TITLE)
            ax.set_xlim(0, 1.15)
            ax.invert_yaxis()

        fig.tight_layout()
        return fig
