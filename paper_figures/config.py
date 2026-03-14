"""
config.py — Paths and IEEE-style matplotlib settings for paper figures.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent.parent
EXPORT_DIR  = ROOT_DIR / "exports_hpo_pareto_nested"
OUTPUT_DIR  = Path(__file__).parent / "output"

# ── IEEE Style ───────────────────────────────────────────────────────────────

class IEEEStyle:
    """Matplotlib style constants following IEEE double-column format guidelines."""

    # Figure sizes (inches)
    FIGSIZE_SINGLE_COL  = (3.5, 2.6)    # single column
    FIGSIZE_DOUBLE_COL  = (7.16, 3.2)   # full double column
    FIGSIZE_DOUBLE_TALL = (7.16, 4.5)   # double column, taller

    # Font sizes (pt)
    FONT_SIZE_BASE   = 9
    FONT_SIZE_TITLE  = 10
    FONT_SIZE_TICK   = 8
    FONT_SIZE_LEGEND = 8
    FONT_SIZE_ANNOT  = 7.5

    # Lines & markers
    LINE_WIDTH   = 1.0
    MARKER_SIZE  = 5
    CAP_SIZE     = 3     # error bar cap size

    # DPI (for rasterised elements within vector PDF)
    DPI = 300

    # Colorblind-safe palette (Wong 2011)
    COLORS = {
        "blue":        "#0072B2",
        "orange":      "#E69F00",
        "green":       "#009E73",
        "red":         "#D55E00",
        "purple":      "#CC79A7",
        "sky":         "#56B4E9",
        "yellow":      "#F0E442",
        "black":       "#000000",
        "gray":        "#999999",
        "light_gray":  "#CCCCCC",
    }

    # Variant display names and colors for ablation figures
    VARIANT_LABELS = {
        "multitask":          "Multitask",
        "cls_only":           "Cls-only",
        "seg_only":           "Seg-only",
        "unweighted_losses":  "Unweighted",
    }
    VARIANT_COLORS = {
        "multitask":          "#0072B2",
        "cls_only":           "#E69F00",
        "seg_only":           "#009E73",
        "unweighted_losses":  "#D55E00",
    }

    @classmethod
    def apply(cls) -> None:
        """Apply IEEE rcParams globally to matplotlib."""
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family":          "serif",
            "font.serif":           ["Times New Roman", "DejaVu Serif"],
            "font.size":            cls.FONT_SIZE_BASE,
            "axes.titlesize":       cls.FONT_SIZE_TITLE,
            "axes.labelsize":       cls.FONT_SIZE_BASE,
            "xtick.labelsize":      cls.FONT_SIZE_TICK,
            "ytick.labelsize":      cls.FONT_SIZE_TICK,
            "legend.fontsize":      cls.FONT_SIZE_LEGEND,
            "lines.linewidth":      cls.LINE_WIDTH,
            "lines.markersize":     cls.MARKER_SIZE,
            "axes.linewidth":       0.8,
            "grid.linewidth":       0.5,
            "grid.alpha":           0.4,
            "axes.grid":            True,
            "axes.axisbelow":       True,
            "figure.dpi":           cls.DPI,
            "savefig.dpi":          cls.DPI,
            "savefig.bbox":         "tight",
            "savefig.pad_inches":   0.02,
            "pdf.fonttype":         42,   # embed fonts as Type 42 (TrueType) for IEEE
            "ps.fonttype":          42,
        })
