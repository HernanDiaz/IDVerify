"""
export_hires.py — Export all paper figures in high-resolution format for journal submission.

Elsevier requires separate high-resolution files in TIFF or EPS format.

Output (paper_figures/output/hires/):
    fig0_architecture_compact.eps   (vector, from SVG via cairosvg)
    fig1_pareto_front.eps           (vector, matplotlib)
    fig2_nested_cv.eps              (vector, matplotlib)
    fig3_ablation.eps               (vector, matplotlib)
    fig4_challenge.eps              (vector, matplotlib)
    fig5_scalar_vs_pareto.eps       (vector, matplotlib)
    fig_qual_attack_input.tiff      (300 DPI raster)
    fig_qual_attack_mask.tiff       (300 DPI raster)
    fig_qual_attack_overlay.tiff    (300 DPI raster)

Usage:
    cd E:\\PycharmProjects\\DocVerify
    venv\\Scripts\\python paper_figures\\export_hires.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import IEEEStyle, OUTPUT_DIR
from data_loader import load_paper_data
from figures.pareto_front     import ParetoFrontFigure
from figures.nested_cv        import NestedCVFigure
from figures.ablation         import AblationFigure
from figures.challenge        import ChallengeFigure
from figures.scalar_vs_pareto import ScalarVsPareto

HIRES_DIR   = OUTPUT_DIR / "hires"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "prl" / "figures"
TIFF_DPI    = 300


def save_eps(generator, filename: str) -> None:
    """Generate a figure and save it as EPS."""
    path = HIRES_DIR / filename
    fig = generator.generate()
    fig.savefig(path, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  [saved] {filename}  ({path.stat().st_size // 1024} KB)")


def convert_svg_to_eps(svg_path: Path, eps_path: Path) -> None:
    """Convert SVG to EPS using svglib + reportlab."""
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPS
    drawing = svg2rlg(str(svg_path))
    renderPS.drawToFile(drawing, str(eps_path))
    print(f"  [saved] {eps_path.name}  ({eps_path.stat().st_size // 1024} KB)")


def convert_png_to_tiff(png_path: Path, tiff_path: Path) -> None:
    """Convert PNG to TIFF at TIFF_DPI, preserving RGB."""
    img = Image.open(png_path).convert("RGB")
    img.save(tiff_path, format="TIFF", dpi=(TIFF_DPI, TIFF_DPI), compression="tiff_lzw")
    print(f"  [saved] {tiff_path.name}  ({tiff_path.stat().st_size // 1024} KB)")


def main() -> None:
    print("=" * 55)
    print(" DocVerify — High-Resolution Figure Export")
    print("=" * 55)

    HIRES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Apply IEEE style ──────────────────────────────────────────────────────
    IEEEStyle.apply()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Loading experiment data...")
    data = load_paper_data()

    # ── EPS: matplotlib figures ───────────────────────────────────────────────
    print(f"\n[2/3] Exporting matplotlib figures as EPS -> {HIRES_DIR}")
    print("-" * 55)
    generators = [
        ("fig1_pareto_front.eps",     ParetoFrontFigure(data)),
        ("fig2_nested_cv.eps",        NestedCVFigure(data)),
        ("fig3_ablation.eps",         AblationFigure(data)),
        ("fig4_challenge.eps",        ChallengeFigure(data)),
        ("fig5_scalar_vs_pareto.eps", ScalarVsPareto(data)),
    ]
    for filename, gen in generators:
        save_eps(gen, filename)

    # ── EPS: architecture SVG ─────────────────────────────────────────────────
    svg_path = OUTPUT_DIR / "fig0_architecture_compact.svg"
    eps_path = HIRES_DIR / "fig0_architecture_compact.eps"
    print(f"\n  Converting SVG architecture figure to EPS...")
    try:
        convert_svg_to_eps(svg_path, eps_path)
    except Exception as e:
        print(f"  [WARNING] Could not convert SVG: {e}")
        print(f"            Skipping {eps_path.name}")

    # ── TIFF: qualitative PNG figures ─────────────────────────────────────────
    print(f"\n[3/3] Converting qualitative figures to TIFF ({TIFF_DPI} DPI) -> {HIRES_DIR}")
    print("-" * 55)
    qual_figures = [
        "fig_qual_attack_input.png",
        "fig_qual_attack_mask.png",
        "fig_qual_attack_overlay.png",
    ]
    for png_name in qual_figures:
        png_path  = FIGURES_DIR / png_name
        tiff_name = png_path.stem + ".tiff"
        tiff_path = HIRES_DIR / tiff_name
        if png_path.exists():
            convert_png_to_tiff(png_path, tiff_path)
        else:
            print(f"  [WARNING] Not found: {png_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    exported = sorted(HIRES_DIR.iterdir())
    print(f"\nOK {len(exported)} files exported to: {HIRES_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
