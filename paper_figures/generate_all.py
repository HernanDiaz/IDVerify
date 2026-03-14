"""
generate_all.py — Entry point: generates all paper figures as vector PDFs.

Usage:
    cd E:\\PycharmProjects\\DocVerify
    venv\\Scripts\\python paper_figures\\generate_all.py

Output:
    paper_figures/output/fig1_pareto_front.pdf
    paper_figures/output/fig2_nested_cv.pdf
    paper_figures/output/fig3_ablation.pdf
    paper_figures/output/fig4_challenge.pdf
    paper_figures/output/fig5_scalar_vs_pareto.pdf
"""

import sys
from pathlib import Path

# Allow imports from paper_figures/ without installing as a package
sys.path.insert(0, str(Path(__file__).parent))

from config import IEEEStyle, OUTPUT_DIR
from data_loader import load_paper_data
from figures.pareto_front     import ParetoFrontFigure
from figures.nested_cv        import NestedCVFigure
from figures.ablation         import AblationFigure
from figures.challenge        import ChallengeFigure
from figures.scalar_vs_pareto import ScalarVsPareto


def main() -> None:
    print("=" * 55)
    print(" DocVerify — Paper Figure Generator")
    print("=" * 55)

    # Apply IEEE style globally before any figure is created
    IEEEStyle.apply()

    # Load all data once (Dependency Inversion: inject into each figure)
    print("\n[1/2] Loading experiment data...")
    data = load_paper_data()
    print(f"  pareto_front  : {len(data.pareto_front)} trials")
    print(f"  all_trials    : {len(data.all_trials)} trials")
    print(f"  nested_cv     : {len(data.nested_cv)} outer folds")
    print(f"  blind_test    : {len(data.blind_test)} rows (seeds × variants)")
    print(f"  stat_tests    : {len(data.stat_tests)} comparisons")
    print(f"  challenge     : {len(data.challenge_detail)} seed rows")
    print(f"  scalar        : {len(data.scalar_selected)} rows")

    # Define figures: (filename, generator_class)
    figures = [
        ("fig1_pareto_front.pdf",     ParetoFrontFigure(data)),
        ("fig2_nested_cv.pdf",        NestedCVFigure(data)),
        ("fig3_ablation.pdf",         AblationFigure(data)),
        ("fig4_challenge.pdf",        ChallengeFigure(data)),
        ("fig5_scalar_vs_pareto.pdf", ScalarVsPareto(data)),
    ]

    print(f"\n[2/2] Generating {len(figures)} figures -> {OUTPUT_DIR}")
    print("-" * 55)
    for filename, generator in figures:
        output_path = OUTPUT_DIR / filename
        generator.save(output_path)

    print("-" * 55)
    print(f"\nOK All figures saved to: {OUTPUT_DIR.resolve()}")
    print("  Open any .pdf in a viewer to inspect quality.\n")


if __name__ == "__main__":
    main()
