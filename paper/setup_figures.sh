#!/bin/bash
# setup_figures.sh — Copy PDF figures into paper/figures/
# Run from the paper/ directory:
#   cd paper && bash setup_figures.sh
#
# Requires Inkscape for the SVG → PDF conversion of the architecture figure.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIG_SRC="$SCRIPT_DIR/../paper_figures/output"
DEST="$SCRIPT_DIR/figures"

mkdir -p "$DEST"

echo "Copying PDF figures..."
cp "$FIG_SRC/fig1_pareto_front.pdf"    "$DEST/"
cp "$FIG_SRC/fig2_nested_cv.pdf"       "$DEST/"
cp "$FIG_SRC/fig3_ablation.pdf"        "$DEST/"
cp "$FIG_SRC/fig4_challenge.pdf"       "$DEST/"
cp "$FIG_SRC/fig5_scalar_vs_pareto.pdf" "$DEST/"

echo "Converting architecture SVG → PDF (requires Inkscape)..."
if command -v inkscape &>/dev/null; then
    inkscape --export-type=pdf \
             --export-filename="$DEST/fig0_architecture.pdf" \
             "$FIG_SRC/fig0_architecture_compact.svg"
    echo "Architecture figure converted."
else
    echo "WARNING: Inkscape not found. Convert manually:"
    echo "  inkscape --export-type=pdf --export-filename=figures/fig0_architecture.pdf \\"
    echo "           ../paper_figures/output/fig0_architecture_compact.svg"
    echo ""
    echo "Or use an online converter: https://cloudconvert.com/svg-to-pdf"
fi

echo ""
echo "Done. figures/ contains:"
ls "$DEST/"
