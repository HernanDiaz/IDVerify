"""
pareto_flow_svg.py — Pareto multi-objective HPO methodology flowchart.

IEEE single-column (≈3.5 in wide) × ~4 in.  Standalone SVG generator.

Run:
    python paper_figures/figures/pareto_flow_svg.py
Output:
    paper_figures/output/fig_pareto_flow.svg
"""

from pathlib import Path

# ── Layout (typographic points, 1 pt = 1/72 in) ──────────────────────────────
BW   = 190   # box width
BH   = 27    # box height (title + subtitle fit inside)
ARW  = 14    # arrow-zone height between consecutive boxes
MX   = 10    # left / right canvas margin
MY   = 12    # top / bottom canvas margin
CW   = BW + 2 * MX   # canvas width  (= 210 pt  ≈ 2.92 in)

FONT = "'Times New Roman', Georgia, serif"
FS_T = 7.2   # box title
FS_B = 6.2   # box subtitle
FS_H = 8.5   # section headers (unused here, kept for consistency)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DATA   = "#D6EAF8"   # input data
C_ARCH   = "#D5F5E3"   # architecture
C_HPO    = "#FDEBD0"   # HPO / optimisation
C_PARETO = "#FAD7A0"   # Pareto front
C_SEL    = "#FADBD8"   # model selection
C_CV     = "#D6EAF8"   # nested CV evaluation
C_BT     = "#F4ECF7"   # blind test
C_EDGE   = "#2C3E50"
C_ARR    = "#2C3E50"

# ── Pipeline stages  (color, bold title, light subtitle) ──────────────────────
STAGES = [
    (C_DATA,   "Document Image Dataset",
               "4 000 images  ·  forgery / tampering"),
    (C_ARCH,   "Multi-task U-Net Architecture",
               "Shared encoder  ·  decoder  ·  cls head"),
    (C_HPO,    "Multi-objective HPO  (Optuna TPE)",
               "Obj 1: PR-AUC        Obj 2: Dice Global"),
    (C_PARETO, "Pareto Front",
               "51 non-dominated configurations"),
    (C_SEL,    "Model Selection",
               "Min. Euclidean dist. to ideal point (1, 1)"),
    (C_CV,     "Nested CV Evaluation",
               "10-fold outer  ×  5-fold inner HPO"),
    (C_BT,     "Blind Test",
               "30 independent random seeds"),
]

# ── SVG primitive helpers ─────────────────────────────────────────────────────
def _r(x, y, w, h, fill, stroke=C_EDGE, lw=0.7):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}" rx="0"/>')


def _t(cx, cy, text, size=FS_T, weight="normal", fill=C_EDGE, anchor="middle"):
    return (f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{text}</text>')


def _av(x, y1, y2):
    """Vertical arrow with arrowhead marker."""
    ye = y2 - 5
    return (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{ye:.1f}" '
            f'stroke="{C_ARR}" stroke-width="0.8" marker-end="url(#ah)"/>')


# ── Main builder ──────────────────────────────────────────────────────────────
def build_svg() -> tuple[str, float, float]:
    n  = len(STAGES)
    CH = MY + n * BH + (n - 1) * ARW + MY
    cx = MX + BW / 2

    lines: list[str] = []
    w_in = CW / 72
    h_in = CH / 72

    lines += [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_in:.3f}in" height="{h_in:.3f}in" '
        f'viewBox="0 0 {CW} {CH}">',
        # Arrowhead marker
        f'<defs>'
        f'<marker id="ah" markerWidth="6" markerHeight="5" '
        f'refX="5.5" refY="2.5" orient="auto">'
        f'<path d="M0,0 L0,5 L6,2.5 z" fill="{C_ARR}"/>'
        f'</marker>'
        f'</defs>',
        # White background
        f'<rect x="0" y="0" width="{CW}" height="{CH}" fill="white"/>',
    ]

    for i, (color, title, sub) in enumerate(STAGES):
        y = MY + i * (BH + ARW)
        lines.append(_r(MX, y, BW, BH, color))
        lines.append(_t(cx, y + BH * 0.32, title, weight="bold"))
        lines.append(_t(cx, y + BH * 0.70, sub, size=FS_B, fill="#444444"))
        if i < n - 1:
            lines.append(_av(cx, y + BH, y + BH + ARW))

    lines.append('</svg>')
    return '\n'.join(lines), w_in, h_in


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    out_dir = Path(__file__).parent.parent / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    svg, w, h = build_svg()
    out = out_dir / 'fig_pareto_flow.svg'
    out.write_text(svg, encoding='utf-8')
    print(f'[saved] {out}\n  physical size: {w:.3f}in × {h:.3f}in')
