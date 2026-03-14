"""
nested_cv_svg.py — Nested cross-validation protocol diagram.

IEEE double-column (≈7.16 in wide) × ~3.2 in.  Standalone SVG generator.

Layout
------
Left panel  : Outer K=10 fold CV  – coloured fold-strip grid
Right panel : Inner loop detail   – 4 stacked boxes (inner CV → HPO → Pareto → select)
Bottom strip: Evaluation + aggregation (full width)

Run:
    python paper_figures/figures/nested_cv_svg.py
Output:
    paper_figures/output/fig_nested_cv.svg
"""

from pathlib import Path

# ── Canvas & margin ───────────────────────────────────────────────────────────
CW  = 515    # canvas width  (pt)  ≈ 7.15 in
MX  = 8      # left/right margin
MY  = 14     # top/bottom margin

FONT = "'Times New Roman', Georgia, serif"
FS_T  = 7.0   # box title
FS_B  = 6.2   # box subtitle / small label
FS_LB = 6.0   # tiny fold-row labels
FS_H  = 7.8   # panel header

# ── Column geometry ───────────────────────────────────────────────────────────
LEFT_W  = 236   # width of left panel content (fold grid + labels)
MID_GAP = 28    # gap between panels (arrow zone)
RIGHT_X = MX + LEFT_W + MID_GAP          # right panel start x
RIGHT_W = CW - RIGHT_X - MX              # right panel width  ≈ 235 pt

# ── Fold-grid geometry ────────────────────────────────────────────────────────
K_OUT   = 10    # outer folds
LAB_W   = 30    # width reserved for "Fold i" label
SG      = 1.0   # gap between segments within a row
SW      = (LEFT_W - LAB_W - K_OUT * SG + SG) / K_OUT   # segment width ≈ 19.5 pt
RH      = 8     # fold-row height
RG      = 2.5   # gap between fold rows

# ── Inner-loop box geometry ───────────────────────────────────────────────────
IB_H  = 24   # inner box height
IB_W  = RIGHT_W
IA_H  = 13   # arrow zone between inner boxes

# ── Bottom strip geometry ─────────────────────────────────────────────────────
BOT_H = 26   # bottom aggregation box height
BOT_G = 14   # gap above bottom box (arrow)

# ── Colours ───────────────────────────────────────────────────────────────────
C_TRAIN  = "#AED6F1"   # outer train segment
C_TEST   = "#F1948A"   # outer test segment
C_HPAN   = "#EBF5FB"   # panel background (very light)
C_ICV    = "#D6EAF8"   # inner CV box
C_IHPO   = "#FDEBD0"   # Optuna HPO box
C_IPAR   = "#D5F5E3"   # Pareto box
C_ISEL   = "#FAD7A0"   # selection box
C_BOT    = "#F4ECF7"   # bottom aggregation box
C_EDGE   = "#2C3E50"
C_ARR    = "#2C3E50"
C_PHEAD  = "#1A5276"   # panel header text colour

# ── Inner-loop stages ─────────────────────────────────────────────────────────
INNER_STAGES = [
    (C_ICV,  "5-Fold Inner CV  (on outer train split)",
             "stratified · repeated per Optuna trial"),
    (C_IHPO, "Multi-objective HPO  ·  Optuna TPE",
             "Obj 1: PR-AUC    Obj 2: Dice Global"),
    (C_IPAR, "Pareto Front Construction",
             "retain non-dominated configurations"),
    (C_ISEL, "Best Configuration Selection",
             "min. Euclidean distance to ideal (1, 1)"),
]

# ── SVG helpers ───────────────────────────────────────────────────────────────
def _r(x, y, w, h, fill, stroke=C_EDGE, lw=0.6, rx=0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}" rx="{rx}"/>')


def _t(cx, cy, text, size=FS_T, weight="normal", fill=C_EDGE, anchor="middle"):
    return (f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{text}</text>')


def _av(x, y1, y2, lw=0.8, color=C_ARR, dashed=False):
    """Vertical downward arrow."""
    ye  = y2 - 5
    dsh = ' stroke-dasharray="3,2"' if dashed else ''
    return (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{ye:.1f}" '
            f'stroke="{color}" stroke-width="{lw}"{dsh} marker-end="url(#ah)"/>')


def _ah(x1, y, x2, lw=0.8, color=C_ARR):
    """Horizontal rightward arrow."""
    xe = x2 - 5
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{xe:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{lw}" marker-end="url(#ah)"/>')


# ── Builder ───────────────────────────────────────────────────────────────────
def build_svg() -> tuple[str, float, float]:

    # ── vertical positions ────────────────────────────────────────────────────
    hdr_y    = MY                          # panel headers baseline
    hdr_h    = 12                          # header bar height
    grid_y   = hdr_y + hdr_h + 5          # fold grid top
    grid_h   = K_OUT * RH + (K_OUT - 1) * RG
    leg_y    = grid_y + grid_h + 6        # legend below grid
    leg_h    = 9

    inner_y  = hdr_y + hdr_h + 5          # inner boxes top (same baseline as grid)
    inner_h  = len(INNER_STAGES) * IB_H + (len(INNER_STAGES) - 1) * IA_H

    main_bot = max(leg_y + leg_h, inner_y + inner_h)   # bottom of main section

    bot_y    = main_bot + BOT_G            # aggregation box top
    CH       = bot_y + BOT_H + MY         # total canvas height

    w_in = CW / 72
    h_in = CH / 72

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_in:.3f}in" height="{h_in:.3f}in" '
        f'viewBox="0 0 {CW} {CH}">',
        f'<defs>'
        f'<marker id="ah" markerWidth="6" markerHeight="5" '
        f'refX="5.5" refY="2.5" orient="auto">'
        f'<path d="M0,0 L0,5 L6,2.5 z" fill="{C_ARR}"/>'
        f'</marker>'
        f'</defs>',
        f'<rect x="0" y="0" width="{CW}" height="{CH}" fill="white"/>',
    ]

    # ── LEFT PANEL: outer fold grid ───────────────────────────────────────────

    # Panel header bar
    lines.append(_r(MX, hdr_y, LEFT_W, hdr_h, C_HPAN, lw=0.5))
    lines.append(_t(MX + LEFT_W / 2, hdr_y + hdr_h / 2,
                    "OUTER LOOP  (K = 10)", size=FS_H, weight="bold",
                    fill=C_PHEAD))

    # Fold rows
    for k in range(K_OUT):
        row_y = grid_y + k * (RH + RG)
        # "Fold k+1" label
        lines.append(_t(MX + LAB_W - 3, row_y + RH / 2,
                        f"Fold {k + 1}", size=FS_LB, anchor="end"))
        # 10 segments
        for j in range(K_OUT):
            sx = MX + LAB_W + j * (SW + SG)
            color = C_TEST if j == k else C_TRAIN
            lines.append(_r(sx, row_y, SW, RH, color, lw=0.3))

    # Legend (below grid)
    lx = MX + LAB_W
    sq = 7
    lines.append(_r(lx, leg_y, sq, sq, C_TRAIN, lw=0.3))
    lines.append(_t(lx + sq + 3, leg_y + sq / 2,
                    "Outer train (9 folds)", size=FS_LB, anchor="start"))
    lx2 = lx + 95
    lines.append(_r(lx2, leg_y, sq, sq, C_TEST, lw=0.3))
    lines.append(_t(lx2 + sq + 3, leg_y + sq / 2,
                    "Outer test (1 fold)", size=FS_LB, anchor="start"))

    # ── HORIZONTAL ARROW  left panel → right panel ────────────────────────────
    arr_y   = hdr_y + hdr_h + inner_h / 2   # vertical centre of inner boxes
    arr_x1  = MX + LEFT_W
    arr_x2  = RIGHT_X
    lines.append(_ah(arr_x1, arr_y, arr_x2))

    # ── RIGHT PANEL: inner loop ───────────────────────────────────────────────
    rcx = RIGHT_X + IB_W / 2

    # Panel header bar
    lines.append(_r(RIGHT_X, hdr_y, IB_W, hdr_h, C_HPAN, lw=0.5))
    lines.append(_t(rcx, hdr_y + hdr_h / 2,
                    "INNER LOOP  (per outer fold)", size=FS_H, weight="bold",
                    fill=C_PHEAD))

    for i, (color, title, sub) in enumerate(INNER_STAGES):
        by = inner_y + i * (IB_H + IA_H)
        lines.append(_r(RIGHT_X, by, IB_W, IB_H, color))
        lines.append(_t(rcx, by + IB_H * 0.32, title, weight="bold"))
        lines.append(_t(rcx, by + IB_H * 0.70, sub, size=FS_B, fill="#444444"))
        if i < len(INNER_STAGES) - 1:
            lines.append(_av(rcx, by + IB_H, by + IB_H + IA_H))

    # ── ARROW  inner loop → bottom strip ─────────────────────────────────────
    # Downward from bottom of last inner box, midpoint horizontally
    last_inner_bot = inner_y + len(INNER_STAGES) * IB_H + (len(INNER_STAGES) - 1) * IA_H
    bot_cx = CW / 2
    # Polyline: right-panel bottom → down → centre → bottom box
    rx_mid  = rcx
    turn_y  = last_inner_bot + BOT_G / 2
    lines.append(
        f'<polyline points="'
        f'{rx_mid:.1f},{last_inner_bot:.1f} '
        f'{rx_mid:.1f},{turn_y:.1f} '
        f'{bot_cx:.1f},{turn_y:.1f} '
        f'{bot_cx:.1f},{bot_y - 5:.1f}" '
        f'fill="none" stroke="{C_ARR}" stroke-width="0.8" '
        f'marker-end="url(#ah)"/>'
    )

    # ── BOTTOM STRIP: evaluation + aggregation ────────────────────────────────
    lines.append(_r(MX, bot_y, CW - 2 * MX, BOT_H, C_BOT))
    lines.append(_t(bot_cx, bot_y + BOT_H * 0.30,
                    "Retrain selected config on full outer train  →  evaluate on outer test fold",
                    weight="bold"))
    lines.append(_t(bot_cx, bot_y + BOT_H * 0.70,
                    "Aggregate K=10 results  →  report mean ± std  (PR-AUC, Dice Global)",
                    size=FS_B, fill="#444444"))

    lines.append('</svg>')
    return '\n'.join(lines), w_in, h_in


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    out_dir = Path(__file__).parent.parent / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    svg, w, h = build_svg()
    out = out_dir / 'fig_nested_cv.svg'
    out.write_text(svg, encoding='utf-8')
    print(f'[saved] {out}\n  physical size: {w:.3f}in × {h:.3f}in')
