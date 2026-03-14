"""
architecture_svg.py — DocVerify architecture diagram (Patel box style).

Generates paper_figures/output/fig0_architecture.svg
Two-row layout: encoder (left→right, top), decoder (right→left, bottom).
Skip connections drawn as vertical dashed arrows between rows.
Segmentation head to the LEFT of decoder block 5.
Classification head to the RIGHT of enc6/bottleneck.
Run standalone:  python architecture_svg.py
"""

from pathlib import Path

# ── Palette ───────────────────────────────────────────────────────────────────
C_ENC    = "#D6EAF8"   # encoder boxes – light blue
C_DEC    = "#FDEBD0"   # decoder boxes – light orange
C_CLS    = "#FADBD8"   # classification / segmentation heads – light red
C_BN_BOX = "#D5F5E3"   # bottleneck – light green
C_POOL   = "#EAECEE"   # pool/upsample boxes – light grey
C_IO     = "#F2F3F4"   # input / output result boxes
C_EDGE   = "#2C3E50"   # box border
C_SKIP   = "#7F8C8D"   # skip connection arrows
C_ARR    = "#2C3E50"   # main flow arrows
C_DIM    = "#555555"   # dimension label colour

# ── Box geometry (px) ─────────────────────────────────────────────────────────
BW      = 138    # operation box width
BH      = 25     # operation box height
PH      = 27     # pool / upsample box height
HGAP    = 48     # horizontal gap between stage groups (arrow zone)
ROW_GAP = 90     # vertical gap between encoder and decoder rows
MX      = 18     # left margin
MY      = 22     # top margin
FONT    = "'Times New Roman', Georgia, serif"

# ── Architecture definition ───────────────────────────────────────────────────
ENCODER = [
    {   # Block 1
        "title": "Encoder Block 1",
        "ops":  ["Conv2D  8 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": None,
        "skip_label": None,          # no skip from enc1
        "out": "224×224×8",
    },
    {   # Block 2  (skip s224)
        "title": "Encoder Block 2",
        "ops":  ["Conv2D 16 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 16 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": "AvgPool2D (2×2)",
        "skip_label": "s224: 224×224×16",
        "out": "112×112×16",
    },
    {   # Block 3  (skip s112)
        "title": "Encoder Block 3",
        "ops":  ["Conv2D 32 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 32 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 32 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": "AvgPool2D (2×2)",
        "skip_label": "s112: 112×112×32",
        "out": "56×56×32",
    },
    {   # Block 4  (skip s56)
        "title": "Encoder Block 4",
        "ops":  ["Conv2D 64 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 64 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 64 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 64 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": "AvgPool2D (2×2)",
        "skip_label": "s56: 56×56×64",
        "out": "28×28×64",
    },
    {   # Block 5  (skip s28)
        "title": "Encoder Block 5",
        "ops":  ["Conv2D 128 filt. (5×5)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": "MaxPool2D (2×2)",
        "skip_label": "s28: 28×28×128",
        "out": "14×14×128",
    },
    {   # Block 6  (skip s14)
        "title": "Encoder Block 6",
        "ops":  ["Conv2D 256 filt. (5×5)",
                 "LeakyReLU (α=0.2)",
                 "BatchNorm2D"],
        "pool": "MaxPool2D (2×2)",
        "skip_label": "s14: 14×14×256",
        "out": "7×7×256",
    },
]

# Bottleneck box (drawn to the right of enc6, at encoder row level)
BOTTLENECK_OPS = ["AdaptiveAvgPool2D(1)", "Bottleneck  7×7×256"]

# Decoder stages (dec14 first, dec224 last; flow is right → left)
DECODER = [
    {   "title": "Decoder Block 1  (7→14)",
        "ops":  ["Conv2D 128 filt. (1×1)",
                 "LeakyReLU (α=0.2)",
                 "Upsample ×2 (bilinear)",
                 "Concat  s14  (→384ch)",
                 "Conv2D 128 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D 128 filt. (3×3)",
                 "LeakyReLU (α=0.2)"],
        "out": "14×14×128",
        "enc_skip": 5,
    },
    {   "title": "Decoder Block 2  (14→28)",
        "ops":  ["Upsample ×2 (bilinear)",
                 "Concat  s28  (→256ch)",
                 "Conv2D  64 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D  64 filt. (3×3)",
                 "LeakyReLU (α=0.2)"],
        "out": "28×28×64",
        "enc_skip": 4,
    },
    {   "title": "Decoder Block 3  (28→56)",
        "ops":  ["Upsample ×2 (bilinear)",
                 "Concat  s56  (→128ch)",
                 "Conv2D  32 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D  32 filt. (3×3)",
                 "LeakyReLU (α=0.2)"],
        "out": "56×56×32",
        "enc_skip": 3,
    },
    {   "title": "Decoder Block 4  (56→112)",
        "ops":  ["Upsample ×2 (bilinear)",
                 "Concat s112  (→64ch)",
                 "Conv2D  16 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D  16 filt. (3×3)",
                 "LeakyReLU (α=0.2)"],
        "out": "112×112×16",
        "enc_skip": 2,
    },
    {   "title": "Decoder Block 5  (112→224)",
        "ops":  ["Upsample ×2 (bilinear)",
                 "Concat s224  (→32ch)",
                 "Conv2D   8 filt. (3×3)",
                 "LeakyReLU (α=0.2)",
                 "Conv2D   8 filt. (3×3)",
                 "LeakyReLU (α=0.2)"],
        "out": "224×224×8",
        "enc_skip": 1,
    },
]

CLS_HEAD_OPS = [
    "Dropout",
    "Linear (256 → 32)",
    "LeakyReLU (α=0.2)",
    "Dropout",
    "Linear (32 → 16)",
    "LeakyReLU (α=0.2)",
    "Dropout",
    "Linear (16 → 16)",
    "LeakyReLU (α=0.2)",
    "Dropout",
    "Linear (16 → 1)",
]

SEG_HEAD_OPS = [
    "Conv2D 1 filt. (1×1)",
    "Interpolate (bilinear)",
]


# ── SVG primitives ────────────────────────────────────────────────────────────

def _rect(x, y, w, h, fill, stroke=C_EDGE, sw=0.9, rx=2):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def _text(x, y, s, size=7.5, weight="normal", fill="#111111", anchor="middle"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'font-family="{FONT}">{s}</text>')


def _arrow_h(x1, y, x2, color=C_ARR, lw=1.1):
    """Horizontal arrow with arrowhead at x2. Works for both left and right directions."""
    if x2 >= x1:
        x_end = x2 - 7   # right-pointing: stop before tip
    else:
        x_end = x2 + 7   # left-pointing: stop before tip
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x_end:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{lw}" marker-end="url(#ah)"/>')


def _arrow_v(x, y1, y2, color=C_ARR, lw=1.0, dashed=False):
    dash = ' stroke-dasharray="5,3"' if dashed else ''
    direction = 1 if y2 > y1 else -1
    return (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2 - direction * 7:.1f}" '
            f'stroke="{color}" stroke-width="{lw}"{dash} marker-end="url(#ah)"/>')


def _dim_label(x, y, s, size=6.8):
    return _text(x, y, s, size=size, fill=C_DIM)


# ── Stage drawing ─────────────────────────────────────────────────────────────

def _draw_stage(ops, x, y_top, fill=C_ENC, pool_label=None, title=None):
    """Draw stacked op boxes. Returns list of SVG elements and bottom y."""
    parts = []
    cy = y_top
    if title:
        title_fill = "#C5D9E8" if fill == C_ENC else (
            "#F0C9A0" if fill == C_DEC else (
            "#C9E8C5" if fill == C_BN_BOX else "#E8C5C5"))
        parts.append(_rect(x, cy, BW, 20, title_fill, stroke=C_EDGE, sw=0.9))
        parts.append(_text(x + BW / 2, cy + 10, title, size=6.8, weight="bold"))
        cy += 20
    for op in ops:
        parts.append(_rect(x, cy, BW, BH, fill))
        parts.append(_text(x + BW / 2, cy + BH / 2, op, size=7.2))
        cy += BH
    if pool_label:
        parts.append(_rect(x, cy, BW, PH, C_POOL))
        parts.append(_text(x + BW / 2, cy + PH / 2, pool_label, size=7.2))
        cy += PH
    return parts, cy


def stage_h(ops, pool=None, title=True):
    return (20 if title else 0) + len(ops) * BH + (PH if pool else 0)


# ── Main SVG builder ──────────────────────────────────────────────────────────

def build_svg() -> str:
    parts = []

    # ── X positions ────────────────────────────────────────────────────────
    # Encoder blocks start at MX (input box now sits below enc1, not to its left)
    enc_x = [MX + i * (BW + HGAP) for i in range(6)]
    # dec[i] is below enc[5-i]:  dec[0]↔enc[5], ..., dec[4]↔enc[1]
    dec_x  = [enc_x[5 - i] for i in range(5)]

    # Bottleneck & classification head: to the RIGHT of enc6
    CLS_X   = enc_x[5] + BW + HGAP + 10
    CANVAS_W = CLS_X + BW + 30

    # Segmentation head: to the LEFT of dec[4] (= enc_x[1] column)
    SEG_X = dec_x[4] - (BW + HGAP)   # = enc_x[0]

    # ── Row heights ─────────────────────────────────────────────────────────
    enc_heights = [stage_h(e["ops"], e["pool"]) for e in ENCODER]
    max_enc_h   = max(enc_heights)

    R1_TOP = MY
    R1_BOT = R1_TOP + max_enc_h + 10

    R2_TOP = R1_BOT + ROW_GAP

    dec_heights  = [stage_h(d["ops"]) for d in DECODER]
    max_dec_h    = max(dec_heights)
    min_dec_h    = min(dec_heights)

    # Bottom-align all decoder blocks: tallest starts at R2_TOP, shorter ones lower
    dec_bot_y    = R2_TOP + min_dec_h                    # shared bottom y
    dec_y_starts = [dec_bot_y - h for h in dec_heights]  # per-block top y
    R2_BOT       = dec_bot_y + 10

    # ── Bottleneck arrow y (used for ALL encoder horizontal arrows) ─────────
    BN_Y    = R1_TOP
    bn_arr_y = BN_Y + (20 + len(BOTTLENECK_OPS) * BH) / 2   # centre of bottleneck

    # ── Classification head vertical positions ──────────────────────────────
    bn_h    = stage_h(BOTTLENECK_OPS, title=True)
    CLS_TOP = BN_Y + bn_h + 20
    cls_h   = stage_h(CLS_HEAD_OPS, title=True)
    CLS_BOT = CLS_TOP + cls_h
    SCORE_Y = CLS_BOT + 15

    # ── Segmentation head vertical positions ────────────────────────────────
    seg_h_px = stage_h(SEG_HEAD_OPS, title=True)
    seg_arr_y = R2_TOP + seg_h_px / 2    # y for horizontal arrow dec[4]→seg head
    SEG_BOT   = R2_TOP + seg_h_px
    MASK_Y    = SEG_BOT + 15

    # Canvas bottom flush with decoder/legend bottom + one top-margin worth of padding
    CANVAS_H = dec_bot_y + MY

    # ── SVG header ──────────────────────────────────────────────────────────
    parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{CANVAS_W}" height="{CANVAS_H}"
     viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="{C_ARR}"/>
    </marker>
    <marker id="ahs" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="{C_SKIP}"/>
    </marker>
  </defs>
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="white"/>
''')

    # ── INPUT box — placed below Encoder Block 1, upward arrow ────────────
    enc1_bot = R1_TOP + enc_heights[0]          # bottom of enc1 (no pool)
    inp_y    = enc1_bot + 50                    # 50 px gap below enc1
    inp_x    = enc_x[0]                         # same column as enc1
    parts.append(_rect(inp_x, inp_y, BW, 28, C_IO))
    parts.append(_text(inp_x + BW / 2, inp_y + 14,
                       "Input  224×224×3", size=7.5, weight="bold"))
    # Upward arrow from input top → enc1 bottom
    parts.append(_arrow_v(inp_x + BW / 2, inp_y, enc1_bot))
    parts.append(_dim_label(inp_x + BW / 2 + 34, (inp_y + enc1_bot) / 2, "224×224×3"))

    # ── ENCODER STAGES ─────────────────────────────────────────────────────
    enc_box_bottoms = []

    for i, enc in enumerate(ENCODER):
        x = enc_x[i]
        stg, bottom_y = _draw_stage(enc["ops"], x, R1_TOP,
                                     fill=C_ENC, pool_label=enc["pool"],
                                     title=enc["title"])
        parts.extend(stg)
        enc_box_bottoms.append(bottom_y)

        # Horizontal arrow to next stage (all at bn_arr_y level)
        if i < 5:
            x_end = enc_x[i + 1]
            parts.append(_arrow_h(x + BW, bn_arr_y, x_end))
            parts.append(_dim_label((x + BW + x_end) / 2, bn_arr_y - 8, enc["out"]))

    # ── BOTTLENECK ─────────────────────────────────────────────────────────
    bn_x = CLS_X
    bn_parts, bn_bot = _draw_stage(BOTTLENECK_OPS, bn_x, BN_Y,
                                    fill=C_BN_BOX, title="Bottleneck")
    parts.extend(bn_parts)

    # Arrow enc6 → bottleneck (at bn_arr_y, same level as all encoder arrows)
    parts.append(_arrow_h(enc_x[5] + BW, bn_arr_y, bn_x))
    parts.append(_dim_label((enc_x[5] + BW + bn_x) / 2, bn_arr_y - 8,
                              ENCODER[5]["out"]))

    # ── CLASSIFICATION HEAD ────────────────────────────────────────────────
    parts.append(_arrow_v(bn_x + BW / 2, bn_bot, CLS_TOP))
    parts.append(_dim_label(bn_x + BW / 2 + 32, (bn_bot + CLS_TOP) / 2, "256-d vector"))

    cls_parts, cls_bot = _draw_stage(CLS_HEAD_OPS, bn_x, CLS_TOP,
                                      fill=C_CLS, title="Classification Head")
    parts.extend(cls_parts)

    # Score output box
    parts.append(_arrow_v(bn_x + BW / 2, cls_bot, SCORE_Y))
    parts.append(_rect(bn_x, SCORE_Y, BW, 28, C_IO))
    parts.append(_text(bn_x + BW / 2, SCORE_Y + 14,
                        "Score  (sigmoid)", size=7.5, weight="bold", fill="#8B0000"))

    # ── DECODER STAGES (flow: right → left) ────────────────────────────────
    # Vertical arrow from enc6 column bottom → dec[0] top (bottom-aligned position)
    dec_arr_x0 = dec_x[0] + BW / 2
    parts.append(_arrow_v(dec_arr_x0, R1_BOT, dec_y_starts[0], color=C_ARR, lw=1.1))
    parts.append(_dim_label(dec_arr_x0 + 32, (R1_BOT + dec_y_starts[0]) / 2, "7×7×256"))

    for i, dec in enumerate(DECODER):
        x = dec_x[i]
        stg, _ = _draw_stage(dec["ops"], x, dec_y_starts[i], fill=C_DEC, title=dec["title"])
        parts.extend(stg)

        # Arrow to next decoder block — goes RIGHT-TO-LEFT
        # Use centre of the shared bottom-aligned zone (same for all inter-block arrows)
        if i < 4:
            arr_y2 = dec_bot_y - min_dec_h / 2  # centre of all bottom-aligned blocks
            parts.append(_arrow_h(dec_x[i], arr_y2, dec_x[i + 1] + BW))
            mid_x = (dec_x[i] + dec_x[i + 1] + BW) / 2
            parts.append(_dim_label(mid_x, arr_y2 - 8, dec["out"]))

    # ── SEGMENTATION HEAD (left of dec[4]) ─────────────────────────────────
    # Horizontal arrow from dec[4] left side → seg head right side
    parts.append(_arrow_h(dec_x[4], seg_arr_y, SEG_X + BW))
    parts.append(_dim_label((dec_x[4] + SEG_X + BW) / 2, seg_arr_y - 8,
                              DECODER[4]["out"]))

    seg_parts, seg_bot = _draw_stage(SEG_HEAD_OPS, SEG_X, R2_TOP,
                                      fill=C_CLS, title="Segmentation Head")
    parts.extend(seg_parts)

    # Mask output box below segmentation head
    mask_xc = SEG_X + BW / 2
    parts.append(_arrow_v(mask_xc, seg_bot, MASK_Y))
    parts.append(_rect(SEG_X, MASK_Y, BW, 28, C_IO))
    parts.append(_text(mask_xc, MASK_Y + 14,
                        "Mask  224×224×1", size=7.5, weight="bold", fill="#00008B"))

    # ── SKIP CONNECTIONS (enc → dec, dashed vertical arrows) ───────────────
    # enc[1]→dec[4], enc[2]→dec[3], enc[3]→dec[2], enc[4]→dec[1], enc[5]→dec[0]
    skip_pairs = [(1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
    for ei, di in skip_pairs:
        sx = enc_x[ei] + BW / 2
        y_enc_bot = enc_box_bottoms[ei]
        y_dec_top = dec_y_starts[di]   # use each block's actual (bottom-aligned) top
        parts.append(
            f'<line x1="{sx:.1f}" y1="{y_enc_bot:.1f}" '
            f'x2="{sx:.1f}" y2="{y_dec_top - 7:.1f}" '
            f'stroke="{C_SKIP}" stroke-width="1.0" stroke-dasharray="5,3" '
            f'marker-end="url(#ahs)"/>'
        )
        lbl = ENCODER[ei]["skip_label"]
        if lbl:
            parts.append(_dim_label(sx + 32, (y_enc_bot + y_dec_top) / 2, lbl, size=6.3))

    # ── ROW LABELS ─────────────────────────────────────────────────────────
    enc_label_x = enc_x[0] + 3 * (BW + HGAP)
    parts.append(_text(enc_label_x, R1_TOP - 12,
                        "PATEL CNN ENCODER", size=9, weight="bold", fill="#1A5276"))
    # Same x as PATEL CNN ENCODER, at decoder row level
    parts.append(_text(enc_label_x, R2_TOP - 12,
                        "U-NET DECODER", size=9, weight="bold", fill="#784212"))
    parts.append(_text(bn_x + BW / 2, BN_Y - 12,
                        "DUAL HEADS", size=9, weight="bold", fill="#641E16"))

    # ── LEGEND BOX — 2 columns × 3 rows, bottom-right corner ──────────────
    leg_items = [
        (C_ENC,    "Encoder operation"),
        (C_DEC,    "Decoder operation"),
        (C_CLS,    "Class. / Seg. head"),
        (C_BN_BOX, "Bottleneck / GAP"),
        (C_POOL,   "Pool / Upsample"),
        (None,     "Skip connection"),
    ]
    leg_rows   = 3                              # items per column (ceil(6/2))
    leg_item_h = 19
    leg_pad    = 8
    leg_col_w  = 78                             # width of one column (icon + label)
    leg_w      = 2 * leg_col_w + 2 * leg_pad + 2
    leg_h      = leg_pad + 16 + leg_rows * leg_item_h + leg_pad
    leg_x      = CANVAS_W - leg_w - 12
    leg_y      = dec_bot_y - leg_h              # bottom of legend aligns with decoder bottom

    # Outer box + title
    parts.append(_rect(leg_x, leg_y, leg_w, leg_h, "white", stroke=C_EDGE, sw=0.8))
    parts.append(_text(leg_x + leg_w / 2, leg_y + leg_pad + 7,
                        "Legend", size=7.5, weight="bold"))

    for k, (c, lbl) in enumerate(leg_items):
        col = k // leg_rows          # 0 = left column, 1 = right column
        row = k % leg_rows
        ix  = leg_x + leg_pad + col * leg_col_w
        iy  = leg_y + leg_pad + 16 + row * leg_item_h
        if c is not None:
            parts.append(_rect(ix, iy + 1, 12, 12, c, sw=0.6))
        else:
            parts.append(
                f'<line x1="{ix:.1f}" y1="{iy + 7:.1f}" '
                f'x2="{ix + 12:.1f}" y2="{iy + 7:.1f}" '
                f'stroke="{C_SKIP}" stroke-width="1.2" stroke-dasharray="4,2"/>'
            )
        parts.append(_text(ix + 16, iy + 7, lbl, size=7.0, anchor="start"))

    parts.append("</svg>")
    return "\n".join(parts)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "output" / "fig0_architecture.svg"
    out.parent.mkdir(exist_ok=True)
    svg = build_svg()
    out.write_text(svg, encoding="utf-8")
    print(f"[saved] {out}")
    # Quick sanity check
    import re
    boxes = len(re.findall(r'<rect ', svg))
    arrows = len(re.findall(r'marker-end', svg))
    print(f"  boxes={boxes}  arrows={arrows}  canvas size: ", end="")
    w = re.search(r'width="(\d+)"', svg)
    h = re.search(r'height="(\d+)"', svg)
    if w and h:
        print(f"{w.group(1)} × {h.group(1)} px")
