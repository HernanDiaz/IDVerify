"""
architecture.py — Figure 0: DocVerify architecture block diagram.

Pseudo-3D feature-map bricks scaled by log2(spatial) (height) and
log2(channels) (width), arranged in the classic U-Net landscape layout
with staircase skip connections and dual output heads.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from base import FigureGenerator
from config import IEEEStyle

# ── Architecture spec ─────────────────────────────────────────────────────────
_ENC = [
    ("Conv\nBlock 1", 224,  8),
    ("Conv\nBlock 2", 112, 16),
    ("Conv\nBlock 3",  56, 32),
    ("Conv\nBlock 4",  28, 64),
    ("Conv\nBlock 5",  14, 128),
]
_BN  = ("Bottleneck",   7, 256)
_DEC = [
    ("Dec\nBlock 1",  14, 128),
    ("Dec\nBlock 2",  28,  64),
    ("Dec\nBlock 3",  56,  32),
    ("Dec\nBlock 4", 112,  16),
    ("Dec\nBlock 5", 224,   8),
]

# ── Colours (Wong-safe palette) ───────────────────────────────────────────────
_CE    = "#3A7FC1"   # encoder  – blue
_CB    = "#2F9E6B"   # bottleneck – green
_CD    = "#D97A20"   # decoder  – orange
_CCLS  = "#B0463C"   # cls head – red
_CSEG  = "#9B4F96"   # seg head – purple
_CSKIP = "#888888"   # skip connection – grey

# ── Layout constants (data coords) ────────────────────────────────────────────
_YBASE = 1.30          # common bottom y of all feature-map blocks
_HMIN, _HMAX = 0.30, 2.50   # block height range (spatial 7 → 224)
_WMIN, _WMAX = 0.16, 0.66   # block width  range (ch     8 → 256)
_D     = 0.055         # pseudo-3D depth offset
_SEP   = 0.95          # enc/dec block centre-to-centre spacing
_BNSEP = 1.16          # extra gap around bottleneck


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bh(sp: int) -> float:
    t = (np.log2(sp) - np.log2(7)) / (np.log2(224) - np.log2(7))
    return _HMIN + t * (_HMAX - _HMIN)


def _bw(ch: int) -> float:
    t = (np.log2(ch) - np.log2(8)) / (np.log2(256) - np.log2(8))
    return _WMIN + t * (_WMAX - _WMIN)


def _dim(c: str, f: float) -> tuple:
    return tuple(np.clip(np.array(to_rgb(c)) * f, 0, 1))


def _block3d(ax, xc: float, sp: int, ch: int, color: str,
             label: str = "", zbase: int = 3) -> dict:
    """Draw a pseudo-3D feature-map brick; return geometry dict."""
    h, w = _bh(sp), _bw(ch)
    d = _D
    x0, y0 = xc - w / 2, _YBASE

    # right face (darker)
    ax.add_patch(MplPolygon(
        [(x0+w, y0), (x0+w+d, y0+d), (x0+w+d, y0+h+d), (x0+w, y0+h)],
        closed=True, facecolor=_dim(color, 0.68), edgecolor="none", zorder=zbase))

    # top face (lighter)
    ax.add_patch(MplPolygon(
        [(x0, y0+h), (x0+d, y0+h+d), (x0+w+d, y0+h+d), (x0+w, y0+h)],
        closed=True, facecolor=_dim(color, 1.28), edgecolor="none", zorder=zbase))

    # front face
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), w, h,
        linewidth=0.6, edgecolor="white", facecolor=color, zorder=zbase+1))

    # label (centred in front face)
    lines = label.split("\n")
    for k, line in enumerate(lines):
        dy = (k - (len(lines) - 1) / 2) * 0.18
        ax.text(xc, y0 + h/2 + dy, line,
                ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="white", zorder=zbase+2)

    # dimension annotation below block
    ax.text(xc + d/2, y0 - 0.13, f"{sp}²·{ch}ch",
            ha="center", va="top", fontsize=5.0, color="#555555", zorder=zbase+2)

    return dict(x0=x0, y0=y0, w=w, h=h, xc=xc)


def _oplus(ax, xc, yc, r=0.07, color=_CSKIP, zorder=7):
    """Draw a circled-plus (⊕) symbol using patches (font-independent)."""
    ax.add_patch(mpatches.Circle((xc, yc), r,
                                 fill=False, edgecolor=color,
                                 lw=0.8, zorder=zorder))
    ax.plot([xc - r, xc + r], [yc, yc], color=color, lw=0.7, zorder=zorder)
    ax.plot([xc, xc], [yc - r, yc + r], color=color, lw=0.7, zorder=zorder)


def _arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=7,
                                color=color, lw=lw, shrinkA=0, shrinkB=0),
                zorder=6)


# ── Figure class ──────────────────────────────────────────────────────────────

class ArchitectureFigure(FigureGenerator):
    """DocVerify encoder–decoder architecture diagram."""

    def __init__(self) -> None:
        pass  # no PaperData needed

    def generate(self) -> Figure:
        fig, ax = plt.subplots(figsize=(12.5, 5.0))
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0.0, 5.0)
        ax.axis("off")

        # ── X centres ────────────────────────────────────────────────────
        enc_xc = [0.45 + i * _SEP for i in range(5)]   # 0.45 … 4.25
        bn_xc  = enc_xc[-1] + _BNSEP                   # 5.41
        dec_xc = [bn_xc + _BNSEP + i * _SEP for i in range(5)]  # 6.57 … 10.37

        # ── Skip connections (drawn FIRST, behind blocks) ─────────────────
        # y_skip[k] = height of horizontal segment for pair k (innermost first)
        # Pair order: (E5→D1), (E4→D2), (E3→D3), (E2→D4), (E1→D5)
        skip_pairs = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
        y_skips    = [
            _YBASE + _bh(_ENC[4][1]) + 0.20,   # above E5/D1 top
            _YBASE + _bh(_ENC[3][1]) + 0.22,
            _YBASE + _bh(_ENC[2][1]) + 0.24,
            _YBASE + _bh(_ENC[1][1]) + 0.26,
            _YBASE + _bh(_ENC[0][1]) + 0.28,
        ]

        for (ei, di), ys in zip(skip_pairs, y_skips):
            exc = enc_xc[ei]
            dxc = dec_xc[di]
            ey_top = _YBASE + _bh(_ENC[ei][1])
            dy_top = _YBASE + _bh(_DEC[di][1])

            # vertical up from encoder top
            ax.plot([exc, exc], [ey_top, ys],
                    color=_CSKIP, lw=0.75, ls="--", zorder=1)
            # horizontal skip line
            ax.plot([exc, dxc], [ys, ys],
                    color=_CSKIP, lw=0.75, ls="--", zorder=1)
            # vertical down + arrowhead into decoder top
            _arrow(ax, dxc, ys, dxc, dy_top + 0.03, color=_CSKIP, lw=0.75)

            # ⊕ concatenation marker at decoder block top
            _oplus(ax, dxc, dy_top + 0.13, r=0.07, color=_CSKIP)

        # ── Encoder blocks ────────────────────────────────────────────────
        enc_b = []
        for i, (lbl, sp, ch) in enumerate(_ENC):
            enc_b.append(_block3d(ax, enc_xc[i], sp, ch, _CE, lbl))

        # ── Bottleneck ────────────────────────────────────────────────────
        lbl_bn, sp_bn, ch_bn = _BN
        bn_b = _block3d(ax, bn_xc, sp_bn, ch_bn, _CB, lbl_bn)

        # ── Decoder blocks ────────────────────────────────────────────────
        dec_b = []
        for i, (lbl, sp, ch) in enumerate(_DEC):
            dec_b.append(_block3d(ax, dec_xc[i], sp, ch, _CD, lbl))

        # ── Encoder flow arrows ───────────────────────────────────────────
        for i in range(4):
            b, bn_ = enc_b[i], enc_b[i+1]
            xr = b["x0"] + b["w"]
            xl = bn_["x0"]
            ymid = _YBASE + min(b["h"], bn_["h"]) / 2
            _arrow(ax, xr + 0.02, _YBASE + b["h"]/2,
                       xl - 0.02, _YBASE + bn_["h"]/2)
            ax.text((xr + xl) / 2, ymid + 0.22,
                    "MP↓2", ha="center", va="bottom",
                    fontsize=4.8, color="#555555", zorder=7)

        # E5 → BN
        b5, bb = enc_b[4], bn_b
        _arrow(ax, b5["x0"]+b5["w"]+0.02, _YBASE+b5["h"]/2,
                   bb["x0"]-0.02,          _YBASE+bb["h"]/2)
        ax.text((b5["x0"]+b5["w"] + bb["x0"])/2,
                _YBASE + min(b5["h"], bb["h"])/2 + 0.22,
                "MP↓2", ha="center", va="bottom",
                fontsize=4.8, color="#555555", zorder=7)

        # BN → D1
        _arrow(ax, bb["x0"]+bb["w"]+0.02, _YBASE+bb["h"]/2,
                   dec_b[0]["x0"]-0.02,   _YBASE+dec_b[0]["h"]/2)

        # ── Decoder flow arrows ───────────────────────────────────────────
        for i in range(4):
            b, bn_ = dec_b[i], dec_b[i+1]
            xr = b["x0"] + b["w"]
            xl = bn_["x0"]
            ymid = _YBASE + min(b["h"], bn_["h"]) / 2
            _arrow(ax, xr + 0.02, _YBASE + b["h"]/2,
                       xl - 0.02, _YBASE + bn_["h"]/2)
            ax.text((xr + xl) / 2, ymid + 0.18,
                    "Up↑2", ha="center", va="bottom",
                    fontsize=4.8, color="#555555", zorder=7)

        # ── Input annotation ──────────────────────────────────────────────
        e1 = enc_b[0]
        ax.annotate("Input\n224×224×3",
                    xy=(enc_xc[0], _YBASE + e1["h"] + 0.03),
                    xytext=(enc_xc[0] - 0.15, _YBASE + e1["h"] + 0.55),
                    ha="center", va="bottom",
                    fontsize=6.2, fontweight="bold", color="#222222",
                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                    lw=0.9, mutation_scale=7,
                                    shrinkA=2, shrinkB=2),
                    zorder=8)

        # ── Classification head (below BN) ────────────────────────────────
        cw, ch_h = 1.40, 0.34
        cy = 0.22
        ax.add_patch(mpatches.FancyBboxPatch(
            (bn_xc - cw/2, cy), cw, ch_h,
            boxstyle="round,pad=0.03",
            facecolor=_CCLS, edgecolor="white", lw=0.6, zorder=5))
        ax.text(bn_xc, cy + ch_h/2,
                "GAP → Dropout → FC(1) → σ",
                ha="center", va="center",
                fontsize=5.8, fontweight="bold", color="white", zorder=6)
        ax.text(bn_xc, cy - 0.08,
                r"Score $\in$ [0, 1]",
                ha="center", va="top",
                fontsize=5.8, color=_CCLS, zorder=6)
        _arrow(ax, bn_xc, _YBASE,
                   bn_xc, cy + ch_h + 0.02,
                   color=_CCLS, lw=0.9)

        # ── Segmentation head (right of D5) ───────────────────────────────
        d5 = dec_b[4]
        sx = d5["x0"] + d5["w"] + _D + 0.12
        sw, sh_h = 0.90, 0.40
        sy = _YBASE + d5["h"] / 2 - sh_h / 2
        ax.add_patch(mpatches.FancyBboxPatch(
            (sx, sy), sw, sh_h,
            boxstyle="round,pad=0.03",
            facecolor=_CSEG, edgecolor="white", lw=0.6, zorder=5))
        ax.text(sx + sw/2, sy + sh_h/2,
                "Conv 1×1 → σ\nMask 224×224×1",
                ha="center", va="center",
                fontsize=5.8, fontweight="bold", color="white", zorder=6)
        _arrow(ax, d5["x0"]+d5["w"]+_D+0.02, _YBASE+d5["h"]/2,
                   sx - 0.02,                  sy + sh_h/2,
                   color=_CSEG, lw=0.9)

        # ── Section labels ────────────────────────────────────────────────
        y_lbl = _YBASE - 0.55
        ax.text(np.mean(enc_xc), y_lbl, "ENCODER",
                ha="center", va="top",
                fontsize=7.5, fontweight="bold", color=_CE)
        ax.text(bn_xc, y_lbl, "BOTTLENECK",
                ha="center", va="top",
                fontsize=7.5, fontweight="bold", color=_CB)
        ax.text(np.mean(dec_xc), y_lbl, "DECODER",
                ha="center", va="top",
                fontsize=7.5, fontweight="bold", color=_CD)

        # ── Horizontal legend (bottom strip) ──────────────────────────────
        ly = 0.10
        items = [
            (_CE,    "Encoder block (2×Conv + LeakyReLU(0.2) + BN + MaxPool)"),
            (_CB,    "Bottleneck"),
            (_CD,    "Decoder block (Bilinear upsample + 2×Conv + BN)"),
            (_CCLS,  "Classification head"),
            (_CSEG,  "Segmentation head"),
            (_CSKIP, r"Skip connection  ($(+)$ channel-wise concatenation)"),
        ]
        lx = 0.20
        for k, (c, lbl) in enumerate(items):
            col = 0 if k < 3 else 1
            row = k if k < 3 else k - 3
            bx = lx + col * 6.2
            by_text = ly + (2 - row) * 0.27
            if c == _CSKIP:
                ax.plot([bx, bx + 0.18], [by_text, by_text],
                        color=c, lw=1.0, ls="--", zorder=5)
            else:
                ax.add_patch(mpatches.Rectangle(
                    (bx, by_text - 0.07), 0.18, 0.15,
                    facecolor=c, edgecolor="none", zorder=5))
            ax.text(bx + 0.24, by_text, lbl,
                    ha="left", va="center",
                    fontsize=5.2, color="#333333", zorder=5)

        fig.tight_layout(pad=0.15)
        return fig
