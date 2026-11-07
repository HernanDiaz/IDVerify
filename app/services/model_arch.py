"""DocVerify network architecture (Patel CNN encoder + U-Net decoder).

Vendored verbatim from the research model so the published checkpoint loads
with identical layer shapes. Do not change layer definitions here without a
matching checkpoint, or ``load_state_dict`` will fail.

Outputs a dict:
  * ``cls``  — (B, 1) logits; positive => manipulated document.
  * ``mask`` — (B, 1, H, W) logits; positive => altered pixel.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_RELU_ALPHA = 0.2


class DecBlock(nn.Module):
    """Decoder block: bilinear upsample x2, concatenate skip, two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, alpha: float = LEAKY_RELU_ALPHA):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DocVerifyModel(nn.Module):
    """Multi-task CNN: binary authenticity classification + altered-region mask."""

    def __init__(
        self,
        in_ch: int = 3,
        alpha: float = LEAKY_RELU_ALPHA,
        dropout_rate: float = 0.5,
        dec_ch: int = 128,
    ):
        super().__init__()
        self.alpha = alpha

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(8),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(16),
        )
        self.pool1 = nn.AvgPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(32),
        )
        self.pool2 = nn.AvgPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(64),
        )
        self.pool3 = nn.AvgPool2d(2)

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(128),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.enc6 = nn.Sequential(
            nn.Conv2d(128, 256, 5, padding=2, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm2d(256),
        )
        self.pool5 = nn.MaxPool2d(2)

        # Classification head
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 32), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 16), nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
        )

        # Segmentation head (U-Net decoder)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(256, dec_ch, 1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
        )
        self.dec14 = DecBlock(dec_ch, 256, dec_ch, alpha)
        self.dec28 = DecBlock(dec_ch, 128, dec_ch // 2, alpha)
        self.dec56 = DecBlock(dec_ch // 2, 64, dec_ch // 4, alpha)
        self.dec112 = DecBlock(dec_ch // 4, 32, dec_ch // 8, alpha)
        self.dec224 = DecBlock(dec_ch // 8, 16, dec_ch // 16, alpha)
        self.mask_out = nn.Conv2d(dec_ch // 16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2], x.shape[3]

        e1 = self.enc1(x)
        s224 = self.enc2(e1)
        e2 = self.pool1(s224)
        s112 = self.enc3(e2)
        e3 = self.pool2(s112)
        s56 = self.enc4(e3)
        e4 = self.pool3(s56)
        s28 = self.enc5(e4)
        e5 = self.pool4(s28)
        s14 = self.enc6(e5)
        bottleneck = self.pool5(s14)

        c = self.cls_gap(bottleneck).flatten(1)
        cls_out = self.cls_head(c)

        m = self.mask_proj(bottleneck)
        m = self.dec14(m, s14)
        m = self.dec28(m, s28)
        m = self.dec56(m, s56)
        m = self.dec112(m, s112)
        m = self.dec224(m, s224)
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        mask_out = self.mask_out(m)

        return {"cls": cls_out, "mask": mask_out}


def build_model(params: dict[str, Any], device: torch.device) -> DocVerifyModel:
    """Instantiate the model from a checkpoint's ``params`` dict and move to device."""
    model = DocVerifyModel(
        alpha=float(params.get("alpha", LEAKY_RELU_ALPHA)),
        dropout_rate=float(params["dropout_rate"]),
        dec_ch=int(params["dec_ch"]),
    )
    return model.to(device)
