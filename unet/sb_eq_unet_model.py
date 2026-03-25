"""Stage-1 SB-EQ U-Net generator built from the milesial U-Net baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .checkpointing import wrap_with_checkpoint
from .sb_eq_parts import (
    BarycenterDisentangler,
    DualBranchBlock,
    DynamicMaskNetwork,
    EqUp,
    OutConv,
    timestep_embedding,
)


class SBEQUNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        out_channels: int = 3,
        bilinear: bool = False,
        time_dim: int = 128,
        residual_scale: float = 1.0,
        predict_residual: bool = True,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.time_dim = time_dim
        self.residual_scale = residual_scale
        self.predict_residual = predict_residual

        factor = 2 if bilinear else 1
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16 // factor

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.inc = DualBranchBlock(n_channels, c1, time_dim, downsample=False)
        self.down1 = DualBranchBlock(c1, c2, time_dim, downsample=True)
        self.down2 = DualBranchBlock(c2, c3, time_dim, downsample=True)
        self.down3 = DualBranchBlock(c3, c4, time_dim, downsample=True)
        self.down4 = DualBranchBlock(c4, c5, time_dim, downsample=True)

        self.disentangler = BarycenterDisentangler(c5, time_dim)
        self.masker = DynamicMaskNetwork(c5, num_skip_levels=4)

        d1 = base_channels * 8 // factor
        d2 = base_channels * 4 // factor
        d3 = base_channels * 2 // factor
        d4 = base_channels

        self.up1 = EqUp(c5, c4, d1, bilinear, time_dim)
        self.up2 = EqUp(d1, c3, d2, bilinear, time_dim)
        self.up3 = EqUp(d2, c2, d3, bilinear, time_dim)
        self.up4 = EqUp(d3, c1, d4, bilinear, time_dim)
        self.outc = OutConv(d4, out_channels)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        source = x
        time_emb = self.time_mlp(timestep_embedding(time_step, self.time_dim))

        x1, fe1, fc1 = self.inc(x, time_emb)
        x2, fe2, fc2 = self.down1(x1, time_emb)
        x3, fe3, fc3 = self.down2(x2, time_emb)
        x4, fe4, fc4 = self.down3(x3, time_emb)
        x5, _, _ = self.down4(x4, time_emb)

        barycenter, residual = self.disentangler(x5, time_emb)
        mask_res, mask_reg, mask_eq = self.masker(
            barycenter=barycenter,
            residual=residual,
            skip_sizes=[x4.shape[-2:], x3.shape[-2:], x2.shape[-2:], x1.shape[-2:]],
            output_size=source.shape[-2:],
        )

        latent = barycenter + self.residual_scale * (residual * mask_res)
        skip4 = mask_eq[0] * fe4 + (1.0 - mask_eq[0]) * fc4
        skip3 = mask_eq[1] * fe3 + (1.0 - mask_eq[1]) * fc3
        skip2 = mask_eq[2] * fe2 + (1.0 - mask_eq[2]) * fc2
        skip1 = mask_eq[3] * fe1 + (1.0 - mask_eq[3]) * fc1

        x = self.up1(latent, skip4, time_emb)
        x = self.up2(x, skip3, time_emb)
        x = self.up3(x, skip2, time_emb)
        x = self.up4(x, skip1, time_emb)

        delta = self.outc(x)
        prediction = source + delta if self.predict_residual else delta
        return {
            'prediction': prediction,
            'barycenter': barycenter,
            'residual': residual,
            'mask_res': mask_res,
            'mask_reg': mask_reg,
            'mask_eq': mask_eq,
        }

    def use_checkpointing(self) -> None:
        self.inc = wrap_with_checkpoint(self.inc)
        self.down1 = wrap_with_checkpoint(self.down1)
        self.down2 = wrap_with_checkpoint(self.down2)
        self.down3 = wrap_with_checkpoint(self.down3)
        self.down4 = wrap_with_checkpoint(self.down4)
        self.up1 = wrap_with_checkpoint(self.up1)
        self.up2 = wrap_with_checkpoint(self.up2)
        self.up3 = wrap_with_checkpoint(self.up3)
        self.up4 = wrap_with_checkpoint(self.up4)
        self.outc = wrap_with_checkpoint(self.outc)
