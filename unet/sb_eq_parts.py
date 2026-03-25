"""Stage-1 SB-EQ U-Net parts built on top of the milesial U-Net baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DiscreteEqConv2d(nn.Module):
    """A simple C4 weight-sharing approximation for the EQ branch."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.padding = kernel_size // 2
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = []
        for k in range(4):
            kernel = torch.rot90(self.weight, k=k, dims=(-2, -1))
            responses.append(F.conv2d(x, kernel, bias=None, padding=self.padding))
        return self.fuse(torch.cat(responses, dim=1))


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, use_eq: bool) -> None:
        super().__init__()
        conv = DiscreteEqConv2d if use_eq else lambda in_ch, out_ch: nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = make_group_norm(in_channels)
        self.conv1 = conv(in_channels, out_channels)
        self.norm2 = make_group_norm(out_channels)
        self.conv2 = conv(out_channels, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DualBranchBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, downsample: bool = False) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.eq_block = TimeConditionedResidualBlock(in_channels, out_channels, time_dim, use_eq=True)
        self.conv_block = TimeConditionedResidualBlock(in_channels, out_channels, time_dim, use_eq=False)
        self.fuse = nn.Sequential(
            make_group_norm(out_channels * 2),
            nn.SiLU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool(x)
        fe = self.eq_block(x, time_emb)
        fc = self.conv_block(x, time_emb)
        fused = self.fuse(torch.cat([fe, fc], dim=1))
        return fused, fe, fc


class BarycenterDisentangler(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.trunk = TimeConditionedResidualBlock(channels, channels, time_dim, use_eq=False)
        self.barycenter_head = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual_head = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x, time_emb)
        return self.barycenter_head(h), self.residual_head(h)


class DynamicMaskNetwork(nn.Module):
    def __init__(self, bottleneck_channels: int, num_skip_levels: int) -> None:
        super().__init__()
        hidden = max(bottleneck_channels // 2, 32)
        self.trunk = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mask_res_head = nn.Conv2d(hidden, bottleneck_channels, kernel_size=1)
        self.mask_reg_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.mask_eq_heads = nn.ModuleList([nn.Conv2d(hidden, 1, kernel_size=1) for _ in range(num_skip_levels)])

    def forward(
        self,
        barycenter: torch.Tensor,
        residual: torch.Tensor,
        skip_sizes: list[tuple[int, int]],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        h = self.trunk(torch.cat([barycenter, residual], dim=1))
        mask_res = torch.sigmoid(self.mask_res_head(h))
        mask_reg = torch.sigmoid(
            F.interpolate(self.mask_reg_head(h), size=output_size, mode='bilinear', align_corners=False)
        )
        mask_eq = []
        for head, size in zip(self.mask_eq_heads, skip_sizes):
            mask_eq.append(torch.sigmoid(F.interpolate(head(h), size=size, mode='bilinear', align_corners=False)))
        return mask_res, mask_reg, mask_eq


class EqUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool, time_dim: int) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
        self.block = TimeConditionedResidualBlock(out_channels + skip_channels, out_channels, time_dim, use_eq=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x, time_emb)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
