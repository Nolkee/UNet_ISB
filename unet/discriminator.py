"""PatchGAN discriminator for Stage-2 adversarial training.

Standard NLayerDiscriminator with 70x70 receptive field, spectral normalization
on all convolutional layers, and InstanceNorm2d (skip norm on first layer).
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (Isola et al., 2017) with spectral normalization.

    Architecture for default ``n_layers=3``, ``ndf=64``::

        C64(no norm) → C128 → C256 → C512(stride-1) → 1-ch patch map

    Each block: ``Conv2d(4×4, stride=2) → InstanceNorm → LeakyReLU(0.2)``.
    Final layer uses ``stride=1`` to keep spatial resolution.
    All ``Conv2d`` layers are wrapped with :func:`spectral_norm`.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default 3).
    ndf : int
        Base number of discriminator filters (default 64).
    n_layers : int
        Number of intermediate down-sampling blocks (default 3).
        3 layers yields a 70×70 receptive field for 4×4 kernels.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        # First layer: Conv → LeakyReLU, no normalization
        sequence: list[nn.Module] = [
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers: Conv → Norm → LeakyReLU
        nf_prev = ndf
        for n in range(1, n_layers):
            nf_curr = min(ndf * (2 ** n), 512)
            sequence += [
                spectral_norm(nn.Conv2d(nf_prev, nf_curr, kernel_size=4, stride=2, padding=1)),
                norm_layer(nf_curr),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            nf_prev = nf_curr

        # Penultimate layer: stride-1 to maintain spatial size
        nf_curr = min(ndf * (2 ** n_layers), 512)
        sequence += [
            spectral_norm(nn.Conv2d(nf_prev, nf_curr, kernel_size=4, stride=1, padding=1)),
            norm_layer(nf_curr),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final layer: 1-channel patch map (raw logits for LSGAN)
        sequence += [
            spectral_norm(nn.Conv2d(nf_curr, 1, kernel_size=4, stride=1, padding=1)),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch-level real/fake score map.

        Parameters
        ----------
        x : Tensor
            Input image of shape ``[B, C, H, W]``.

        Returns
        -------
        Tensor
            Patch score map of shape ``[B, 1, H', W']`` where ``H'`` and ``W'``
            depend on the input spatial size and ``n_layers``.
        """
        return self.model(x)
