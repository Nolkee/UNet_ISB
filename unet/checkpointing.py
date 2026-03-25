from __future__ import annotations

import torch
import torch.nn as nn


class CheckpointModule(nn.Module):
    """Wrap a module so its forward pass uses gradient checkpointing."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        def run_module(*inputs):
            return self.module(*inputs, **kwargs)

        return torch.utils.checkpoint.checkpoint(run_module, *args, use_reentrant=False)


def wrap_with_checkpoint(module: nn.Module) -> nn.Module:
    return CheckpointModule(module)
