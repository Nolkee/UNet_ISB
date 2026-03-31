from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target) ** 2 + self.eps ** 2).mean()


class HighFrequencyDetailLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask_reg: torch.Tensor) -> torch.Tensor:
        mask_reg = F.interpolate(mask_reg, size=prediction.shape[-2:], mode='bilinear', align_corners=False)
        dx_pred = prediction[..., :, 1:] - prediction[..., :, :-1]
        dx_target = target[..., :, 1:] - target[..., :, :-1]
        dy_pred = prediction[..., 1:, :] - prediction[..., :-1, :]
        dy_target = target[..., 1:, :] - target[..., :-1, :]

        mask_x = mask_reg[..., :, 1:]
        mask_y = mask_reg[..., 1:, :]
        loss_x = (mask_x * (dx_pred - dx_target).abs()).sum() / mask_x.sum().clamp_min(1e-6)
        loss_y = (mask_y * (dy_pred - dy_target).abs()).sum() / mask_y.sum().clamp_min(1e-6)
        return loss_x + loss_y


class PatchNCELoss(nn.Module):
    def __init__(self, patch_size: int = 7, patch_stride: int = 4, temperature: float = 0.07, max_patches: int = 128) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.temperature = temperature
        self.max_patches = max_patches

    def forward(self, source: torch.Tensor, prediction: torch.Tensor, mask_reg: torch.Tensor) -> torch.Tensor:
        query = F.unfold(prediction, kernel_size=self.patch_size, stride=self.patch_stride).transpose(1, 2)
        key = F.unfold(source.detach(), kernel_size=self.patch_size, stride=self.patch_stride).transpose(1, 2)
        if query.size(1) == 0:
            return prediction.new_tensor(0.0)

        indices = None
        if query.size(1) > self.max_patches:
            indices = torch.linspace(
                0,
                query.size(1) - 1,
                steps=self.max_patches,
                device=query.device,
                dtype=torch.long,
            )
            query = query.index_select(1, indices)
            key = key.index_select(1, indices)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        logits = torch.matmul(query, key.transpose(1, 2)) / self.temperature
        log_prob = logits.log_softmax(dim=-1)
        positive = -torch.diagonal(log_prob, dim1=1, dim2=2)

        pooled_mask = F.avg_pool2d(mask_reg, kernel_size=self.patch_size, stride=self.patch_stride).flatten(1)
        if indices is not None:
            pooled_mask = pooled_mask.index_select(1, indices)
        weights = pooled_mask / pooled_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (positive * weights).sum(dim=1).mean()


class ResidualContrastiveLoss(nn.Module):
    def __init__(self, feature_dim: int, temperature: float = 0.1, projector_dim: int = 128) -> None:
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
        )

    def forward(self, residual: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels >= 0
        if valid.sum() < 2:
            return residual.new_tensor(0.0)

        features = F.adaptive_avg_pool2d(residual[valid], output_size=1).flatten(1)
        features = F.normalize(self.projector(features), dim=1)
        labels = labels[valid]

        similarity = torch.matmul(features, features.t()) / self.temperature
        logits_mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        positives = (labels[:, None] == labels[None, :]) & logits_mask
        positive_counts = positives.sum(dim=1)
        if (positive_counts > 0).sum() == 0:
            return residual.new_tensor(0.0)

        exp_similarity = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-6))
        mean_log_prob_pos = (positives * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob_pos[positive_counts > 0].mean()


class Stage1RestorationLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        high_frequency_weight: float = 0.5,
        patch_nce_weight: float = 0.1,
        bro_weight: float = 0.05,
        irc_weight: float = 0.05,
        charbonnier_eps: float = 1e-3,
        patch_size: int = 7,
        patch_stride: int = 4,
        patch_temperature: float = 0.07,
        max_patches: int = 128,
        contrastive_temperature: float = 0.1,
        projector_dim: int = 128,
        residual_feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.high_frequency_weight = high_frequency_weight
        self.patch_nce_weight = patch_nce_weight
        self.bro_weight = bro_weight
        self.irc_weight = irc_weight

        self.reconstruction = CharbonnierLoss(charbonnier_eps)
        self.high_frequency = HighFrequencyDetailLoss()
        self.patch_nce = PatchNCELoss(patch_size, patch_stride, patch_temperature, max_patches)
        self.irc = ResidualContrastiveLoss(residual_feature_dim, contrastive_temperature, projector_dim)

    def forward(self, outputs: dict[str, torch.Tensor | list[torch.Tensor]], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs['prediction'].float()
        target = batch['target'].float()
        source = batch['image'].float()
        mask_reg = outputs['mask_reg'].float()
        barycenter = outputs['barycenter'].float()
        residual = outputs['residual'].float()
        labels = batch['degradation_label']

        reconstruction = self.reconstruction(prediction, target)
        high_frequency = self.high_frequency(prediction, target, mask_reg)
        patch_nce = self.patch_nce(source, prediction, mask_reg)
        bro = F.cosine_similarity(barycenter.flatten(1), residual.flatten(1), dim=1).abs().mean()
        irc = self.irc(residual, labels)

        total = (
            self.reconstruction_weight * reconstruction
            + self.high_frequency_weight * high_frequency
            + self.patch_nce_weight * patch_nce
            + self.bro_weight * bro
            + self.irc_weight * irc
        )

        metrics = {
            'loss_total': total.detach(),
            'loss_reconstruction': reconstruction.detach(),
            'loss_high_frequency': high_frequency.detach(),
            'loss_patch_nce': patch_nce.detach(),
            'loss_bro': bro.detach(),
            'loss_irc': irc.detach(),
        }
        return total, metrics
