from __future__ import annotations

import torch


METRIC_DEPENDENCY_ERROR = (
    'Final PSNR/SSIM/LPIPS evaluation requires the optional dependencies '
    '`torchmetrics` and `lpips`. Install the updated requirements and rerun `python check_env.py` '
    'before starting Stage-1 training with validation enabled.'
)


def ensure_restoration_metric_dependencies():
    try:
        import lpips
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )
    except ImportError as exc:
        raise ImportError(METRIC_DEPENDENCY_ERROR) from exc
    return lpips, peak_signal_noise_ratio, structural_similarity_index_measure


class RestorationMetricTracker:
    def __init__(self, device: torch.device):
        lpips_module, psnr_fn, ssim_fn = ensure_restoration_metric_dependencies()
        self._psnr_fn = psnr_fn
        self._ssim_fn = ssim_fn
        self._lpips_model = lpips_module.LPIPS(net='alex').to(device)
        self._lpips_model.eval()
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._lpips_sum = 0.0
        self._sample_count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions, targets = self._prepare_pair(predictions, targets)
        lpips_values = self._lpips_model(
            self._prepare_lpips_input(predictions),
            self._prepare_lpips_input(targets),
        ).view(-1)

        for prediction, target, lpips_value in zip(predictions, targets, lpips_values):
            prediction = prediction.unsqueeze(0)
            target = target.unsqueeze(0)
            self._psnr_sum += float(self._psnr_fn(prediction, target, data_range=1.0).item())
            self._ssim_sum += float(self._ssim_fn(prediction, target, data_range=1.0).item())
            self._lpips_sum += float(lpips_value.item())
            self._sample_count += 1

    def compute(self) -> dict[str, float | int]:
        if self._sample_count == 0:
            raise ValueError('Cannot compute restoration metrics without any validation samples.')
        return {
            'sample_count': self._sample_count,
            'psnr': self._psnr_sum / self._sample_count,
            'ssim': self._ssim_sum / self._sample_count,
            'lpips': self._lpips_sum / self._sample_count,
        }

    @staticmethod
    def _prepare_pair(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = predictions.detach().float().clamp(0, 1)
        targets = targets.detach().float().clamp(0, 1)
        if predictions.shape != targets.shape:
            raise ValueError(
                f'Predictions and targets must have the same shape, got {tuple(predictions.shape)} '
                f'and {tuple(targets.shape)}'
            )
        if predictions.ndim != 4:
            raise ValueError(f'Expected NCHW tensors for restoration metrics, got {tuple(predictions.shape)}')
        return predictions, targets

    @staticmethod
    def _prepare_lpips_input(images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        if channels == 1:
            images = images.repeat(1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f'LPIPS expects 1 or 3 channels, got {channels}')
        return images.mul(2.0).sub(1.0)
