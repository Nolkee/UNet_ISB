# Stage-1 SB-EQ Restoration on Top of `pytorch-unet`

This folder keeps the original `milesial/pytorch-unet` segmentation baseline intact and adds a new stage-1 restoration path:

- `unet/sb_eq_unet_model.py`: stage-1 generator
- `unet/sb_eq_parts.py`: dual-branch encoder, barycenter/residual disentanglement, dynamic masks, EQ decoder blocks
- `utils/restoration_data_loading.py`: paired restoration dataset
- `utils/restoration_losses.py`: stage-1 losses only
- `train_stage1_restoration.py`: remote training entrypoint

## What is included in stage 1

- U-Net backbone structure from `milesial/pytorch-unet`
- dual encoder (`EQ branch + Conv branch`)
- bottleneck `b/r` disentanglement
- `mask_eq`, `mask_res`, `mask_reg`
- high-frequency detail loss, PatchNCE, BRO, IRC

## What is intentionally excluded

- PatchGAN discriminator
- adversarial loss
- full Wasserstein barycenter regularization

Those belong to stages 2 and 3.

## Remote Ubuntu usage

Run this on your remote server, not on your local machine:

```bash
python train_stage1_restoration.py \
  --train-input-dir /path/to/train/inputs \
  --train-target-dir /path/to/train/targets \
  --val-input-dir /path/to/val/inputs \
  --val-target-dir /path/to/val/targets \
  --save-dir /path/to/checkpoints_stage1 \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --amp
```

If you have timestep / degradation metadata, provide manifests with columns:

```text
input_path,target_path,timestep,degradation_type,degradation_level,degradation_label
```
