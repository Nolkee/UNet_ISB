"""Stage-3 training: Barycenter Regularization + GAN + content losses.

Loads Stage-2 best checkpoint (generator + discriminator + criterion) and adds
domain-adversarial degradation-invariance loss on the barycenter branch via a
Gradient Reversal Layer.  IRC (residual contrastive) is re-enabled to strengthen
the residual branch's degradation-discriminative property.

Usage:
    python train_stage3_restoration.py \
        --load-stage2 checkpoints_stage2/best_stage2.pth \
        --train-input-dir /path/to/noisy_esc \
        --train-target-dir /path/to/rss_norm \
        --device cuda --epochs 50
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import SBEQUNet, NLayerDiscriminator
from utils.restoration_data_loading import PairedRestorationDataset
from utils.restoration_losses import (
    LSGANLoss,
    Stage1RestorationLoss,
    Stage2RestorationLoss,
    Stage3RestorationLoss,
)
from utils.restoration_metrics import RestorationMetricTracker, ensure_restoration_metric_dependencies

# Reuse shared utilities from Stage-1/2 training
from train_stage1_restoration import (
    move_batch_to_device,
    summarize_metrics,
    format_metrics,
    format_metric,
    resolve_device,
    select_validation_indices,
    save_validation_triplet,
)


def create_argparser():
    parser = argparse.ArgumentParser(description='Train stage-3 SB-EQ U-Net with barycenter regularization')

    # Data paths
    parser.add_argument('--train-input-dir', type=str, required=True)
    parser.add_argument('--train-target-dir', type=str, required=True)
    parser.add_argument('--val-input-dir', type=str, default='')
    parser.add_argument('--val-target-dir', type=str, default='')
    parser.add_argument('--train-manifest', type=str, default='')
    parser.add_argument('--val-manifest', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='checkpoints_stage3')
    parser.add_argument('--device', type=str, default='auto')

    # Stage-2 checkpoint (REQUIRED)
    parser.add_argument('--load-stage2', type=str, required=True,
                        help='Path to Stage-2 best checkpoint (required)')

    # Resume Stage-3 training
    parser.add_argument('--resume', type=str, default='',
                        help='Resume Stage-3 training from a Stage-3 checkpoint')

    # Training hyperparameters
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=8)
    parser.add_argument('--g-lr', type=float, default=5e-5, help='Generator learning rate (halved from Stage-2)')
    parser.add_argument('--d-lr', type=float, default=1e-4, help='Discriminator learning rate (halved from Stage-2)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)

    # Model architecture
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=1)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--time-dim', type=int, default=128)
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--direct-prediction', action='store_true', default=False)

    # Discriminator architecture
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)

    # Data augmentation
    parser.add_argument('--image-size', type=int, default=320)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--default-timestep', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--val-save-count', type=int, default=4)
    parser.add_argument('--val-save-subdir', type=str, default='val_triplets')

    # Content loss weights (forwarded to Stage1RestorationLoss)
    parser.add_argument('--reconstruction-weight', type=float, default=1.0)
    parser.add_argument('--high-frequency-weight', type=float, default=0.8)
    parser.add_argument('--patch-nce-weight', type=float, default=0.15)
    parser.add_argument('--bro-weight', type=float, default=0.05)
    parser.add_argument('--irc-weight', type=float, default=0.05, help='IRC re-enabled in Stage-3')
    parser.add_argument('--charbonnier-eps', type=float, default=1e-3)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--patch-stride', type=int, default=4)
    parser.add_argument('--patch-temperature', type=float, default=0.07)
    parser.add_argument('--max-patches', type=int, default=128)
    parser.add_argument('--contrastive-temperature', type=float, default=0.1)
    parser.add_argument('--projector-dim', type=int, default=128)

    # Adversarial loss (from Stage-2, no re-warmup)
    parser.add_argument('--adv-weight', type=float, default=1.0)
    parser.add_argument('--adv-warmup-epochs', type=int, default=0,
                        help='GAN adversarial warmup (0 = already warm from Stage-2)')

    # Barycenter regularization (NEW in Stage-3)
    parser.add_argument('--wb-weight', type=float, default=0.1,
                        help='Barycenter adversarial loss weight')
    parser.add_argument('--wb-warmup-epochs', type=int, default=5,
                        help='Epochs to ramp L_WB weight from 0 to target')
    parser.add_argument('--grl-max-lambda', type=float, default=1.0,
                        help='Maximum GRL scaling factor')
    parser.add_argument('--grl-ramp-epochs', type=int, default=10,
                        help='Epochs to ramp GRL lambda from 0 to max')
    parser.add_argument('--irc-warmup-epochs', type=int, default=5,
                        help='Epochs to ramp IRC weight from 0 to target (projector was untrained in Stage-2)')
    parser.add_argument('--num-degradation-classes', type=int, default=0,
                        help='Number of degradation classes (0 = auto-detect from dataset)')
    parser.add_argument('--wb-hidden-dim', type=int, default=256,
                        help='Hidden dimension of degradation classifier')

    return parser


def raise_if_non_finite(loss: torch.Tensor, label: str, epoch: int, step: int) -> None:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f'Non-finite {label} at epoch={epoch} step={step}: {loss.item():.6f}')


def compute_d_accuracy(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> tuple[float, float]:
    d_real_acc = (real_scores > 0.5).float().mean().item()
    d_fake_acc = (fake_scores < 0.5).float().mean().item()
    return d_real_acc, d_fake_acc


def evaluate(
    generator, stage3_criterion, loader, device, amp,
    epoch: int, val_save_count: int, val_save_dir: Path | None,
):
    """Validation loop — content losses only."""
    generator.eval()
    stage3_criterion.eval()
    history = []
    selected_indices = select_validation_indices(len(loader.dataset), val_save_count) if val_save_dir else set()
    sample_index = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            batch['image'] = batch['image'].to(dtype=torch.float32, memory_format=torch.channels_last)
            batch['target'] = batch['target'].to(dtype=torch.float32, memory_format=torch.channels_last)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                outputs = generator(batch['image'], batch['time_step'])

            # Use stage1 content criterion for validation
            _, content_metrics = stage3_criterion.stage2_criterion.stage1_criterion(outputs, batch)

            predictions = outputs['prediction']
            batch_size = batch['image'].shape[0]
            sample_ids = batch['id']
            for batch_offset in range(batch_size):
                current_index = sample_index + batch_offset
                if current_index in selected_indices and val_save_dir is not None:
                    save_validation_triplet(
                        output_dir=val_save_dir,
                        epoch=epoch,
                        sample_index=current_index,
                        sample_id=sample_ids[batch_offset],
                        image=batch['image'][batch_offset],
                        prediction=predictions[batch_offset],
                        target=batch['target'][batch_offset],
                    )
            sample_index += batch_size

            history.append({key: float(value.item()) for key, value in content_metrics.items()})
    return summarize_metrics(history)


def evaluate_restoration_metrics(generator, loader, device, amp):
    generator.eval()
    tracker = RestorationMetricTracker(device)

    with torch.no_grad():
        with tqdm(total=len(loader.dataset), desc='Final evaluation', unit='img') as pbar:
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                batch['image'] = batch['image'].to(dtype=torch.float32, memory_format=torch.channels_last)
                batch['target'] = batch['target'].to(dtype=torch.float32, memory_format=torch.channels_last)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    outputs = generator(batch['image'], batch['time_step'])

                tracker.update(outputs['prediction'], batch['target'])
                pbar.update(batch['image'].shape[0])

    return tracker.compute()


def run_final_evaluation(generator, loader, device, amp, save_dir: Path, checkpoint_path: Path | None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_label = 'in-memory model state'

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        checkpoint_label = checkpoint_path.name
    else:
        best = save_dir / 'best_stage3.pth'
        if best.exists():
            checkpoint = torch.load(best, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            checkpoint_label = best.name

    metrics = evaluate_restoration_metrics(generator, loader, device, amp)
    results = {'checkpoint': checkpoint_label, **metrics}
    output_path = save_dir / 'final_eval_metrics.json'
    output_path.write_text(json.dumps(results, indent=2))
    logging.info(
        '[final-eval] checkpoint=%s sample_count=%d psnr=%.4f ssim=%.4f lpips=%.4f',
        checkpoint_label, metrics['sample_count'], metrics['psnr'], metrics['ssim'], metrics['lpips'],
    )
    logging.info('Saved final restoration metrics to %s', output_path)


def train_model(
    generator,
    discriminator,
    stage3_criterion,
    gan_loss,
    g_optimizer,
    d_optimizer,
    train_loader,
    val_loader,
    device,
    args,
    start_epoch: int = 1,
    best_val: float = float('inf'),
):
    g_scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    d_scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'train_args.json').write_text(json.dumps(vars(args), indent=2))
    val_save_dir = save_dir / args.val_save_subdir if args.val_save_count > 0 else None

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        stage3_criterion.train()
        epoch_history = []

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                batch['image'] = batch['image'].to(dtype=torch.float32, memory_format=torch.channels_last)
                batch['target'] = batch['target'].to(dtype=torch.float32, memory_format=torch.channels_last)

                # ---- Forward pass through generator ----
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    outputs = generator(batch['image'], batch['time_step'])
                fake = outputs['prediction']
                real = batch['target']

                # ---- Update Discriminator ----
                d_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    real_scores = discriminator(real)
                    fake_scores_d = discriminator(fake.detach())
                    d_loss = gan_loss.forward_d(real_scores, fake_scores_d)

                if not torch.isfinite(d_loss):
                    logging.warning('Skipping step %d (epoch %d): non-finite d_loss', step, epoch)
                    pbar.update(batch['image'].shape[0])
                    continue

                d_scaler.scale(d_loss).backward()
                d_scaler.unscale_(d_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                d_scaler.step(d_optimizer)
                d_scaler.update()

                # ---- Update Generator (content + adv + L_WB via GRL) ----
                g_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    fake_scores_g = discriminator(fake)
                    g_loss, metrics = stage3_criterion(outputs, batch, fake_scores_g, epoch)

                if not torch.isfinite(g_loss):
                    logging.warning('Skipping step %d (epoch %d): non-finite g_loss', step, epoch)
                    pbar.update(batch['image'].shape[0])
                    continue

                g_scaler.scale(g_loss).backward()
                g_scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(generator.parameters()) + list(stage3_criterion.parameters()), args.grad_clip
                )
                g_scaler.step(g_optimizer)
                g_scaler.update()

                # ---- Logging ----
                d_real_acc, d_fake_acc = compute_d_accuracy(real_scores.detach(), fake_scores_d.detach())
                step_metrics = {key: float(value.item()) for key, value in metrics.items()}
                step_metrics['d_loss'] = float(d_loss.item())
                step_metrics['d_real_acc'] = d_real_acc
                step_metrics['d_fake_acc'] = d_fake_acc
                epoch_history.append(step_metrics)

                wb_acc_str = f'{step_metrics.get("wb_classifier_acc", -1):.0%}'
                pbar.update(batch['image'].shape[0])
                pbar.set_postfix(
                    g=float(metrics['loss_total'].item()),
                    d=float(d_loss.item()),
                    wb=wb_acc_str,
                    dr=f'{d_real_acc:.0%}',
                    df=f'{d_fake_acc:.0%}',
                )

        train_metrics = summarize_metrics(epoch_history)
        logging.info('[train] %s', format_metrics(train_metrics))

        # ---- Validation ----
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate(
                generator, stage3_criterion, val_loader, device, args.amp,
                epoch=epoch, val_save_count=args.val_save_count, val_save_dir=val_save_dir,
            )
            logging.info('[val] %s', format_metrics(val_metrics))

        # ---- Checkpointing ----
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'criterion_state_dict': stage3_criterion.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'args': vars(args),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch:03d}.pth')
        if val_metrics and val_metrics.get('loss_total', float('inf')) < best_val:
            best_val = val_metrics['loss_total']
            torch.save(checkpoint, save_dir / 'best_stage3.pth')


def main():
    args = create_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = resolve_device(args.device)
    logging.info('Using device %s', device)

    # ---- Build generator ----
    generator = SBEQUNet(
        n_channels=args.in_channels,
        out_channels=args.out_channels,
        bilinear=args.bilinear,
        time_dim=args.time_dim,
        residual_scale=args.residual_scale,
        predict_residual=not args.direct_prediction,
        base_channels=args.base_channels,
    )
    generator = generator.to(memory_format=torch.channels_last)
    generator.to(device=device)

    # ---- Build discriminator ----
    discriminator = NLayerDiscriminator(
        in_channels=args.out_channels,
        ndf=args.ndf,
        n_layers=args.n_layers,
    )
    discriminator = discriminator.to(memory_format=torch.channels_last)
    discriminator.to(device=device)

    # ---- Load Stage-2 checkpoint (generator + discriminator) ----
    stage2_ckpt = torch.load(args.load_stage2, map_location=device)
    generator.load_state_dict(stage2_ckpt['generator_state_dict'], strict=False)
    discriminator.load_state_dict(stage2_ckpt['discriminator_state_dict'], strict=False)
    logging.info('Loaded Stage-2 generator + discriminator from %s', args.load_stage2)

    # ---- Build datasets first (to detect num_degradation_classes) ----
    train_set = PairedRestorationDataset(
        input_dir=args.train_input_dir,
        target_dir=args.train_target_dir,
        image_size=args.image_size,
        crop_size=args.crop_size,
        manifest=args.train_manifest or None,
        default_timestep=args.default_timestep,
        in_channels=args.in_channels,
        is_train=True,
    )
    if not train_set.has_valid_labels:
        raise ValueError(
            'Stage-3 requires degradation labels for barycenter regularization. '
            'Provide degradation_label / degradation_type in the manifest.'
        )

    num_classes = args.num_degradation_classes
    if num_classes <= 0:
        num_classes = len(train_set.label_map) if train_set.label_map else max(
            sample.label for sample in train_set.samples
        ) + 1
        logging.info('Auto-detected %d degradation classes from training dataset', num_classes)

    # ---- Build loss stack ----
    residual_feature_dim = args.base_channels * 16 // (2 if args.bilinear else 1)

    stage1_criterion = Stage1RestorationLoss(
        reconstruction_weight=args.reconstruction_weight,
        high_frequency_weight=args.high_frequency_weight,
        patch_nce_weight=args.patch_nce_weight,
        bro_weight=args.bro_weight,
        irc_weight=args.irc_weight,
        charbonnier_eps=args.charbonnier_eps,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_temperature=args.patch_temperature,
        max_patches=args.max_patches,
        contrastive_temperature=args.contrastive_temperature,
        projector_dim=args.projector_dim,
        residual_feature_dim=residual_feature_dim,
    ).to(device)

    stage2_criterion = Stage2RestorationLoss(
        stage1_criterion=stage1_criterion,
        adversarial_weight=args.adv_weight,
        warmup_epochs=args.adv_warmup_epochs,
    ).to(device)

    # Restore Stage-2 criterion learned parameters
    if 'criterion_state_dict' in stage2_ckpt:
        stage2_criterion.load_state_dict(stage2_ckpt['criterion_state_dict'], strict=False)
        logging.info('Restored Stage-2 criterion state')

    stage3_criterion = Stage3RestorationLoss(
        stage2_criterion=stage2_criterion,
        wb_weight=args.wb_weight,
        wb_warmup_epochs=args.wb_warmup_epochs,
        grl_max_lambda=args.grl_max_lambda,
        grl_ramp_epochs=args.grl_ramp_epochs,
        feature_dim=residual_feature_dim,
        hidden_dim=args.wb_hidden_dim,
        num_classes=num_classes,
        irc_warmup_epochs=args.irc_warmup_epochs,
    ).to(device)

    gan_loss = LSGANLoss()

    # ---- Optimizers (TTUR, halved from Stage-2) ----
    g_optimizer = optim.AdamW(
        list(generator.parameters()) + list(stage3_criterion.parameters()),
        lr=args.g_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    d_optimizer = optim.AdamW(
        discriminator.parameters(),
        lr=args.d_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )

    # ---- Resume Stage-3 checkpoint if provided ----
    start_epoch = 1
    best_val = float('inf')
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(resume_ckpt['generator_state_dict'], strict=False)
        discriminator.load_state_dict(resume_ckpt['discriminator_state_dict'], strict=False)
        if 'criterion_state_dict' in resume_ckpt:
            stage3_criterion.load_state_dict(resume_ckpt['criterion_state_dict'])
        if 'g_optimizer_state_dict' in resume_ckpt:
            g_optimizer.load_state_dict(resume_ckpt['g_optimizer_state_dict'])
        if 'd_optimizer_state_dict' in resume_ckpt:
            d_optimizer.load_state_dict(resume_ckpt['d_optimizer_state_dict'])
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        loaded_val = resume_ckpt.get('val_metrics', {})
        if loaded_val:
            best_val = float(loaded_val.get('loss_total', best_val))
        logging.info('Resumed Stage-3 training from %s (epoch %d)', args.resume, start_epoch)

    # ---- Dataloaders ----
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    val_loader = None
    if args.val_input_dir and args.val_target_dir:
        val_set = PairedRestorationDataset(
            input_dir=args.val_input_dir,
            target_dir=args.val_target_dir,
            image_size=args.image_size,
            crop_size=args.crop_size,
            manifest=args.val_manifest or None,
            default_timestep=args.default_timestep,
            in_channels=args.in_channels,
            is_train=False,
        )
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    if start_epoch > args.epochs:
        logging.info('Checkpoint epoch %d already meets or exceeds requested epochs=%d; nothing to train.',
                     start_epoch - 1, args.epochs)
        if val_loader is None:
            logging.warning('No validation dataset configured; skipping final evaluation.')
            return
        ensure_restoration_metric_dependencies()
        run_final_evaluation(generator, val_loader, device, args.amp, Path(args.save_dir), None)
        return

    if val_loader is not None:
        ensure_restoration_metric_dependencies()

    if start_epoch > 1:
        logging.info('Resuming stage-3 training from epoch %d', start_epoch)

    train_model(
        generator=generator,
        discriminator=discriminator,
        stage3_criterion=stage3_criterion,
        gan_loss=gan_loss,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
        start_epoch=start_epoch,
        best_val=best_val,
    )

    if val_loader is None:
        logging.warning('No validation dataset configured; skipping final PSNR/SSIM/LPIPS evaluation.')
        return

    run_final_evaluation(generator, val_loader, device, args.amp, Path(args.save_dir), None)


if __name__ == '__main__':
    main()
