import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import SBEQUNet
from utils.restoration_data_loading import PairedRestorationDataset
from utils.restoration_losses import Stage1RestorationLoss


def create_argparser():
    parser = argparse.ArgumentParser(description='Train stage-1 SB-EQ U-Net for paired restoration')

    parser.add_argument('--train-input-dir', type=str, required=True, help='Directory with degraded/noisy inputs')
    parser.add_argument('--train-target-dir', type=str, required=True, help='Directory with clean targets')
    parser.add_argument('--val-input-dir', type=str, default='', help='Validation input directory')
    parser.add_argument('--val-target-dir', type=str, default='', help='Validation target directory')
    parser.add_argument('--train-manifest', type=str, default='', help='Optional train CSV/TSV manifest')
    parser.add_argument('--val-manifest', type=str, default='', help='Optional val CSV/TSV manifest')
    parser.add_argument('--save-dir', type=str, default='checkpoints_stage1', help='Checkpoint output directory')
    parser.add_argument('--load', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Training device, e.g. cuda, cuda:0, cpu')

    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=4)
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)

    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--out-channels', type=int, default=3)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--time-dim', type=int, default=128)
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--direct-prediction', action='store_true', default=False,
                        help='Disable residual prediction and predict targets directly')

    parser.add_argument('--image-size', type=int, default=256, help='Resize images before cropping')
    parser.add_argument('--crop-size', type=int, default=256, help='Training/validation crop size')
    parser.add_argument('--default-timestep', type=float, default=1.0,
                        help='Fallback timestep used when no manifest is provided')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--val-save-count', type=int, default=4,
                        help='Number of validation triplets to save per epoch (0 disables saving)')
    parser.add_argument('--val-save-subdir', type=str, default='val_triplets',
                        help='Subdirectory under save-dir for validation triplets')

    parser.add_argument('--reconstruction-weight', type=float, default=1.0)
    parser.add_argument('--high-frequency-weight', type=float, default=0.5)
    parser.add_argument('--patch-nce-weight', type=float, default=0.1)
    parser.add_argument('--bro-weight', type=float, default=0.05)
    parser.add_argument('--irc-weight', type=float, default=0.05)
    parser.add_argument('--charbonnier-eps', type=float, default=1e-3)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--patch-stride', type=int, default=4)
    parser.add_argument('--patch-temperature', type=float, default=0.07)
    parser.add_argument('--max-patches', type=int, default=128)
    parser.add_argument('--contrastive-temperature', type=float, default=0.1)
    parser.add_argument('--projector-dim', type=int, default=128)

    return parser


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def summarize_metrics(history):
    if not history:
        return {}
    keys = history[0].keys()
    return {key: sum(item[key] for item in history) / len(history) for key in keys}


def select_validation_indices(dataset_size: int, count: int) -> set[int]:
    if dataset_size <= 0 or count <= 0:
        return set()
    if count >= dataset_size:
        return set(range(dataset_size))
    if count == 1:
        return {0}

    step = (dataset_size - 1) / (count - 1)
    indices = {min(dataset_size - 1, round(i * step)) for i in range(count)}
    next_index = 0
    while len(indices) < count:
        indices.add(next_index)
        next_index += 1
    return indices


def sanitize_sample_id(sample_id: str) -> str:
    sanitized = ''.join(char if char.isalnum() or char in {'-', '_', '.'} else '_' for char in sample_id.strip())
    return sanitized or 'sample'


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0, 1)
    if image.ndim != 3:
        raise ValueError(f'Expected CHW tensor for image saving, got shape {tuple(image.shape)}')
    if image.shape[0] == 1:
        return Image.fromarray(image.squeeze(0).mul(255).round().byte().numpy())
    if image.shape[0] == 3:
        return Image.fromarray(image.permute(1, 2, 0).mul(255).round().byte().numpy())
    raise ValueError(f'Expected 1 or 3 channels for image saving, got {image.shape[0]}')


def save_validation_triplet(
    output_dir: Path,
    epoch: int,
    sample_index: int,
    sample_id: str,
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    epoch_dir = output_dir / f'epoch_{epoch:03d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'{sample_index:04d}_{sanitize_sample_id(sample_id)}'
    tensor_to_pil_image(image).save(epoch_dir / f'{prefix}_input.png')
    tensor_to_pil_image(prediction).save(epoch_dir / f'{prefix}_pred.png')
    tensor_to_pil_image(target).save(epoch_dir / f'{prefix}_target.png')


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate(model, criterion, loader, device, amp, epoch: int, val_save_count: int, val_save_dir: Path | None):
    model.eval()
    criterion.eval()
    history = []
    selected_indices = select_validation_indices(len(loader.dataset), val_save_count) if val_save_dir else set()
    sample_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch['image'] = batch['image'].to(dtype=torch.float32, memory_format=torch.channels_last)
            batch['target'] = batch['target'].to(dtype=torch.float32, memory_format=torch.channels_last)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                outputs = model(batch['image'], batch['time_step'])
                _, metrics = criterion(outputs, batch)

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

            history.append({key: float(value.item()) for key, value in metrics.items()})
    return summarize_metrics(history)


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    args,
    start_epoch: int = 1,
    best_val: float = float('inf'),
):
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'train_args.json').write_text(json.dumps(vars(args), indent=2))
    val_save_dir = save_dir / args.val_save_subdir if args.val_save_count > 0 else None

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        criterion.train()
        epoch_history = []
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                batch['image'] = batch['image'].to(dtype=torch.float32, memory_format=torch.channels_last)
                batch['target'] = batch['target'].to(dtype=torch.float32, memory_format=torch.channels_last)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    outputs = model(batch['image'], batch['time_step'])
                    loss, metrics = criterion(outputs, batch)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_history.append({key: float(value.item()) for key, value in metrics.items()})
                pbar.update(batch['image'].shape[0])
                pbar.set_postfix(loss=float(metrics['loss_total'].item()))

        train_metrics = summarize_metrics(epoch_history)
        logging.info('[train] %s', ' '.join(f'{k}={v:.4f}' for k, v in train_metrics.items()))

        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                criterion,
                val_loader,
                device,
                args.amp,
                epoch=epoch,
                val_save_count=args.val_save_count,
                val_save_dir=val_save_dir,
            )
            logging.info('[val] %s', ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch:03d}.pth')
        if val_metrics and val_metrics.get('loss_total', float('inf')) < best_val:
            best_val = val_metrics['loss_total']
            torch.save(checkpoint, save_dir / 'best_stage1.pth')


def main():
    args = create_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = resolve_device(args.device)
    logging.info('Using device %s', device)

    model = SBEQUNet(
        n_channels=args.in_channels,
        out_channels=args.out_channels,
        bilinear=args.bilinear,
        time_dim=args.time_dim,
        residual_scale=args.residual_scale,
        predict_residual=not args.direct_prediction,
        base_channels=args.base_channels,
    )
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'criterion_state_dict' in checkpoint:
            logging.info('Checkpoint contains criterion state and will be restored after criterion init')
        logging.info('Loaded checkpoint from %s', args.load)

    criterion = Stage1RestorationLoss(
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
        residual_feature_dim=args.base_channels * 16 // (2 if args.bilinear else 1),
    ).to(device)

    if args.load and 'criterion_state_dict' in checkpoint:
        criterion.load_state_dict(checkpoint['criterion_state_dict'])

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    best_val = float('inf')
    if args.load and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        loaded_val = checkpoint.get('val_metrics', {})
        if loaded_val:
            best_val = float(loaded_val.get('loss_total', best_val))

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
    if args.irc_weight > 0 and not train_set.has_valid_labels:
        raise ValueError(
            'IRC loss requires degradation labels, but no valid labels were found in the training dataset. '
            'Provide degradation_label in the manifest or set --irc-weight 0.'
        )
    if not train_set.uses_manifest:
        logging.warning(
            'No training manifest provided: all samples will use default timestep %.4f and IRC labels will be absent.',
            args.default_timestep,
        )
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
        return

    if start_epoch > 1:
        logging.info('Resuming stage-1 training from epoch %d', start_epoch)

    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        args,
        start_epoch=start_epoch,
        best_val=best_val,
    )


if __name__ == '__main__':
    main()
