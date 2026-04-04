from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(filename: str | Path) -> Image.Image:
    path = Path(filename)
    ext = path.suffix.lower()
    if ext == '.npy':
        array = np.load(path)
        return Image.fromarray(np.asarray(array, dtype=np.float32))
    if ext in {'.pt', '.pth'}:
        tensor = torch.load(path, map_location='cpu')
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        array = np.asarray(tensor)
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        return Image.fromarray(array)
    return Image.open(path)


@dataclass(frozen=True)
class RestorationSample:
    input_path: Path
    target_path: Path
    timestep: float
    label: int
    sample_id: str


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        image_size: int | None = None,
        crop_size: int | None = None,
        manifest: str | Path | None = None,
        default_timestep: float = 1.0,
        in_channels: int = 3,
        is_train: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.manifest = Path(manifest) if manifest else None
        self.image_size = image_size
        self.crop_size = crop_size
        self.default_timestep = default_timestep
        self.in_channels = in_channels
        self.is_train = is_train
        self.label_map: dict[str, int] = {}
        self.samples = self._build_samples()
        self.has_valid_labels = any(sample.label >= 0 for sample in self.samples)
        self.uses_manifest = self.manifest is not None
        if not self.samples:
            raise RuntimeError(f'No paired restoration samples found in {self.input_dir} and {self.target_dir}')

    def _build_samples(self) -> list[RestorationSample]:
        if self.manifest:
            return self._samples_from_manifest(self.manifest)
        input_files = {path.name: path for path in self.input_dir.iterdir() if path.is_file() and not path.name.startswith('.')}
        target_files = {path.name: path for path in self.target_dir.iterdir() if path.is_file() and not path.name.startswith('.')}
        names = sorted(input_files.keys() & target_files.keys())
        return [
            RestorationSample(
                input_path=input_files[name],
                target_path=target_files[name],
                timestep=self.default_timestep,
                label=-1,
                sample_id=Path(name).stem,
            )
            for name in names
        ]

    def _samples_from_manifest(self, manifest: Path) -> list[RestorationSample]:
        delimiter = '\t' if manifest.suffix.lower() == '.tsv' else ','
        manifest_root = manifest.parent
        samples = []
        with manifest.open('r', newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = reader.fieldnames or []
            input_key, target_key = self._resolve_manifest_path_keys(fields)
            for row in reader:
                raw_input = (row.get(input_key) or '').strip()
                raw_target = (row.get(target_key) or '').strip()
                if not raw_input or not raw_target:
                    continue

                input_path = self._resolve_path(self.input_dir, raw_input, manifest_root)
                target_path = self._resolve_path(self.target_dir, raw_target, manifest_root)
                if not input_path.exists() or not target_path.exists():
                    continue

                label = self._resolve_label(
                    row.get('degradation_label'),
                    row.get('degradation_type') or row.get('noise_type'),
                    row.get('degradation_level') or row.get('sigma'),
                )
                sample_id = self._resolve_sample_id(row, input_path)
                samples.append(
                    RestorationSample(
                        input_path=input_path,
                        target_path=target_path,
                        timestep=float((row.get('timestep') or self.default_timestep)),
                        label=label,
                        sample_id=sample_id,
                    )
                )
        return samples

    def _resolve_manifest_path_keys(self, fields: list[str]) -> tuple[str, str]:
        field_map = {field.strip().lower(): field for field in fields if field}
        input_candidates = [
            'input_path',
            f'{self.input_dir.name.lower()}_path',
            self.input_dir.name.lower(),
        ]
        target_candidates = [
            'target_path',
            f'{self.target_dir.name.lower()}_path',
            self.target_dir.name.lower(),
        ]

        input_key = next((field_map[candidate] for candidate in input_candidates if candidate in field_map), None)
        target_key = next((field_map[candidate] for candidate in target_candidates if candidate in field_map), None)
        if input_key is None or target_key is None:
            raise KeyError(
                'Manifest missing path columns. '
                f'Expected one of {input_candidates} for input and {target_candidates} for target, '
                f'but got {fields}'
            )
        return input_key, target_key

    def _resolve_sample_id(self, row: dict[str, str], input_path: Path) -> str:
        volume = (row.get('volume') or '').strip()
        slice_id = (row.get('slice_id') or '').strip()
        if volume and slice_id:
            return f'{volume}_s{slice_id}'
        return input_path.stem

    def _resolve_path(self, default_root: Path, raw_path: str, manifest_root: Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if (manifest_root / path).exists():
            return manifest_root / path
        return default_root / path

    def _resolve_label(self, raw_label: str | None, degradation_type: str | None, degradation_level: str | None) -> int:
        if raw_label not in {None, ''}:
            return int(raw_label)
        if degradation_type is None and degradation_level is None:
            return -1
        key = f'{degradation_type or "unknown"}::{degradation_level or "na"}'
        if key not in self.label_map:
            self.label_map[key] = len(self.label_map)
        return self.label_map[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self._load(sample.input_path)
        target = self._load(sample.target_path)
        image, target = self._resize(image, target)
        image, target = self._crop(image, target)
        image, target = self._flip(image, target)
        return {
            'image': self._to_tensor(image),
            'target': self._to_tensor(target),
            'time_step': torch.tensor(sample.timestep, dtype=torch.float32),
            'degradation_label': torch.tensor(sample.label, dtype=torch.long),
            'id': sample.sample_id,
        }

    def _load(self, path: Path) -> Image.Image:
        image = load_image(path)
        if path.suffix.lower() in {'.npy', '.pt', '.pth'}:
            return image
        return image.convert('L') if self.in_channels == 1 else image.convert('RGB')

    def _resize(self, image: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.image_size is None:
            return image, target
        size = (self.image_size, self.image_size)
        return (
            image.resize(size, resample=Image.BICUBIC),
            target.resize(size, resample=Image.BICUBIC),
        )

    def _crop(self, image: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.crop_size is None:
            return image, target
        width, height = image.size
        crop = self.crop_size
        if width < crop or height < crop:
            resize_shape = (max(width, crop), max(height, crop))
            image = image.resize(resize_shape, resample=Image.BICUBIC)
            target = target.resize(resize_shape, resample=Image.BICUBIC)
            width, height = image.size
        if self.is_train:
            left = random.randint(0, width - crop)
            top = random.randint(0, height - crop)
        else:
            left = max((width - crop) // 2, 0)
            top = max((height - crop) // 2, 0)
        box = (left, top, left + crop, top + crop)
        return image.crop(box), target.crop(box)

    def _flip(self, image: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.is_train:
            return image, target
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)
        return image, target

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, None]
        array = array.transpose((2, 0, 1))
        if image.mode != 'F' and (array > 1).any():
            array = array / 255.0
        return torch.as_tensor(array.copy()).float().contiguous()
