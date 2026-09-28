from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from forest_monitoring.config import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_SEGMENTATION_THRESHOLD,
    FOREST_SEGMENTATION_DATA_DIR,
    IMAGE_SIZE,
    MODEL_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    SEED,
    UNET_MODEL_PATH,
)
from forest_monitoring.models.unet import UNet

SPLIT_ALIASES = {
    "train": {"train", "training"},
    "validation": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}
MASK_SUFFIXES = {".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ForestSample:
    image_path: Path
    mask_path: Path
    split: str
    biome: str


def _split_for_path(path: Path) -> str | None:
    for part in path.parts:
        token = part.lower().replace("-", "_").replace(" ", "_")
        for split, aliases in SPLIT_ALIASES.items():
            if token in aliases or any(alias in token for alias in aliases):
                return split
    return None


def _biome_for_path(path: Path) -> str:
    for part in path.parts:
        token = part.lower()
        if "amazon" in token:
            return "amazon"
        if "atlantic" in token:
            return "atlantic_forest"
    return "unspecified"


def _canonical_stem(path: Path) -> str:
    stem = path.stem.lower()
    for prefix in ("mask_", "label_", "gt_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("_mask", "_label", "_gt", "_groundtruth"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def discover_forest_samples(data_root: str | Path) -> tuple[list[ForestSample], dict[str, dict[str, int]]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Forest dataset directory does not exist: {root}")

    image_counts = {split: 0 for split in SPLIT_ALIASES}
    paired_counts = {split: 0 for split in SPLIT_ALIASES}
    samples: list[ForestSample] = []
    image_dirs = [
        path for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in {"image", "images"}
    ]
    mask_dir_names = {"mask", "masks", "label", "labels", "groundtruth", "ground_truth"}
    for image_dir in sorted(image_dirs):
        image_path_parent = image_dir.parent
        mask_dirs = [
            path for path in image_path_parent.iterdir()
            if path.is_dir() and path.name.lower() in mask_dir_names
        ]
        if len(mask_dirs) != 1:
            continue
        mask_dir = mask_dirs[0]
        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            split = _split_for_path(image_path)
            if split is None:
                continue
            image_counts[split] += 1
            mask_matches = [
                path for path in mask_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in MASK_SUFFIXES
                and _canonical_stem(path) == _canonical_stem(image_path)
            ]
            if len(mask_matches) != 1:
                continue
            mask_path = mask_matches[0]
            samples.append(ForestSample(image_path, mask_path, split, _biome_for_path(image_path)))
            paired_counts[split] += 1

    counts = {
        split: {"images": image_counts[split], "paired_masks": paired_counts[split]}
        for split in SPLIT_ALIASES
    }
    return samples, counts


def _read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        image = src.read()
        metadata = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
        }
    if image.shape[0] != DEFAULT_INPUT_CHANNELS:
        raise ValueError(f"Expected {DEFAULT_INPUT_CHANNELS} bands in {path}; found {image.shape[0]}")
    if image.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} tile in {path}; found {image.shape[1:]}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite values in image {path}")
    return image, metadata


def _read_mask(path: Path) -> tuple[np.ndarray, list[int], str]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            raw = src.read(1)
            dtype = src.dtypes[0]
    else:
        with Image.open(path) as source:
            raw = np.asarray(source.convert("L"))
        dtype = str(raw.dtype)
    classes = np.unique(raw).tolist()
    if not set(classes).issubset({0, 1, 255}):
        raise ValueError(f"Unexpected mask classes in {path}: {classes[:20]}")
    return (raw > 0).astype(np.float32), classes, dtype


def inspect_forest_dataset(data_root: str | Path = FOREST_SEGMENTATION_DATA_DIR) -> dict:
    samples, split_counts = discover_forest_samples(data_root)
    if not samples:
        raise ValueError(f"No labelled TIFF/PNG samples found under {data_root}")
    sample = samples[0]
    image, metadata = _read_raster(sample.image_path)
    mask, classes, mask_dtype = _read_mask(sample.mask_path)
    if mask.shape != image.shape[1:]:
        raise ValueError(f"Image/mask dimension mismatch: {sample.image_path} {image.shape[1:]} vs {sample.mask_path} {mask.shape}")
    return {
        "source": "https://zenodo.org/records/4498086",
        "task": "forest/non-forest semantic segmentation",
        "root": str(Path(data_root).resolve()),
        "split_counts": split_counts,
        "labelled_pairs": len(samples),
        "example": {
            "split": sample.split,
            "biome": sample.biome,
            "image_path": str(sample.image_path.resolve()),
            "image_shape_chw": list(image.shape),
            "image_dtype": str(image.dtype),
            "image_value_range": [float(image.min()), float(image.max())],
            "image_metadata": metadata,
            "mask_path": str(sample.mask_path.resolve()),
            "mask_shape_hw": list(mask.shape),
            "mask_dtype": mask_dtype,
            "mask_value_range_raw": [int(min(classes)), int(max(classes))],
            "mask_classes_raw": classes,
            "mask_classes_meaning": {"0": "non-forest", "1 or 255": "forest"},
        },
    }


class ForestSegmentationDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, samples: list[ForestSample] | None = None):
        all_samples, _ = discover_forest_samples(data_root) if samples is None else (samples, {})
        self.samples = [sample for sample in all_samples if sample.split == split]
        self.split = split
        if not self.samples:
            raise ValueError(f"No labelled samples in official '{split}' split under {data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image, _ = _read_raster(sample.image_path)
        mask, _, _ = _read_mask(sample.mask_path)
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Image/mask dimension mismatch: {sample.image_path} and {sample.mask_path}")
        image = np.clip(image.astype(np.float32) / 10000.0, 0.0, 1.0)
        return torch.from_numpy(image.copy()), torch.from_numpy(mask[None, :, :].copy())


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    total = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (total + eps)).mean()


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    true_positive = false_positive = false_negative = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += float((criterion(logits, masks) + _dice_loss(logits, masks)).item())
            predicted = torch.sigmoid(logits) >= threshold
            actual = masks >= 0.5
            true_positive += int((predicted & actual).sum().item())
            false_positive += int((predicted & ~actual).sum().item())
            false_negative += int((~predicted & actual).sum().item())
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    return {
        "loss": total_loss / max(1, len(loader)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "iou": true_positive / (true_positive + false_positive + false_negative + 1e-8),
        "dice": 2 * true_positive / (2 * true_positive + false_positive + false_negative + 1e-8),
    }


def _save_real_examples(dataset: ForestSegmentationDataset) -> Path:
    if len(dataset) < 3:
        raise ValueError(f"Need at least 3 real training samples to create the requested preview; found {len(dataset)}")
    figure, axes = plt.subplots(3, 3, figsize=(14, 12))
    for index in range(3):
        image, mask = dataset[index]
        rgb = image[:3].permute(1, 2, 0).numpy()
        forest = mask[0].numpy() >= 0.5
        axes[index, 0].imshow(rgb)
        axes[index, 0].set_title(f"RGB: {dataset.samples[index].image_path.name}")
        axes[index, 1].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title("Ground-truth forest mask")
        axes[index, 2].imshow(rgb)
        overlay = np.zeros((*forest.shape, 4), dtype=np.float32)
        overlay[forest] = (0.0, 1.0, 0.2, 0.45)
        axes[index, 2].imshow(overlay)
        axes[index, 2].set_title("Forest mask overlay")
        for axis in axes[index]:
            axis.axis("off")
    output_path = OUTPUTS_DIR / "forest_segmentation_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def _save_test_predictions(
    model: nn.Module,
    dataset: ForestSegmentationDataset,
    device: torch.device,
    threshold: float,
    count: int = 3,
) -> Path:
    if len(dataset) < count:
        raise ValueError(f"Need at least {count} held-out test samples for prediction visualization; found {len(dataset)}")
    figure, axes = plt.subplots(count, 4, figsize=(18, 4 * count))
    model.eval()
    with torch.inference_mode():
        for index in range(count):
            image, mask = dataset[index]
            prediction = (torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy() >= threshold)
            rgb = image[:3].permute(1, 2, 0).numpy()
            target = mask[0].numpy() >= 0.5
            axes[index, 0].imshow(rgb)
            axes[index, 0].set_title(f"RGB: {dataset.samples[index].image_path.name}")
            axes[index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[index, 1].set_title("Ground truth")
            axes[index, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[index, 2].set_title("Predicted forest mask")
            axes[index, 3].imshow(rgb)
            overlay = np.zeros((*prediction.shape, 4), dtype=np.float32)
            overlay[prediction] = (0.0, 1.0, 0.2, 0.45)
            axes[index, 3].imshow(overlay)
            axes[index, 3].set_title("Prediction overlay")
            for axis in axes[index]:
                axis.axis("off")
    output_path = OUTPUTS_DIR / "forest_segmentation_test_predictions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def train_unet(
    data_root: str | Path = FOREST_SEGMENTATION_DATA_DIR,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    threshold: float = DEFAULT_SEGMENTATION_THRESHOLD,
    max_train_samples: int | None = 100,
    max_validation_samples: int | None = 10,
) -> dict:
    inspection = inspect_forest_dataset(data_root)
    train_set = ForestSegmentationDataset(data_root, "train")
    validation_set = ForestSegmentationDataset(data_root, "validation")
    available_validation_samples = len(validation_set)
    if max_validation_samples is not None:
        if max_validation_samples < 1:
            raise ValueError("max_validation_samples must be at least 1")
        sample_count = min(max_validation_samples, available_validation_samples)
        validation_set.samples = random.Random(SEED + 1).sample(validation_set.samples, sample_count)
    available_train_samples = len(train_set)
    if max_train_samples is not None:
        if max_train_samples < 3:
            raise ValueError("max_train_samples must be at least 3 to support training visualizations")
        sample_count = min(max_train_samples, available_train_samples)
        train_set.samples = random.Random(SEED).sample(train_set.samples, sample_count)
    print(
        f"Training samples: {len(train_set)}/{available_train_samples}; "
        f"validation samples: {len(validation_set)}/{available_validation_samples}"
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=DEFAULT_INPUT_CHANNELS, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + _dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
        validation_metrics = _evaluate(model, validation_loader, device, threshold)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{epochs}: validation_loss={validation_metrics['loss']:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "in_channels": DEFAULT_INPUT_CHANNELS,
        "image_size": IMAGE_SIZE,
        "bands": ["B4", "B3", "B2", "B8"],
        "threshold": threshold,
        "task": "forest_non_forest_segmentation",
        "source": "https://zenodo.org/records/4498086",
    }, UNET_MODEL_PATH)

    model.load_state_dict(best_state)
    model.to(device)
    result = {
        "status": "trained",
        "source": "https://zenodo.org/records/4498086",
        "task": "forest_non_forest_segmentation",
        "trained_for_deforestation_labels": False,
        "deforestation_method": "forest mask before AND non-forest mask after",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "training_samples_used": len(train_set),
        "training_samples_available": available_train_samples,
        "validation_samples_used": len(validation_set),
        "validation_samples_available": available_validation_samples,
        "threshold": threshold,
        "dataset_inspection": inspection,
        "validation": _evaluate(model, validation_loader, device, threshold),
    }
    try:
        test_set = ForestSegmentationDataset(data_root, "test")
    except ValueError:
        test_set = None
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        result["test"] = _evaluate(model, test_loader, device, threshold)
        result["test_visualization"] = str(_save_test_predictions(model, test_set, device, threshold))

    result["training_visualization"] = str(_save_real_examples(train_set))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "forest_segmentation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS_DIR / "unet_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(inspect_forest_dataset(), indent=2))
