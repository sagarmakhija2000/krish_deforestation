from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from forest_monitoring.config import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_SEGMENTATION_THRESHOLD,
    IMAGE_SIZE,
    RESULTS_DIR,
    UNET_MODEL_PATH,
)
from forest_monitoring.models.unet import UNet


def _load_sentinel_image(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Inference requires a four-band Sentinel-2 GeoTIFF in B4, B3, B2, B8 order.")
    with rasterio.open(path) as src:
        image = src.read()
        metadata = {"crs": src.crs, "transform": src.transform, "bounds": src.bounds}
    if image.shape[0] != DEFAULT_INPUT_CHANNELS:
        raise ValueError(f"Expected {DEFAULT_INPUT_CHANNELS} bands in {path}; found {image.shape[0]}")
    if image.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} pixels in {path}; found {image.shape[1:]}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite values found in {path}")
    image = np.clip(image.astype(np.float32) / 10000.0, 0.0, 1.0)
    return image, metadata


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def predict_deforestation(
    before_path: str | Path,
    after_path: str | Path,
    model_path: str | Path = UNET_MODEL_PATH,
    threshold: float = DEFAULT_SEGMENTATION_THRESHOLD,
) -> dict:
    """Derive deforestation from independent forest-segmentation predictions."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained forest-segmentation model not found: {model_path}")
    before, before_metadata = _load_sentinel_image(before_path)
    after, after_metadata = _load_sentinel_image(after_path)
    if before.shape != after.shape:
        raise ValueError(f"Before/after images must have matching shapes; got {before.shape} and {after.shape}")
    if before_metadata["crs"] != after_metadata["crs"]:
        raise ValueError("Before/after Sentinel-2 images must use the same CRS.")
    if before_metadata["transform"] != after_metadata["transform"]:
        raise ValueError("Before/after Sentinel-2 images must cover the same region with the same pixel grid.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "forest_non_forest_segmentation":
        raise ValueError("Checkpoint is not a forest-segmentation model trained for this pipeline.")
    model = UNet(in_channels=checkpoint["in_channels"], out_channels=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    before_tensor = torch.from_numpy(before[None, :, :, :])
    after_tensor = torch.from_numpy(after[None, :, :, :])
    with torch.no_grad():
        forest_before = torch.sigmoid(model(before_tensor))[0, 0].numpy() >= threshold
        forest_after = torch.sigmoid(model(after_tensor))[0, 0].numpy() >= threshold

    deforestation = forest_before & ~forest_after
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    before_mask_path = RESULTS_DIR / "forest_mask_before.png"
    after_mask_path = RESULTS_DIR / "forest_mask_after.png"
    deforestation_mask_path = RESULTS_DIR / "deforestation_mask.png"
    _save_mask(forest_before, before_mask_path)
    _save_mask(forest_after, after_mask_path)
    _save_mask(deforestation, deforestation_mask_path)

    return {
        "task": "forest_non_forest_segmentation",
        "deforestation_is_derived_from_two_forest_predictions": True,
        "before_image": str(Path(before_path).resolve()),
        "after_image": str(Path(after_path).resolve()),
        "forest_pixels_before": int(forest_before.sum()),
        "forest_pixels_after": int(forest_after.sum()),
        "deforested_pixels": int(deforestation.sum()),
        "deforestation_percentage_of_region": float(deforestation.mean() * 100.0),
        "mask_paths": {
            "forest_mask_before": str(before_mask_path),
            "forest_mask_after": str(after_mask_path),
            "deforestation_mask": str(deforestation_mask_path),
        },
    }


if __name__ == "__main__":
    raise SystemExit("Call predict_deforestation(before_path, after_path) with two four-band Sentinel-2 GeoTIFF paths.")
