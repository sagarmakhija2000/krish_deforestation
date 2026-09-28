from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from forest_monitoring.config import SATELLITE_AFTER_PATH, SATELLITE_BEFORE_PATH, SATELLITE_MASK_PATH


def load_image_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            arr = src.read()
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] >= 4:
                arr = arr[:4]
            arr = np.moveaxis(arr, 0, -1)
            # convert a multi-band raster into RGB-like visual data whenever necessary
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            return arr.astype(np.float32)

    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    return img


def compute_ndvi(image: np.ndarray) -> np.ndarray:
    """Compute NDVI from a three-band image or raster-like data.

    For user-uploaded Sentinel-2/Landsat data, the red and NIR bands are typically present.
    In this lightweight prototype, the code accepts either a 4-band raster or a standard RGB image,
    and approximates NDVI from the red and near-infrared channels when available.
    """
    if image.shape[-1] >= 4:
        red = image[..., 2].astype(np.float32)
        nir = image[..., 3].astype(np.float32)
    else:
        red = image[..., 0].astype(np.float32)
        nir = image[..., 1].astype(np.float32)

    denom = nir + red + 1e-8
    return np.divide(nir - red, denom, out=np.zeros_like(red, dtype=np.float32), where=denom != 0)


def deforestation_from_pair(before_image: np.ndarray, after_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    before_ndvi = compute_ndvi(before_image)
    after_ndvi = compute_ndvi(after_image)
    change = np.abs(after_ndvi - before_ndvi)
    threshold = 0.08
    mask = (change > threshold).astype(np.uint8)
    score = change / (change.max() + 1e-8)

    forest_loss_pct = float((mask.mean()) * 100.0)
    return mask, score, forest_loss_pct


def create_demo_satellite_pair(before_path: Path = SATELLITE_BEFORE_PATH, after_path: Path = SATELLITE_AFTER_PATH, mask_path: Path = SATELLITE_MASK_PATH) -> dict:
    """Legacy fallback for simple local testing only."""
    height, width = 128, 128
    y, x = np.mgrid[0:height, 0:width]
    forest_brightness = 90 + 18 * np.sin(x / 12) + 22 * np.cos(y / 15)
    forest_brightness = np.clip(forest_brightness, 20, 200)

    before = np.zeros((height, width, 3), dtype=np.uint8)
    before[:, :, 0] = np.clip(forest_brightness.astype(int), 0, 255)
    before[:, :, 1] = np.clip(forest_brightness.astype(int) + 25, 0, 255)
    before[:, :, 2] = np.clip(forest_brightness.astype(int) - 15, 0, 255)

    yy, xx = np.mgrid[0:height, 0:width]
    deforested = ((xx - 84) ** 2 + (yy - 64) ** 2 <= 28**2) | ((xx - 34) ** 2 + (yy - 102) ** 2 <= 22**2)
    after = before.copy()
    after[deforested, 0] = np.clip(after[deforested, 0] - 60, 0, 255)
    after[deforested, 1] = np.clip(after[deforested, 1] - 75, 0, 255)
    after[deforested, 2] = np.clip(after[deforested, 2] - 35, 0, 255)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[deforested] = 1

    Image.fromarray(before).save(before_path)
    Image.fromarray(after).save(after_path)
    Image.fromarray(mask * 255).save(mask_path)

    return {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "mask_path": str(mask_path),
        "forest_area_before": 0.0,
        "forest_area_after": 0.0,
        "deforested_pixels": int(mask.sum()),
    }


def load_demo_pair(before_path: Path = SATELLITE_BEFORE_PATH, after_path: Path = SATELLITE_AFTER_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = load_image_array(before_path)
    after = load_image_array(after_path)
    mask = np.array(Image.open(SATELLITE_MASK_PATH).convert("L"), dtype=np.float32) / 255.0 if Path(SATELLITE_MASK_PATH).exists() else np.zeros_like(before[:, :, 0], dtype=np.float32)
    return before, after, mask
