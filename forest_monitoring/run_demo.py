from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forest_monitoring.config import FIRE_MODEL_PATH, UNET_MODEL_PATH
from forest_monitoring.geospatial.fusion import compute_vulnerability
from forest_monitoring.training.train_fire_model import train_fire_model
from forest_monitoring.prediction.predict_deforestation import predict_deforestation
from forest_monitoring.prediction.predict_fire_risk import predict_fire_risk


def run_pipeline(
    before_path: str | Path,
    after_path: str | Path,
    temperature: float,
    humidity: float,
    wind_speed: float,
    rainfall: float,
) -> dict:
    if not UNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the U-Net on the Zenodo forest-segmentation dataset first: {UNET_MODEL_PATH}")

    if not FIRE_MODEL_PATH.exists():
        fire_result = train_fire_model()
    else:
        fire_result = {"status": "loaded", "model_path": str(FIRE_MODEL_PATH)}
    if not FIRE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Fire model was not created: {FIRE_MODEL_PATH}")

    # Generate predictions from real, user-supplied Sentinel-2 imagery.
    deforestation_result = predict_deforestation(before_path, after_path)
    fire_prediction_result = predict_fire_risk({
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rainfall": rainfall,
    })
    deforestation_mask = np.asarray(
        Image.open(deforestation_result["mask_paths"]["deforestation_mask"]).convert("L"),
        dtype=np.float32,
    ) / 255.0
    vulnerability_result = compute_vulnerability(
        deforestation_mask,
        fire_prediction_result["fire_risk_probability"],
    )

    summary = {
        "fire_metrics": fire_result,
        "deforestation": deforestation_result,
        "fire_risk": fire_prediction_result,
        "vulnerability": vulnerability_result,
    }

    out_path = Path(__file__).resolve().parent / "results" / "pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forest segmentation, fire risk, and vulnerability predictions.")
    parser.add_argument("before", help="Earlier four-band Sentinel-2 GeoTIFF (B4/B3/B2/B8)")
    parser.add_argument("after", help="Later four-band Sentinel-2 GeoTIFF for the same region")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    args = parser.parse_args()
    print(run_pipeline(
        args.before,
        args.after,
        temperature=args.temperature,
        humidity=args.humidity,
        wind_speed=args.wind_speed,
        rainfall=args.rainfall,
    ))
