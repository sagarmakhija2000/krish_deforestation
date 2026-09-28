from __future__ import annotations

import argparse
import json
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from forest_monitoring.config import DEFAULT_FIRE_THRESHOLDS, FIRE_MODEL_PATH, RESULTS_DIR


def predict_fire_risk(features: Mapping[str, float]) -> dict:
    if not FIRE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained fire-risk model not found: {FIRE_MODEL_PATH}")
    bundle = joblib.load(FIRE_MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["features"]
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Missing fire-risk inputs: {', '.join(missing)}")

    input_row = pd.DataFrame([{column: float(features[column]) for column in feature_columns}])
    if not np.isfinite(input_row.to_numpy()).all():
        raise ValueError("Fire-risk inputs must be finite numeric values")
    probabilities = model.predict_proba(input_row)[0]
    positive_index = list(model.classes_).index(bundle["positive_class"])
    probability = float(probabilities[positive_index])

    if probability < DEFAULT_FIRE_THRESHOLDS["low"]:
        category = "Low"
    elif probability < DEFAULT_FIRE_THRESHOLDS["moderate"]:
        category = "Moderate"
    elif probability < DEFAULT_FIRE_THRESHOLDS["high"]:
        category = "High"
    else:
        category = "Extreme"

    result = {
        "fire_risk_probability": probability,
        "fire_risk_category": category,
        "input": {column: float(input_row.iloc[0][column]) for column in feature_columns},
        "model_source": bundle["source"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "fire_risk_prediction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["prediction_path"] = str(output_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fire risk from weather observations.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(predict_fire_risk({
        "temperature": args.temperature,
        "humidity": args.humidity,
        "wind_speed": args.wind_speed,
        "rainfall": args.rainfall,
    }), indent=2))
