from __future__ import annotations

import csv
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

from forest_monitoring.config import FIRE_FEATURES_PATH, FIRE_MODEL_PATH, REAL_FIRE_DATASET_PATH


def download_public_fire_dataset() -> Path:
    REAL_FIRE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        "https://archive.ics.uci.edu/static/public/547/algerian+forest+fires+dataset.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/Algerian_forest_fires_dataset.csv",
    ]

    for url in candidates:
        try:
            response = requests.get(url, timeout=40)
            if response.status_code != 200:
                continue
            if url.endswith(".zip"):
                with zipfile.ZipFile(BytesIO(response.content)) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv"):
                            csv_bytes = zf.read(name)
                            REAL_FIRE_DATASET_PATH.write_bytes(csv_bytes)
                            return REAL_FIRE_DATASET_PATH
            else:
                REAL_FIRE_DATASET_PATH.write_bytes(response.content)
                return REAL_FIRE_DATASET_PATH
        except Exception:
            continue

    raise FileNotFoundError("Unable to download the Algerian forest fire dataset from the public UCI source.")


def prepare_real_fire_dataset() -> pd.DataFrame:
    if not REAL_FIRE_DATASET_PATH.exists():
        download_public_fire_dataset()

    lines = REAL_FIRE_DATASET_PATH.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    records: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.lower().startswith("day,month,year"):
            header = [column.strip().lower() for column in next(csv.reader([line]))]
            index += 1
            while index < len(lines) and lines[index].strip():
                row = next(csv.reader([lines[index]]))
                if len(row) == len(header):
                    records.append({column: value.strip() for column, value in zip(header, row)})
                index += 1
            continue
        index += 1

    df = pd.DataFrame.from_records(records)
    df = df.rename(columns={
        "temperature": "temperature",
        "rh": "humidity",
        "ws": "wind_speed",
        "rain": "rainfall",
        "classes": "fire_class",
    })
    numeric_features = ["temperature", "humidity", "wind_speed", "rainfall"]
    missing = set(numeric_features + ["fire_class"]) - set(df.columns)
    if missing:
        raise ValueError(f"Fire dataset is missing required fields: {sorted(missing)}")
    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["fire_class"] = df["fire_class"].astype(str).str.strip().str.lower()
    df = df[df["fire_class"].isin({"fire", "not fire"})].dropna(subset=numeric_features).copy()
    df["fire_occurred"] = df["fire_class"].eq("fire").astype(int)
    if df["fire_occurred"].nunique() != 2:
        raise ValueError("Fire dataset must contain both 'fire' and 'not fire' examples")
    return df.reset_index(drop=True)


def train_fire_model() -> dict:
    df = prepare_real_fire_dataset()
    feature_columns = ["temperature", "humidity", "wind_speed", "rainfall"]
    X = df[feature_columns]
    y = df["fire_occurred"]

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X, y)

    FIRE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feature_columns,
        "positive_class": 1,
        "source": "https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset",
    }, FIRE_MODEL_PATH)
    FIRE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIRE_FEATURES_PATH, index=False)

    return {
        "status": "trained",
        "source": "https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset",
        "training_samples": len(df),
        "features": feature_columns,
        "model_path": str(FIRE_MODEL_PATH),
    }


if __name__ == "__main__":
    print(train_fire_model())
