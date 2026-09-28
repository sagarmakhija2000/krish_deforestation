from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forest_monitoring.config import RESULTS_DIR, VULNERABILITY_WEIGHTS


def compute_vulnerability(deforestation_score: np.ndarray, fire_risk_probability: float) -> dict:
    deforestation_score = np.asarray(deforestation_score, dtype=np.float32)
    if deforestation_score.ndim != 2:
        raise ValueError("Deforestation score must be a 2D mask on the target spatial grid")
    if not np.isfinite(deforestation_score).all():
        raise ValueError("Deforestation score contains non-finite values")
    if not np.isfinite(fire_risk_probability) or not 0 <= fire_risk_probability <= 1:
        raise ValueError("Fire-risk probability must be between 0 and 1")

    deforestation_score = np.clip(deforestation_score, 0.0, 1.0)
    fire_risk_grid = np.full(deforestation_score.shape, fire_risk_probability, dtype=np.float32)
    vulnerability = (
        VULNERABILITY_WEIGHTS["deforestation"] * deforestation_score
        + VULNERABILITY_WEIGHTS["fire_risk"] * fire_risk_grid
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = RESULTS_DIR / "vulnerability_heatmap.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(vulnerability, cmap="inferno", vmin=0, vmax=1)
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, label="Vulnerability score", fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "weights": dict(VULNERABILITY_WEIGHTS),
        "grid_shape": list(vulnerability.shape),
        "fire_risk_probability": float(fire_risk_probability),
        "deforestation_score_mean": float(deforestation_score.mean()),
        "vulnerability_score_mean": float(vulnerability.mean()),
        "vulnerability_score_min": float(vulnerability.min()),
        "vulnerability_score_max": float(vulnerability.max()),
        "heatmap_path": str(heatmap_path),
        "fire_risk_spatial_assumption": "one area-level probability is broadcast uniformly across the image grid",
    }
    (RESULTS_DIR / "vulnerability_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
