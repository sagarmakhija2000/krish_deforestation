from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forest_monitoring.config import FIRE_MODEL_PATH, UNET_MODEL_PATH
from forest_monitoring.geospatial.fusion import compute_vulnerability
from forest_monitoring.prediction.predict_deforestation import predict_deforestation
from forest_monitoring.prediction.predict_fire_risk import predict_fire_risk


st.set_page_config(page_title="Forest Monitoring Dashboard", layout="wide")
st.title("AI-Driven Forest Monitoring Dashboard")

with st.form("forest_vulnerability_form"):
    st.subheader("Sentinel-2 images")
    image_col1, image_col2 = st.columns(2)
    with image_col1:
        before_upload = st.file_uploader(
            "Before image (B4/B3/B2/B8 GeoTIFF)", type=["tif", "tiff"], key="before_image"
        )
    with image_col2:
        after_upload = st.file_uploader(
            "After image (same region and pixel grid)", type=["tif", "tiff"], key="after_image"
        )

    st.subheader("Weather conditions")
    weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
    with weather_col1:
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, 30.0)
    with weather_col2:
        humidity = st.slider("Humidity (%)", 5.0, 100.0, 55.0)
    with weather_col3:
        wind_speed = st.slider("Wind speed (km/h)", 0.0, 40.0, 15.0)
    with weather_col4:
        rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 8.0)
    run_analysis = st.form_submit_button("Run forest vulnerability analysis", type="primary")

if run_analysis:
    if not before_upload or not after_upload:
        st.error("Select both before and after Sentinel-2 GeoTIFF images.")
        st.stop()
    if not UNET_MODEL_PATH.exists():
        st.error(f"Forest segmentation model not found: {UNET_MODEL_PATH}")
        st.stop()
    if not FIRE_MODEL_PATH.exists():
        st.error(f"Fire-risk model not found: {FIRE_MODEL_PATH}")
        st.stop()

    with st.spinner("Segmenting both images and calculating the vulnerability map..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            before_path = Path(temp_dir) / Path(before_upload.name).name
            after_path = Path(temp_dir) / Path(after_upload.name).name
            before_path.write_bytes(before_upload.getvalue())
            after_path.write_bytes(after_upload.getvalue())
            try:
                change_result = predict_deforestation(before_path, after_path)
                with rasterio.open(before_path) as source:
                    before_rgb = source.read((1, 2, 3)).transpose(1, 2, 0)
                with rasterio.open(after_path) as source:
                    after_rgb = source.read((1, 2, 3)).transpose(1, 2, 0)
                before_rgb = np.clip(before_rgb.astype(np.float32) / 10000.0, 0.0, 1.0)
                after_rgb = np.clip(after_rgb.astype(np.float32) / 10000.0, 0.0, 1.0)

                fire_result = predict_fire_risk({
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "rainfall": rainfall,
                })
                deforestation_mask = np.asarray(
                    Image.open(change_result["mask_paths"]["deforestation_mask"]).convert("L"),
                    dtype=np.float32,
                ) / 255.0
                vulnerability_result = compute_vulnerability(
                    deforestation_mask,
                    fire_result["fire_risk_probability"],
                )
            except (ValueError, RuntimeError, OSError, KeyError, FileNotFoundError) as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    st.subheader("Input images")
    preview_col1, preview_col2 = st.columns(2)
    preview_col1.image(before_rgb, caption="Before image (B4/B3/B2)")
    preview_col2.image(after_rgb, caption="After image (B4/B3/B2)")

    st.subheader("Forest-change masks")
    mask_col1, mask_col2, mask_col3 = st.columns(3)
    mask_col1.image(change_result["mask_paths"]["forest_mask_before"], caption="Forest mask before")
    mask_col2.image(change_result["mask_paths"]["forest_mask_after"], caption="Forest mask after")
    mask_col3.image(change_result["mask_paths"]["deforestation_mask"], caption="Derived deforestation mask")

    reference_dir = Path(__file__).resolve().parent / "data" / "internet_temporal_pairs"
    reference_before = "before_2018-07-20_rondonia_4band.tif"
    reference_after = "after_2024-07-13_rondonia_4band.tif"
    if (
        Path(before_upload.name).name == reference_before
        and Path(after_upload.name).name == reference_after
        and (reference_dir / "prodes_loss_2019_2020_mask.png").is_file()
        and (reference_dir / "prodes_loss_2019_2020_overlay.png").is_file()
    ):
        st.subheader("Independent deforestation reference")
        st.caption("INPE PRODES mapped loss from 2019–2020; this reference is separate from the U-Net prediction.")
        reference_col1, reference_col2 = st.columns(2)
        reference_col1.image(reference_dir / "prodes_loss_2019_2020_mask.png", caption="PRODES mapped-loss mask")
        reference_col2.image(reference_dir / "prodes_loss_2019_2020_overlay.png", caption="PRODES loss over 2018 RGB")
        prodes_mask = np.asarray(
            Image.open(reference_dir / "prodes_loss_2019_2020_mask.png").convert("L"),
            dtype=np.uint8,
        ) > 0
        st.metric("INPE PRODES mapped loss (2019–2020)", f"{prodes_mask.mean() * 100:.2f}% of this chip")

    st.subheader("Combined risk")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("U-Net predicted deforestation", f"{change_result['deforestation_percentage_of_region']:.2f}%")
    score_col2.metric("Fire probability", f"{fire_result['fire_risk_probability']:.3f}")
    score_col2.caption(f"Fire-risk category: {fire_result['fire_risk_category']}")
    score_col3.metric("Mean vulnerability", f"{vulnerability_result['vulnerability_score_mean']:.3f}")
    st.image(
        vulnerability_result["heatmap_path"],
        caption="Vulnerability = 0.5 × deforestation mask + 0.5 × fire-risk probability",
    )
else:
    st.info("Choose aligned before/after Sentinel-2 GeoTIFFs, enter weather conditions, then run the analysis.")
