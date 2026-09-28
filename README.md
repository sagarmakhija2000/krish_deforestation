# AI-Driven Forest Monitoring System

This project is a small academic ML prototype for the following objective:

- train a U-Net for forest/non-forest segmentation using public Sentinel-2 data
- derive deforestation by comparing forest masks predicted for two dates
- train a Random Forest model for forest-fire prediction using a public fire dataset
- use the trained models on new unseen inputs to generate a forest-vulnerability heatmap
- display the outputs in a simple Streamlit dashboard

This is not a production platform. It is designed to be simple, explainable, and suitable for a B.Tech micro-project.

## Core requirement and data policy

The project follows the requirement that both models must be trained on publicly available datasets from the internet, not on synthetic data.

### Public dataset sources used

1. Forest-fire dataset (real and public)
   - UCI Algerian Forest Fires dataset
   - Source: https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
   - The Random Forest uses temperature, relative humidity, wind speed, and rainfall to predict the fire/no-fire class.

2. Forest segmentation dataset (public Sentinel-2 semantic segmentation dataset)
   - Bragagnolo, L., da Silva, R. V., & Grzybowski, J. M. V. (2021), *Amazon and Atlantic Forest image datasets for semantic segmentation*.
   - Source: https://zenodo.org/records/4498086
   - Images are 512x512 four-band GeoTIFFs in B4, B3, B2, B8 order (RGB + NIR); masks are binary GeoTIFFs.
   - Published splits are train, validation, and test. Amazon contains 499 train, 100 validation, and 20 test images; Atlantic Forest contains 485 train, 100 validation, and 20 test images.

The U-Net is supervised by forest/non-forest masks, not deforestation labels. For a before/after pair, forest is predicted independently for each date and the deforestation mask is computed as forest-before AND non-forest-after. No synthetic labels or fabricated model performance are used.

## Project structure

- [forest_monitoring/config.py](forest_monitoring/config.py)
- [forest_monitoring/dashboard.py](forest_monitoring/dashboard.py)
- [forest_monitoring/preprocessing/satellite.py](forest_monitoring/preprocessing/satellite.py)
- [forest_monitoring/models/unet.py](forest_monitoring/models/unet.py)
- [forest_monitoring/training/train_unet.py](forest_monitoring/training/train_unet.py)
- [forest_monitoring/training/train_fire_model.py](forest_monitoring/training/train_fire_model.py)
- [forest_monitoring/prediction](forest_monitoring/prediction)
- [forest_monitoring/geospatial/fusion.py](forest_monitoring/geospatial/fusion.py)

## Required dataset folders

The extracted Amazon dataset is expected here:

- `forest_monitoring/data/amazon_sentinel/AMAZON/Training/`
- `forest_monitoring/data/amazon_sentinel/AMAZON/Validation/`
- `forest_monitoring/data/amazon_sentinel/AMAZON/Test/`

The loader pairs images and masks from the official split folders and validates the four image bands and 512x512 dimensions. It does not generate masks or substitute synthetic data. The current Amazon extraction contains 499 training, 100 validation, and 20 test pairs.

## Fire-risk training

The fire model is trained from the real public dataset downloaded from the UCI archive.

## U-Net training

The U-Net trains on the Zenodo forest-segmentation labels. Defaults are 10 epochs, a deterministic 100-image training subset, and 10 validation images per epoch. The test split remains separate and is evaluated after training. Pass `max_train_samples=None` to use all 499 Amazon training pairs. Training automatically uses CUDA when available.

## Test the models

Run these commands from the project root in PowerShell. Training writes the best U-Net checkpoint and test report/visualization under `forest_monitoring/models/` and `forest_monitoring/outputs/`.

```powershell
& ".\venv\Scripts\python.exe" -u -c "from forest_monitoring.training.train_fire_model import train_fire_model; print(train_fire_model())"
& ".\venv\Scripts\python.exe" -u -c "from forest_monitoring.training.train_unet import train_unet; print(train_unet())"
```

After both model files exist, start the dashboard:

```powershell
& ".\venv\Scripts\python.exe" -m streamlit run forest_monitoring/dashboard.py
```

Upload matching four-band before/after GeoTIFFs, enter temperature, humidity, wind speed, and rainfall, then run the analysis. Check `forest_monitoring/outputs/forest_segmentation_summary.json` and `forest_segmentation_test_predictions.png` for U-Net test results.

### Real before/after example

The following six-year-separated Sentinel-2 L2A scenes are cropped to the same 512x512 grid and are already in `forest_monitoring/data/internet_temporal_pairs/`:

- Before (2018-07-20): `before_2018-07-20_rondonia_4band.tif`
- After (2024-07-13): `after_2024-07-13_rondonia_4band.tif`

The crop intersects official INPE PRODES mapped deforestation from 2019. The dashboard displays its separate PRODES reference mask/overlay only for this exact image pair. This independent reference is not the U-Net prediction or a U-Net test label.

## New/unseen-data prediction workflow

The final demonstration is designed to work as follows:

1. Train the U-Net on labelled Sentinel-2 forest/non-forest masks
2. Train Random Forest on the public fire dataset
3. Save the trained models
4. Give aligned, same-region four-band Sentinel-2 images from two dates to the U-Net
5. Predict forest masks independently for the earlier and later satellite images
6. Derive deforestation from forest-before AND non-forest-after, and predict fire risk
7. Combine the two scores on a common grid into a vulnerability score
8. Display the final heatmap in the Streamlit dashboard

## Quick start

Install dependencies in the project environment, train both models as above, and launch the dashboard with the command shown in “Test the models.”

## Important notes

- The U-Net is trained for forest segmentation, not directly on deforestation labels.
- The deforestation mask is a derived comparison of two forest-segmentation predictions.
- The fire model uses a real public dataset from the UCI archive.
- If a labelled satellite segmentation dataset is not available, U-Net training stops with an error rather than fabricating samples or results.
- The dashboard accepts newly uploaded before/after satellite images and produces a change-detection and vulnerability output from those inputs.
