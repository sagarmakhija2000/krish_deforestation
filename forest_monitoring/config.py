from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SATELLITE_DIR = DATA_DIR / "satellite"
FIRE_DIR = DATA_DIR / "fire"
WEATHER_DIR = DATA_DIR / "weather"
SENTINEL_CHANGE_DIR = DATA_DIR / "sentinel_change"
FOREST_SEGMENTATION_DATA_DIR = DATA_DIR / "amazon_sentinel"

MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_DIR = PROJECT_ROOT / "training"
PREDICTION_DIR = PROJECT_ROOT / "prediction"
GEOSPATIAL_DIR = PROJECT_ROOT / "geospatial"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

UNET_MODEL_PATH = MODEL_DIR / "unet_best.pth"
FIRE_MODEL_PATH = MODEL_DIR / "fire_risk_model.joblib"

DEFAULT_FIRE_THRESHOLDS = {
    "low": 0.25,
    "moderate": 0.5,
    "high": 0.75,
    "extreme": 0.9,
}

VULNERABILITY_WEIGHTS = {
    "deforestation": 0.5,
    "fire_risk": 0.5,
}

SATELLITE_BEFORE_PATH = SATELLITE_DIR / "before.png"
SATELLITE_AFTER_PATH = SATELLITE_DIR / "after.png"
SATELLITE_MASK_PATH = SATELLITE_DIR / "deforestation_mask.png"
FIRE_FEATURES_PATH = FIRE_DIR / "fire_features.csv"
REAL_FIRE_DATASET_PATH = FIRE_DIR / "Algerian_forest_fires_dataset_UPDATE.csv"
FOREST_CHANGE_DATASET_NAME = "JimmyBrocko/Forest-Change"

MINI_BATCH_SIZE = 8
IMAGE_SIZE = 512
SEED = 42
DEFAULT_INPUT_CHANNELS = 4
SENTINEL_BANDS = ("B4", "B3", "B2", "B8")
DEFAULT_SEGMENTATION_THRESHOLD = 0.5

# Public sources used in this project:
# https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
# https://zenodo.org/records/4498086

# The U-Net is trained for Sentinel-2 forest/non-forest segmentation using B4, B3, B2, and B8.
# Deforestation is derived indirectly by comparing forest masks predicted for two dates.
