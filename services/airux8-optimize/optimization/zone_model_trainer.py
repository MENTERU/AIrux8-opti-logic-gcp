"""
Train per-zone power and indoor-temperature models for the optimizer model path.

Uses BASE_FEATURES only (inference can supply all of them from forecast + candidate settings).
Saves one artifact per zone to 03_Models/{store}_{zone_safe}.joblib with:
  power_model, temp_model, feature_names, impute_means
"""

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.config_train import BASE_FEATURES, TARGET_POWER, TARGET_TEMP
from config.utils import get_data_path

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    Ridge = None
    GradientBoostingRegressor = None
    Pipeline = None
    StandardScaler = None


def _zone_to_safe(zone: str) -> str:
    """Zone name to filename-safe string (e.g. 'Area 1' -> 'Area_1')."""
    return zone.replace(" ", "_")


def _safe_to_zone(safe: str) -> str:
    """Filename-safe string back to zone name."""
    return safe.replace("_", " ")


def _ensure_ac_status(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure A/C Status column: OFF=0, COOL=1, HEAT=2, FAN=3 from A/C Mode and A/C ON/OFF."""
    if "A/C Status" in df.columns:
        return df
    out = df.copy()
    if "A/C ON/OFF" in out.columns and "A/C Mode" in out.columns:
        # OFF when no units on, else use Mode (1=COOL, 2=HEAT, 3=FAN)
        out["A/C Status"] = np.where(
            (out["A/C ON/OFF"].fillna(0) == 0) | (out["A/C ON/OFF"].isna()),
            0,
            out["A/C Mode"].fillna(0),
        )
    else:
        out["A/C Status"] = 0
    return out


def _select_available_features(df: pd.DataFrame, requested: List[str]) -> List[str]:
    """Return requested feature names that exist in df."""
    return [c for c in requested if c in df.columns]


def train_zone_model(
    zone_df: pd.DataFrame,
    zone: str,
    power_model_type: str = "ridge",
    temp_model_type: str = "ridge",
    min_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Train power and temp models for one zone.

    Args:
        zone_df: Historical features for this zone (must include BASE_FEATURES and targets).
        zone: Zone name (for logging).
        power_model_type: 'ridge' or 'gb'.
        temp_model_type: 'ridge' or 'gb'.
        min_samples: Minimum rows to train.

    Returns:
        Dict with power_model, temp_model, feature_names, impute_means, or None if insufficient data.
    """
    if Ridge is None or Pipeline is None or StandardScaler is None:
        logging.warning("sklearn not available; zone model training skipped.")
        return None

    zone_df = _ensure_ac_status(zone_df)
    feature_cols = _select_available_features(zone_df, BASE_FEATURES)
    if not feature_cols:
        logging.warning(f"Zone {zone}: no BASE_FEATURES available in data.")
        return None

    targets = [TARGET_POWER, TARGET_TEMP]
    for t in targets:
        if t not in zone_df.columns:
            logging.warning(f"Zone {zone}: target {t} not in data.")
            return None

    # Drop rows with missing feature or target
    use_cols = feature_cols + targets
    clean = zone_df[use_cols].dropna(how="any")
    if len(clean) < min_samples:
        logging.warning(
            f"Zone {zone}: insufficient samples after dropna ({len(clean)} < {min_samples})."
        )
        return None

    X = clean[feature_cols].astype(float)
    y_power = clean[TARGET_POWER].astype(float)
    y_temp = clean[TARGET_TEMP].astype(float)

    # Impute means for inference when we don't have lags (e.g. first hour)
    impute_means = X.mean().to_dict()

    def _make_pipeline(estimator_type: str):
        if estimator_type == "gb" and GradientBoostingRegressor is not None:
            est = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
        else:
            est = Ridge(alpha=1.0, random_state=42)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", est),
            ]
        )

    pipe_power = _make_pipeline(power_model_type)
    pipe_temp = _make_pipeline(temp_model_type)
    pipe_power.fit(X, y_power)
    pipe_temp.fit(X, y_temp)

    return {
        "power_model": pipe_power,
        "temp_model": pipe_temp,
        "feature_names": feature_cols,
        "impute_means": impute_means,
    }


def save_zone_artifact(artifact: Dict[str, Any], store: str, zone: str) -> str:
    """Save artifact to 03_Models/{store}_{zone_safe}.joblib. Returns path."""
    models_dir = get_data_path("models_path")
    os.makedirs(models_dir, exist_ok=True)
    safe = _zone_to_safe(zone)
    path = os.path.join(models_dir, f"{store}_{safe}.joblib")
    joblib.dump(artifact, path)
    logging.info(f"Saved zone model: {path}")
    return path


def load_zone_artifact(store: str, zone: str, models_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load artifact for (store, zone). models_dir overrides config if provided."""
    if models_dir is None:
        models_dir = get_data_path("models_path")
    safe = _zone_to_safe(zone)
    path = os.path.join(models_dir, f"{store}_{safe}.joblib")
    if not os.path.isfile(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        logging.warning(f"Failed to load zone model {path}: {e}")
        return None


def load_all_zone_models_for_store(
    store: str, zones: Optional[List[str]] = None, models_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all zone artifacts for a store. If zones is None, discover from filenames {store}_*.joblib.

    Returns:
        Dict mapping zone name -> artifact.
    """
    if models_dir is None:
        models_dir = get_data_path("models_path")
    result = {}
    prefix = f"{store}_"
    suffix = ".joblib"
    if zones is not None:
        for zone in zones:
            art = load_zone_artifact(store, zone, models_dir)
            if art is not None:
                result[zone] = art
        return result
    if not os.path.isdir(models_dir):
        return result
    for f in os.listdir(models_dir):
        if f.startswith(prefix) and f.endswith(suffix):
            safe = f[len(prefix) : -len(suffix)]
            zone = _safe_to_zone(safe)
            art = load_zone_artifact(store, zone, models_dir)
            if art is not None:
                result[zone] = art
    return result


def train_and_save_all_zones(
    features_csv_path: str,
    store: str,
    power_model_type: str = "ridge",
    temp_model_type: str = "ridge",
    min_samples: int = 100,
) -> List[str]:
    """
    Load features CSV, train per zone, save to 03_Models. Returns list of saved paths.
    """
    df = pd.read_csv(features_csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = df["Datetime"].dt.date
    if "Hour" not in df.columns:
        df["Hour"] = df["Datetime"].dt.hour
    if "Month" not in df.columns:
        df["Month"] = df["Datetime"].dt.month
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    if "IsWeekend" not in df.columns:
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    if "IsHoliday" not in df.columns:
        df["IsHoliday"] = 0

    saved = []
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone].copy()
        artifact = train_zone_model(
            zone_df,
            zone,
            power_model_type=power_model_type,
            temp_model_type=temp_model_type,
            min_samples=min_samples,
        )
        if artifact is not None:
            save_zone_artifact(artifact, store, zone)
            saved.append(
                os.path.join(get_data_path("models_path"), f"{store}_{_zone_to_safe(zone)}.joblib")
            )
    return saved
