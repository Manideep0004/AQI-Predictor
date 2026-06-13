from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.aqi import aqi_category, calculate_pm25_aqi
from core.constants import FEATURE_COLS
from core.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_model_and_scaler() -> tuple[object, StandardScaler]:
    train_lag_df, _ = load_data()
    scaler = StandardScaler()
    scaler.fit(train_lag_df[FEATURE_COLS])
    model = joblib.load(PROJECT_ROOT / "model.pkl")
    return model, scaler


def enrich_with_predictions(day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df.empty:
        return day_df

    model, scaler = load_model_and_scaler()
    x_day_scaled = scaler.transform(day_df[FEATURE_COLS])
    day_df = day_df.copy()
    day_df["PM2_5_pred"] = model.predict(x_day_scaled).clip(min=0)
    day_df["AQI_predicted"] = day_df["PM2_5_pred"].apply(calculate_pm25_aqi)
    day_df["AQI_pred_cat"] = day_df["AQI_predicted"].apply(aqi_category)

    if "PM2_5" in day_df.columns:
        day_df["AQI_actual"] = day_df["PM2_5"].apply(calculate_pm25_aqi)
        day_df["AQI_actual_cat"] = day_df["AQI_actual"].apply(aqi_category)

    return day_df
