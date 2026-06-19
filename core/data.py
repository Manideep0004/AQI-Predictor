from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core.constants import FEATURE_COLS
from core.features import create_lag_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROJECT_ROOT / "output2.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "test.csv")

    train_df["datetime"] = pd.to_datetime(train_df["datetime"], utc=True)
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], utc=True)

    return create_lag_features(train_df), create_lag_features(test_df)


def get_available_dates() -> list[date]:
    _, test_lag_df = load_data()
    return sorted(test_lag_df["datetime"].dt.date.unique())


def get_day_dataframe(selected_date: date) -> pd.DataFrame:
    _, test_lag_df = load_data()
    day_df = test_lag_df[test_lag_df["datetime"].dt.date == selected_date].copy()
    return day_df.sort_values("datetime")
