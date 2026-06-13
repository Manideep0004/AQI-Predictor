import pandas as pd


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").copy()
    for lag in [1, 3, 6]:
        df[f"PM2_5_lag{lag}"] = df["PM2_5"].shift(lag)
        df[f"WS2M_lag{lag}"] = df["WS2M"].shift(lag)
    return df.dropna().copy()
