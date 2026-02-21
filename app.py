import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "WS2M",
    "RH2M",
    "T2M",
    "ALLSKY_SFC_SW_DWN",
    "PM2_5_lag1",
    "PM2_5_lag3",
    "PM2_5_lag6",
    "WS2M_lag1",
    "WS2M_lag3",
    "WS2M_lag6",
]

AQI_COLOR_MAP = {
    "Good": "#2ecc71",
    "Satisfactory": "#27ae60",
    "Moderate": "#f1c40f",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad",
}


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").copy()
    for lag in [1, 3, 6]:
        df[f"PM2_5_lag{lag}"] = df["PM2_5"].shift(lag)
        df[f"WS2M_lag{lag}"] = df["WS2M"].shift(lag)
    return df.dropna().copy()


def calculate_pm25_aqi(pm25: float) -> int:
    if pm25 <= 30:
        return round((pm25 / 30) * 50)
    if pm25 <= 60:
        return round(((pm25 - 31) / (60 - 31)) * (100 - 51) + 51)
    if pm25 <= 90:
        return round(((pm25 - 61) / (90 - 61)) * (200 - 101) + 101)
    if pm25 <= 120:
        return round(((pm25 - 91) / (120 - 91)) * (300 - 201) + 201)
    if pm25 <= 250:
        return round(((pm25 - 121) / (250 - 121)) * (400 - 301) + 301)
    return round(((pm25 - 251) / (500 - 251)) * (500 - 401) + 401)


def aqi_category(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("output2.csv")
    test_df = pd.read_csv("test.csv")

    train_df["datetime"] = pd.to_datetime(train_df["datetime"], utc=True)
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], utc=True)

    return create_lag_features(train_df), create_lag_features(test_df)


@st.cache_resource
def load_model_and_scaler() -> tuple[object, StandardScaler]:
    train_lag_df, _ = load_data()
    scaler = StandardScaler()
    scaler.fit(train_lag_df[FEATURE_COLS])

    model = joblib.load("model.pkl")
    return model, scaler


def plot_aqi_bands(day_df: pd.DataFrame) -> None:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["AQI_predicted"],
            mode="lines+markers",
            name="Predicted AQI",
            line={"width": 3, "color": "#2d9cdb"},
            marker={"size": 8, "color": day_df["AQI_predicted"], "colorscale": "Turbo"},
            hovertemplate="Time: %{x}<br>AQI: %{y}<extra></extra>",
        )
    )

    fig.add_hrect(y0=0, y1=50, fillcolor="#2ecc71", opacity=0.08, line_width=0)
    fig.add_hrect(y0=51, y1=100, fillcolor="#27ae60", opacity=0.08, line_width=0)
    fig.add_hrect(y0=101, y1=200, fillcolor="#f1c40f", opacity=0.08, line_width=0)
    fig.add_hrect(y0=201, y1=300, fillcolor="#e67e22", opacity=0.08, line_width=0)
    fig.add_hrect(y0=301, y1=500, fillcolor="#e74c3c", opacity=0.08, line_width=0)

    fig.update_layout(
        title="Predicted AQI with Severity Bands",
        xaxis_title="Time (UTC)",
        yaxis_title="AQI",
        yaxis={"range": [0, max(500, int(day_df["AQI_predicted"].max()) + 20)]},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=430,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_pm25_comparison(day_df: pd.DataFrame) -> None:
    fig = go.Figure()

    if "PM2_5" in day_df.columns:
        fig.add_trace(
            go.Scatter(
                x=day_df["datetime"],
                y=day_df["PM2_5"],
                mode="lines+markers",
                name="Actual PM2.5",
                line={"width": 2.5, "color": "#34495e"},
                marker={"size": 7},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["PM2_5_pred"],
            mode="lines+markers",
            name="Predicted PM2.5",
            line={"width": 3, "color": "#ff4b4b"},
            marker={"size": 7},
        )
    )

    fig.update_layout(
        title="Hourly PM2.5: Actual vs Predicted",
        xaxis_title="Time (UTC)",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_aqi_bar(day_df: pd.DataFrame) -> None:
    chart_df = day_df.copy()
    chart_df["hour"] = chart_df["datetime"].dt.strftime("%H:%M")
    bar_colors = chart_df["AQI_pred_cat"].map(AQI_COLOR_MAP)

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["hour"],
                y=chart_df["AQI_predicted"],
                marker={"color": bar_colors, "line": {"color": "#ffffff", "width": 1}},
                text=chart_df["AQI_pred_cat"],
                textposition="outside",
                hovertemplate="Hour: %{x}<br>AQI: %{y}<br>Category: %{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Hourly Predicted AQI (Category Colored)",
        xaxis_title="Hour (UTC)",
        yaxis_title="AQI",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide")
    st.title("AQI Prediction Dashboard")
    st.write("Select a date to predict hourly AQI using your trained model.")

    _, test_lag_df = load_data()
    model, scaler = load_model_and_scaler()

    available_dates = sorted(test_lag_df["datetime"].dt.date.unique())
    selected_date = st.date_input(
        "Choose date",
        value=available_dates[0],
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )

    day_df = test_lag_df[test_lag_df["datetime"].dt.date == selected_date].copy()

    if day_df.empty:
        st.warning("No data available for the selected date.")
        return

    x_day_scaled = scaler.transform(day_df[FEATURE_COLS])
    day_df["PM2_5_pred"] = model.predict(x_day_scaled).clip(min=0)
    day_df["AQI_predicted"] = day_df["PM2_5_pred"].apply(calculate_pm25_aqi)
    day_df["AQI_pred_cat"] = day_df["AQI_predicted"].apply(aqi_category)

    if "PM2_5" in day_df.columns:
        day_df["AQI_actual"] = day_df["PM2_5"].apply(calculate_pm25_aqi)
        day_df["AQI_actual_cat"] = day_df["AQI_actual"].apply(aqi_category)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Predicted AQI", int(day_df["AQI_predicted"].mean()))
    col2.metric("Peak Predicted AQI", int(day_df["AQI_predicted"].max()))
    col3.metric("Average Predicted PM2.5", f"{day_df['PM2_5_pred'].mean():.2f}")

    st.subheader("Predictions Table")
    show_cols = ["datetime", "PM2_5_pred", "AQI_predicted", "AQI_pred_cat"]
    if "PM2_5" in day_df.columns:
        show_cols = [
            "datetime",
            "PM2_5",
            "PM2_5_pred",
            "AQI_actual",
            "AQI_predicted",
            "AQI_actual_cat",
            "AQI_pred_cat",
        ]

    st.dataframe(day_df[show_cols].sort_values("datetime"), use_container_width=True)

    st.subheader("Charts")
    left_col, right_col = st.columns(2)
    with left_col:
        plot_aqi_bands(day_df)
    with right_col:
        plot_pm25_comparison(day_df)
    plot_hourly_aqi_bar(day_df)


if __name__ == "__main__":
    main()
