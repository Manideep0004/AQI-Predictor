from datetime import date

from fastapi import FastAPI, HTTPException

from core.charts import (
    build_aqi_bands_chart,
    build_hourly_aqi_bar_chart,
    build_pm25_comparison_chart,
)
from core.data import get_available_dates, get_day_dataframe
from core.model import enrich_with_predictions
from schemas import (
    ChartResponse,
    DateListResponse,
    DayPredictionResponse,
    HourlyPrediction,
    PredictionMetrics,
)

app = FastAPI(
    title="AQI Prediction API",
    description="Hourly PM2.5 and AQI prediction service (formerly Streamlit dashboard).",
    version="1.0.0",
)


def _get_predicted_day(selected_date: date):
    day_df = get_day_dataframe(selected_date)
    if day_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for date {selected_date.isoformat()}.",
        )
    return enrich_with_predictions(day_df)


def _build_prediction_rows(day_df) -> list[HourlyPrediction]:
    rows: list[HourlyPrediction] = []
    for _, row in day_df.iterrows():
        rows.append(
            HourlyPrediction(
                datetime=row["datetime"].to_pydatetime(),
                pm25_pred=float(row["PM2_5_pred"]),
                aqi_predicted=int(row["AQI_predicted"]),
                aqi_pred_cat=row["AQI_pred_cat"],
                pm25_actual=float(row["PM2_5"]) if "PM2_5" in day_df.columns else None,
                aqi_actual=int(row["AQI_actual"]) if "AQI_actual" in day_df.columns else None,
                aqi_actual_cat=row["AQI_actual_cat"] if "AQI_actual_cat" in day_df.columns else None,
            )
        )
    return rows


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dates", response_model=DateListResponse)
def list_available_dates() -> DateListResponse:
    """List all dates available in the test dataset for prediction."""
    return DateListResponse(dates=get_available_dates())


@app.get("/predictions/{selected_date}", response_model=DayPredictionResponse)
def get_day_predictions(selected_date: date) -> DayPredictionResponse:
    """
    Predict hourly PM2.5 and AQI for a selected date.

    Returns summary metrics and the full hourly predictions table
    (actual vs predicted when ground-truth PM2.5 is available).
    """
    day_df = _get_predicted_day(selected_date)

    metrics = PredictionMetrics(
        average_predicted_aqi=int(day_df["AQI_predicted"].mean()),
        peak_predicted_aqi=int(day_df["AQI_predicted"].max()),
        average_predicted_pm25=float(day_df["PM2_5_pred"].mean()),
    )

    return DayPredictionResponse(
        date=selected_date,
        metrics=metrics,
        predictions=_build_prediction_rows(day_df),
    )


@app.get("/predictions/{selected_date}/metrics", response_model=PredictionMetrics)
def get_day_metrics(selected_date: date) -> PredictionMetrics:
    """Average predicted AQI, peak predicted AQI, and average predicted PM2.5."""
    day_df = _get_predicted_day(selected_date)
    return PredictionMetrics(
        average_predicted_aqi=int(day_df["AQI_predicted"].mean()),
        peak_predicted_aqi=int(day_df["AQI_predicted"].max()),
        average_predicted_pm25=float(day_df["PM2_5_pred"].mean()),
    )


@app.get("/predictions/{selected_date}/table", response_model=list[HourlyPrediction])
def get_predictions_table(selected_date: date) -> list[HourlyPrediction]:
    """Hourly prediction table with actual vs predicted values when available."""
    day_df = _get_predicted_day(selected_date)
    return _build_prediction_rows(day_df)


@app.get("/charts/{selected_date}/aqi-bands", response_model=ChartResponse)
def get_aqi_bands_chart(selected_date: date) -> ChartResponse:
    """Predicted AQI time series with severity band overlays (Plotly JSON)."""
    day_df = _get_predicted_day(selected_date)
    return ChartResponse(
        date=selected_date,
        chart_type="aqi_bands",
        figure=build_aqi_bands_chart(day_df),
    )


@app.get("/charts/{selected_date}/pm25-comparison", response_model=ChartResponse)
def get_pm25_comparison_chart(selected_date: date) -> ChartResponse:
    """Hourly PM2.5 actual vs predicted comparison chart (Plotly JSON)."""
    day_df = _get_predicted_day(selected_date)
    return ChartResponse(
        date=selected_date,
        chart_type="pm25_comparison",
        figure=build_pm25_comparison_chart(day_df),
    )


@app.get("/charts/{selected_date}/hourly-aqi-bar", response_model=ChartResponse)
def get_hourly_aqi_bar_chart(selected_date: date) -> ChartResponse:
    """Category-colored hourly predicted AQI bar chart (Plotly JSON)."""
    day_df = _get_predicted_day(selected_date)
    return ChartResponse(
        date=selected_date,
        chart_type="hourly_aqi_bar",
        figure=build_hourly_aqi_bar_chart(day_df),
    )
