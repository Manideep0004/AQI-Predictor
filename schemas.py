from datetime import date, datetime

from pydantic import BaseModel, Field


class DateListResponse(BaseModel):
    dates: list[date]


class PredictionMetrics(BaseModel):
    average_predicted_aqi: int
    peak_predicted_aqi: int
    average_predicted_pm25: float = Field(..., description="Average predicted PM2.5 (µg/m³)")


class HourlyPrediction(BaseModel):
    datetime: datetime
    pm25_pred: float
    aqi_predicted: int
    aqi_pred_cat: str
    pm25_actual: float | None = None
    aqi_actual: int | None = None
    aqi_actual_cat: str | None = None


class DayPredictionResponse(BaseModel):
    date: date
    metrics: PredictionMetrics
    predictions: list[HourlyPrediction]


class ChartResponse(BaseModel):
    date: date
    chart_type: str
    figure: dict
