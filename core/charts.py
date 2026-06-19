import json

import pandas as pd
import plotly.graph_objects as go

from core.constants import AQI_COLOR_MAP


def _figure_to_dict(fig: go.Figure) -> dict:
    """Return a JSON-serializable Plotly figure (no numpy/pandas types)."""
    return json.loads(fig.to_json())


def build_aqi_bands_chart(day_df: pd.DataFrame) -> dict:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["AQI_predicted"],
            mode="lines+markers",
            name="Predicted AQI",
            line={"width": 3, "color": "#2d9cdb"},
            marker={"size": 8, "color": day_df["AQI_predicted"].tolist(), "colorscale": "Turbo"},
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

    return _figure_to_dict(fig)


def build_pm25_comparison_chart(day_df: pd.DataFrame) -> dict:
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

    return _figure_to_dict(fig)


def build_hourly_aqi_bar_chart(day_df: pd.DataFrame) -> dict:
    chart_df = day_df.copy()
    chart_df["hour"] = chart_df["datetime"].dt.strftime("%H:%M")
    bar_colors = chart_df["AQI_pred_cat"].map(AQI_COLOR_MAP)

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["hour"],
                y=chart_df["AQI_predicted"],
                marker={"color": bar_colors.tolist(), "line": {"color": "#ffffff", "width": 1}},
                text=chart_df["AQI_pred_cat"].tolist(),
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

    return _figure_to_dict(fig)
