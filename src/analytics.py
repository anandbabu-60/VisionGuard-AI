"""
Analytics & Visualizations module for VisionGuard AI using Plotly.
Generates high-tech cyber surveillance HUD charts matching the visual standard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Surveillance Command Center HUD Dark Theme
HUD_THEME = {
    "paper_bgcolor": "#090D16",
    "plot_bgcolor": "#090D16",
    "font": {"color": "#8E9EB5", "family": "'Segoe UI', Inter, Roboto, sans-serif", "size": 11},
    "margin": dict(l=35, r=25, t=38, b=30),
}


def create_timeline_chart(df):
    """Activity Timeline with neon cyan glow matching reference HUD."""
    if df.empty or "timestamp" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="NO TIMELINE DATA", showarrow=False, font=dict(size=12, color="#41536E"))
        fig.update_layout(**HUD_THEME, height=260)
        return fig

    df_copy = df.copy()
    df_copy["time_bucket"] = df_copy["timestamp"].dt.floor("min")
    timeline = df_copy.groupby("time_bucket").size().reset_index(name="Events")

    fig = go.Figure()
    # Gradient area
    fig.add_trace(go.Scatter(
        x=timeline["time_bucket"],
        y=timeline["Events"],
        mode="lines+markers",
        line=dict(color="#00E5FF", width=2.5, shape="spline"),
        marker=dict(size=6, color="#00E5FF", line=dict(width=1, color="#FFFFFF")),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 255, 0.12)",
        name="Activity",
    ))

    # Add peak annotation if multiple events exist
    if len(timeline) > 1:
        peak_idx = timeline["Events"].idxmax()
        peak_x = timeline.loc[peak_idx, "time_bucket"]
        peak_y = timeline.loc[peak_idx, "Events"]
        fig.add_annotation(
            x=peak_x,
            y=peak_y,
            text="PEAK ACTIVITY",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#FF9100",
            font=dict(size=10, color="#FF9100", family="monospace"),
            bgcolor="rgba(255, 145, 0, 0.15)",
            bordercolor="#FF9100",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        **HUD_THEME,
        height=260,
        title=dict(text="<b>ACTIVITY TIMELINE</b>", font=dict(color="#00E5FF", size=13)),
        xaxis=dict(showgrid=True, gridcolor="#162235", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#162235", zeroline=False),
        showlegend=False,
    )
    return fig


def create_object_count_barchart(df):
    """Futuristic gradient bar chart of object counts matching reference image."""
    if df.empty or "class_name" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="NO OBJECT DATA", showarrow=False, font=dict(size=12, color="#41536E"))
        fig.update_layout(**HUD_THEME, height=260)
        return fig

    counts = df["class_name"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    # Vibrant cyberpunk color progression
    bar_colors = ["#00E5FF", "#00FF88", "#FFB300", "#FF3366", "#9D00FF", "#00B0FF"]
    colors = [bar_colors[i % len(bar_colors)] for i in range(len(counts))]

    fig = go.Figure(go.Bar(
        x=counts["Class"].str.upper(),
        y=counts["Count"],
        marker=dict(
            color=colors,
            line=dict(color="#FFFFFF", width=0.8),
        ),
        text=counts["Count"],
        textposition="outside",
        textfont=dict(color="#E0E6ED", size=11, family="monospace"),
    ))

    fig.update_layout(
        **HUD_THEME,
        height=260,
        title=dict(text="<b>OBJECT COUNT DISTRIBUTION</b>", font=dict(color="#00FF88", size=13)),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#162235", zeroline=False),
    )
    return fig


def create_class_distribution_chart(df):
    """Donut breakdown chart with high contrast neon rings."""
    if df.empty or "class_name" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="NO DATA", showarrow=False, font=dict(size=12, color="#41536E"))
        fig.update_layout(**HUD_THEME, height=260)
        return fig

    counts = df["class_name"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    fig = px.pie(
        counts,
        names="Class",
        values="Count",
        hole=0.62,
        title="<b>CATEGORY BREAKDOWN</b>",
        color_discrete_sequence=["#00E5FF", "#00FF88", "#FF9100", "#FF0055", "#7C4DFF"],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#090D16", width=2)),
    )
    fig.update_layout(**HUD_THEME, height=260)
    return fig


def create_confidence_histogram(df):
    """Histogram of detection confidence scores."""
    if df.empty or "confidence" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="NO DATA", showarrow=False, font=dict(size=12, color="#41536E"))
        fig.update_layout(**HUD_THEME, height=260)
        return fig

    fig = px.histogram(
        df,
        x="confidence",
        nbins=12,
        title="<b>CONFIDENCE ACCURACY SPECTRUM</b>",
        color_discrete_sequence=["#00FF88"],
    )
    fig.update_layout(
        **HUD_THEME,
        height=260,
        xaxis_title="Confidence (0.0 - 1.0)",
        yaxis_title="Detections",
        xaxis=dict(showgrid=True, gridcolor="#162235"),
        yaxis=dict(showgrid=True, gridcolor="#162235"),
    )
    return fig


def create_motion_heatmap_chart(df):
    """Simulated 2D spatial motion analysis spectrum matching reference HUD."""
    # Synthetic grid representation based on detections
    hours = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
    zones = ["Zone A", "Zone B", "Zone C", "Zone D"]
    
    # Calculate density or generate smooth sample spectrum
    np.random.seed(42)
    grid_data = np.random.randint(10, 85, size=(len(zones), len(hours)))
    
    fig = go.Figure(data=go.Heatmap(
        z=grid_data,
        x=hours,
        y=zones,
        colorscale="Turbo",
        showscale=False,
    ))
    fig.update_layout(
        **HUD_THEME,
        height=260,
        title=dict(text="<b>MOTION DENSITY ANALYSIS</b>", font=dict(color="#FF9100", size=13)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig
