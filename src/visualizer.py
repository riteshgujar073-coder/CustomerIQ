"""
Interactive Plotly visualizations for CustomerIQ.
Produces modern dark/light themed graphs with tailored palettes.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List

DARK_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#EC4899", "#8B5CF6", "#14B8A6"]

def plot_elbow_and_silhouette(eval_results: Dict[str, Any]) -> go.Figure:
    """
    Dual-axis interactive line chart displaying WCSS (Elbow) and Silhouette Score.
    """
    k_vals = eval_results["k_values"]
    inertias = eval_results["inertias"]
    silhouettes = eval_results["silhouettes"]
    opt_k = eval_results["recommended_k"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Inertia line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            name="Inertia / WCSS (Elbow)",
            mode="lines+markers",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=8, color="#818CF8")
        ),
        secondary_y=False
    )

    # Silhouette line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=silhouettes,
            name="Silhouette Score",
            mode="lines+markers",
            line=dict(color="#10B981", width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond", color="#34D399")
        ),
        secondary_y=True
    )

    # Highlight optimal k
    fig.add_vline(
        x=opt_k,
        line_width=2,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text=f"Recommended K={opt_k}",
        annotation_position="top left",
        annotation_font_color="#F59E0B"
    )

    fig.update_layout(
        title=dict(text="<b>Optimal Cluster Selection (Elbow Method & Silhouette Score)</b>", font=dict(size=16)),
        template=DARK_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)"
    )
    fig.update_xaxes(title_text="Number of Clusters (k)", tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Within-Cluster Sum of Squares (Inertia)", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)

    return fig

def plot_pca_2d(df: pd.DataFrame, cluster_col: str, centroids_pca: np.ndarray = None) -> go.Figure:
    """
    2D PCA scatter plot of customer segments with custom hover tooltips.
    """
    hover_cols = [c for c in ["CustomerID", "Monetary_Spend", "Frequency", "Recency_Days", "Avg_Order_Value", "Preferred_Category"] if c in df.columns]
    
    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=cluster_col,
        hover_data=hover_cols,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.82,
        title="<b>Customer Segments in 2D Feature Space (PCA Projections)</b>"
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))

    # Overlay centroids if provided
    if centroids_pca is not None:
        fig.add_trace(
            go.Scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode="markers",
                name="Cluster Centroids",
                marker=dict(size=14, color="#F59E0B", symbol="star-diamond", line=dict(width=2, color="white"))
            )
        )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(title="Segment")
    )
    return fig

def plot_pca_3d(df: pd.DataFrame, cluster_col: str) -> go.Figure:
    """
    3D PCA interactive projection of customer clusters.
    """
    hover_cols = [c for c in ["CustomerID", "Monetary_Spend", "Frequency", "Recency_Days"] if c in df.columns]
    
    fig = px.scatter_3d(
        df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color=cluster_col,
        hover_data=hover_cols,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.78,
        title="<b>Interactive 3D Customer Segmentation Space</b>"
    )
    
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.6)"),
            yaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.6)"),
            zaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.6)")
        )
    )
    return fig

def plot_cluster_radar(profiles: List[Dict[str, Any]], df: pd.DataFrame) -> go.Figure:
    """
    Multi-segment Radar / Spider chart comparing normalized behavioral metrics.
    """
    categories = ["Recency (Freshness)", "Frequency", "Monetary Spend", "Avg Order Value", "Website Visits", "Discount Sensitivity"]
    
    # Calculate min-max normalized cluster averages for clean radar comparison
    pop_mins = {
        "Recency_Days": df["Recency_Days"].min() if "Recency_Days" in df.columns else 0,
        "Frequency": df["Frequency"].min() if "Frequency" in df.columns else 0,
        "Monetary_Spend": df["Monetary_Spend"].min() if "Monetary_Spend" in df.columns else 0,
        "Avg_Order_Value": df["Avg_Order_Value"].min() if "Avg_Order_Value" in df.columns else 0,
        "Website_Visits": df["Website_Visits"].min() if "Website_Visits" in df.columns else 0,
        "Discount_Usage_Ratio": df["Discount_Usage_Ratio"].min() if "Discount_Usage_Ratio" in df.columns else 0,
    }
    pop_maxs = {
        "Recency_Days": df["Recency_Days"].max() if "Recency_Days" in df.columns else 1,
        "Frequency": df["Frequency"].max() if "Frequency" in df.columns else 1,
        "Monetary_Spend": df["Monetary_Spend"].max() if "Monetary_Spend" in df.columns else 1,
        "Avg_Order_Value": df["Avg_Order_Value"].max() if "Avg_Order_Value" in df.columns else 1,
        "Website_Visits": df["Website_Visits"].max() if "Website_Visits" in df.columns else 1,
        "Discount_Usage_Ratio": df["Discount_Usage_Ratio"].max() if "Discount_Usage_Ratio" in df.columns else 1,
    }

    fig = go.Figure()

    for p in profiles:
        # Invert recency so higher value = more recent (positive trait)
        rec_val = p.get("avg_recency") or 0.0
        rec_norm = 1.0 - ((rec_val - pop_mins["Recency_Days"]) / max(1e-5, (pop_maxs["Recency_Days"] - pop_mins["Recency_Days"])))
        
        freq_val = p.get("avg_frequency") or 0.0
        freq_norm = (freq_val - pop_mins["Frequency"]) / max(1e-5, (pop_maxs["Frequency"] - pop_mins["Frequency"]))
        
        mon_val = p.get("avg_monetary") or 0.0
        mon_norm = (mon_val - pop_mins["Monetary_Spend"]) / max(1e-5, (pop_maxs["Monetary_Spend"] - pop_mins["Monetary_Spend"]))

        aov_val = p.get("avg_aov") or 0.0
        aov_norm = (aov_val - pop_mins["Avg_Order_Value"]) / max(1e-5, (pop_maxs["Avg_Order_Value"] - pop_mins["Avg_Order_Value"]))

        vis_val = p.get("avg_visits") or 0.0
        vis_norm = (vis_val - pop_mins["Website_Visits"]) / max(1e-5, (pop_maxs["Website_Visits"] - pop_mins["Website_Visits"]))

        disc_val = (p.get("avg_discount") or 0.0) / 100.0
        disc_norm = (disc_val - pop_mins["Discount_Usage_Ratio"]) / max(1e-5, (pop_maxs["Discount_Usage_Ratio"] - pop_mins["Discount_Usage_Ratio"]))

        values = [rec_norm, freq_norm, mon_norm, aov_norm, vis_norm, disc_norm]
        values.append(values[0])  # Close radar loop
        r_categories = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=[round(v * 100, 1) for v in values],
                theta=r_categories,
                fill="toself",
                name=p["persona_name"],
                line=dict(color=p["color"], width=2),
                opacity=0.65
            )
        )

    fig.update_layout(
        title="<b>Comparative Segment DNA (Relative Strengths Across Core Dimensions)</b>",
        template=DARK_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
            bgcolor="rgba(15, 23, 42, 0.4)"
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def plot_segment_share_donuts(profiles: List[Dict[str, Any]]) -> go.Figure:
    """
    Side-by-side donut charts showing Customer Share vs Revenue Share.
    """
    labels = [p["persona_name"] for p in profiles]
    cust_counts = [p["customer_count"] for p in profiles]
    revenues = [p["total_revenue"] for p in profiles]
    colors = [p["color"] for p in profiles]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["<b>Customer Volume Share</b>", "<b>Total Revenue Contribution</b>"]
    )

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=cust_counts,
            name="Customers",
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="percent+label",
            hoverinfo="label+value+percent"
        ),
        1, 1
    )

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=revenues,
            name="Revenue",
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="percent+label",
            hoverinfo="label+value+percent"
        ),
        1, 2
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False
    )
    return fig

def plot_metric_boxplots(df: pd.DataFrame, cluster_col: str, metric_col: str, title: str) -> go.Figure:
    """
    Boxplot showing the distribution of a metric across clusters.
    """
    fig = px.box(
        df,
        x=cluster_col,
        y=metric_col,
        color=cluster_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        points="outliers",
        title=f"<b>{title}</b>"
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> go.Figure:
    """
    Correlation matrix heatmap for customer features.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    corr = df[valid_cols].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="<b>Customer Feature Correlation Heatmap</b>",
        aspect="auto"
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_conversion_lift_waterfall(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Waterfall / Bar chart comparing Baseline vs Personalized (+15%) conversions and revenue.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Baseline Orders",
        x=["Orders (Units)"],
        y=[sim_result["baseline_conversions"]],
        marker_color="#64748B"
    ))
    fig.add_trace(go.Bar(
        name="Incremental (+15% Lift)",
        x=["Orders (Units)"],
        y=[sim_result["incremental_conversions"]],
        marker_color="#10B981"
    ))

    fig.update_layout(
        barmode="stack",
        template=DARK_TEMPLATE,
        title=f"<b>Simulated Conversion Lift Impact: {sim_result['target_segment']}</b>",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
