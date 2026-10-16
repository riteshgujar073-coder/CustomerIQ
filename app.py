"""
CustomerIQ — AI-Powered Customer Segmentation & Analytics Platform
Modern Streamlit Dashboard with real ML clustering, Elbow & Silhouette evaluation,
interactive Plotly visualizations, customer personas, and marketing ROI simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_data,
    validate_dataset,
    get_dataset_kpis,
    auto_detect_columns,
    DEFAULT_FEATURE_COLUMNS,
    CORE_RFM_COLUMNS
)
from src.preprocessor import CustomerPreprocessor
from src.model import CustomerClusterEngine
from src.persona import generate_cluster_profiles
from src.simulator import simulate_campaign_roi, simulate_portfolio_impact
from src.visualizer import (
    plot_elbow_and_silhouette,
    plot_pca_2d,
    plot_pca_3d,
    plot_cluster_radar,
    plot_segment_share_donuts,
    plot_metric_boxplots,
    plot_correlation_heatmap,
    plot_conversion_lift_waterfall
)

# Page configuration
st.set_page_config(
    page_title="CustomerIQ — AI Customer Segmentation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Slate Modern Theme)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0B0F19 0%, #0F172A 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers & Badges */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818CF8, #34D399, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .brand-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #34D399;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Persona Card */
    .persona-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 14px;
        border-left: 5px solid #6366F1;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .persona-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .persona-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    /* Objective Pill */
    .objective-box {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 10px 0;
        color: #C7D2FE;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <div class="brand-title">CustomerIQ — AI-Powered Customer Segmentation & Analytics</div>
    <div class="brand-sub">Production Machine Learning Suite for Behavioral Clustering, RFM Profiling & Conversion Optimization</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Engine Settings & Data")

# Data Source Selection
data_option = st.sidebar.radio(
    "Data Source",
    ["Standard E-Commerce Dataset (2,500 records)", "Upload Custom Customer CSV"],
    index=0
)

uploaded_file = None
if data_option == "Upload Custom Customer CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    st.sidebar.caption("File should contain numeric customer interaction/transaction metrics.")

# Sidebar Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Clustering Hyperparameters")

use_auto_k = st.sidebar.checkbox("Auto-select Optimal K (Silhouette & Knee)", value=False)
k_slider = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=9, value=5, disabled=use_auto_k)

scaling_method = st.sidebar.selectbox(
    "Feature Scaling Method",
    ["standard", "robust", "minmax"],
    index=0,
    help="Standard: z-score normalization. Robust: median/IQR based (resilient to outliers)."
)

outlier_method = st.sidebar.selectbox(
    "Outlier Handling Method",
    ["clip", "remove", "none"],
    index=0,
    help="Clip: Winsorizes extreme outliers to IQR thresholds. Remove: Drops outlier records."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: CustomerIQ uses K-Means++ initialization with Euclidean distances in standardized feature space.")

# Data Loading & Processing Caching
@st.cache_data(show_spinner=False)
def load_and_prep_pipeline(source_path_or_bytes, is_uploaded: bool = False):
    if is_uploaded:
        df = load_data(source_path_or_bytes)
    else:
        df = load_data(os.path.join(PROJECT_ROOT, "data", "customer_data.csv"))
    return df

@st.cache_data(show_spinner=False)
def run_cluster_evaluation(df_data, feature_cols, scale_m, out_m):
    prep = CustomerPreprocessor(
        feature_cols=feature_cols,
        scaling_method=scale_m,
        outlier_method=out_m,
        pca_components=3
    )
    X_s, df_proc, meta = prep.fit_transform(df_data)
    eval_res = CustomerClusterEngine.evaluate_cluster_range(X_s, k_min=2, k_max=9)
    return X_s, df_proc, meta, eval_res, prep

# Determine Data Source
if data_option == "Upload Custom Customer CSV" and uploaded_file is not None:
    try:
        raw_df = load_and_prep_pipeline(uploaded_file, is_uploaded=True)
        st.sidebar.success(f"Uploaded {len(raw_df)} customer rows.")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    raw_df = load_and_prep_pipeline(None, is_uploaded=False)

# Validate dataset features
valid_features = [c for c in DEFAULT_FEATURE_COLUMNS if c in raw_df.columns]
if not valid_features or len(valid_features) < 3:
    # Fallback to all numeric columns except ID
    valid_features = [c for c in raw_df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]

# Run Evaluation and Processing
with st.spinner("Analyzing feature distributions and evaluating cluster quality..."):
    X_scaled, df_processed, prep_metadata, eval_results, preprocessor = run_cluster_evaluation(
        raw_df, valid_features, scaling_method, outlier_method
    )

# Selected K
n_clusters = eval_results["recommended_k"] if use_auto_k else k_slider

# Fit K-Means
@st.cache_data(show_spinner=False)
def fit_final_segmentation(X_arr, k_val, _df_proc, feat_cols):
    engine = CustomerClusterEngine(n_clusters=k_val)
    engine.fit(X_arr)
    df_result = _df_proc.copy()
    df_result["Cluster"] = engine.cluster_labels_
    
    profiles = generate_cluster_profiles(df_result, "Cluster", feat_cols)
    cluster_to_persona = {p["cluster_id"]: p["persona_name"] for p in profiles}
    df_result["Persona"] = df_result["Cluster"].map(cluster_to_persona)
    
    return engine, df_result, profiles

engine, df_clustered, profiles = fit_final_segmentation(X_scaled, n_clusters, df_processed, valid_features)
kpis = get_dataset_kpis(df_clustered)

# Project Context Accordion (Highlighting the E-commerce Personalization Goal from reference image)
with st.expander("📋 Project Context: E-Commerce Customer Segmentation Problem Statement & Goals", expanded=False):
    col_p1, col_p2, col_p3 = st.columns([1.2, 1, 1])
    with col_p1:
        st.markdown("**Problem Statement**")
        st.write(
            "An e-commerce company wants to personalize user experience by deploying a machine learning segmentation model "
            "to categorize customers based on preferences and transaction behavior, enabling targeted marketing and personalized recommendations."
        )
    with col_p2:
        st.markdown("**Key Objectives**")
        st.write("1. Develop customer segmentation using K-Means ML.\n2. Understand distinct customer segments & characteristics.\n3. Provide insights for personalized marketing.")
    with col_p3:
        st.markdown("**Milestone Goals**")
        st.write("🎯 **Silhouette Quality**: Balanced, distinct cluster boundaries.\n🏷️ **Top 3 Defining Characteristics** per segment.\n📈 **+15% Conversion Rate** lift through targeted personalization.")

# Navigation Tabs
tabs = st.tabs([
    "📊 Executive KPIs",
    "🔍 Data Exploration & Preprocessing",
    "🧠 Optimal K (Elbow & Silhouette)",
    "🌐 2D & 3D Visualizations",
    "👥 Customer Personas & Playbooks",
    "🚀 15% Conversion ROI Simulator",
    "📁 Customer Export & Scoring"
])

# ==============================================================================
# TAB 1: EXECUTIVE KPIS
# ==============================================================================
with tabs[0]:
    st.markdown("### Executive Performance Dashboard")
    
    # KPI Cards Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{kpis['total_customers']:,}</div>
            <div class="metric-delta">Active Profiles</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${kpis['total_revenue']:,.0f}</div>
            <div class="metric-delta">Gross Spend Tracked</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Order Value</div>
            <div class="metric-value">${kpis['avg_order_value']:.2f}</div>
            <div class="metric-delta">Basket Mean</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        churn_risk_count = sum(p["customer_count"] for p in profiles if "At-Risk" in p["persona_name"])
        churn_risk_pct = round((churn_risk_count / max(1, len(df_clustered))) * 100, 1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">At-Risk Segment</div>
            <div class="metric-value">{churn_risk_pct}%</div>
            <div class="metric-delta" style="color: #EF4444;">{churn_risk_count} Dormant Customers</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Quality</div>
            <div class="metric-value">{engine.metrics_['silhouette_score']:.3f}</div>
            <div class="metric-delta" style="color: #6366F1;">Silhouette Score (K={n_clusters})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Donut Charts & Summary Table
    col_chart, col_table = st.columns([1.2, 1])
    with col_chart:
        st.plotly_chart(plot_segment_share_donuts(profiles), use_container_width=True)
    with col_table:
        st.markdown("#### Segment Revenue & Headcount Summary")
        summary_rows = []
        for p in profiles:
            summary_rows.append({
                "Segment": f"{p['icon']} {p['persona_name']}",
                "Customers": f"{p['customer_count']:,} ({p['customer_pct']}%)",
                "Revenue ($)": f"${p['total_revenue']:,.2f}",
                "Rev Share": f"{p['revenue_pct']}%",
                "Avg AOV": f"${p['avg_aov']:.2f}" if p['avg_aov'] else "N/A"
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        
        st.info("💡 **Revenue Concentration Insight**: High-value champions typically generate a disproportionate share (35-45%) of total revenue despite being under 20% of customer count.")

# ==============================================================================
# TAB 2: DATA EXPLORATION & PREPROCESSING
# ==============================================================================
with tabs[1]:
    st.markdown("### Data Quality, Audit & Preprocessing Inspection")
    
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.markdown("#### Feature Statistics & Distribution Summary")
        st.dataframe(df_clustered[valid_features].describe().T[["mean", "std", "min", "50%", "max"]].round(2), use_container_width=True)
    
    with col_e2:
        st.markdown("#### Outlier Treatment Summary (IQR Bounds)")
        outlier_data = []
        for feat, info in prep_metadata["outlier_summary"]["per_feature"].items():
            outlier_data.append({
                "Feature": feat,
                "Outliers Detected": info["count"],
                "Outlier %": f"{info['pct']}%",
                "Lower Cap": info["lower_bound"],
                "Upper Cap": info["upper_bound"]
            })
        st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)
        st.caption(f"Strategy: **{outlier_method.upper()}** (Multiplied by {preprocessor.outlier_iqr_multiplier}x IQR)")

    st.markdown("---")
    col_heat, col_box = st.columns([1.1, 1])
    with col_heat:
        st.plotly_chart(plot_correlation_heatmap(df_clustered, valid_features), use_container_width=True)
    with col_box:
        metric_choice = st.selectbox(
            "Select Metric to Inspect Boxplot Across Segments",
            [c for c in ["Monetary_Spend", "Recency_Days", "Frequency", "Avg_Order_Value", "Discount_Usage_Ratio", "Website_Visits"] if c in df_clustered.columns]
        )
        st.plotly_chart(plot_metric_boxplots(df_clustered, "Persona", metric_choice, f"{metric_choice} by Customer Persona"), use_container_width=True)

# ==============================================================================
# TAB 3: OPTIMAL K (ELBOW & SILHOUETTE)
# ==============================================================================
with tabs[2]:
    st.markdown("### Cluster Number Selection: Elbow Method & Silhouette Analysis")
    st.markdown("Evaluating cluster compactness (WCSS) and separation (Silhouette Score) across $k \in [2, 9]$.")
    
    st.plotly_chart(plot_elbow_and_silhouette(eval_results), use_container_width=True)
    
    col_rec, col_met_table = st.columns([1, 1.2])
    with col_rec:
        st.markdown("#### 🎯 Mathematical Evaluation & Decision")
        st.markdown(f"""
        - **Optimal K (Silhouette Criterion)**: **`K = {eval_results['optimal_k_silhouette']}`** (Peak Silhouette Score = `{max(eval_results['silhouettes']):.4f}`)
        - **Optimal K (Elbow Knee Criterion)**: **`K = {eval_results['optimal_k_elbow']}`** (Max perpendicular distance from linear trend)
        - **Active Model Setting**: **`K = {n_clusters}`**
        """)
        
        st.markdown("""
        **Metric Interpretation Guide**:
        - **Inertia / WCSS**: Sum of squared distances to closest centroid. Decreases monotonically; the inflection point ("elbow") indicates diminishing returns.
        - **Silhouette Score**: Measures within-cluster cohesion vs nearest-cluster separation. Scores $> 0.35$ demonstrate meaningful customer grouping.
        - **Davies-Bouldin Index**: Ratio of within-cluster distances to between-cluster distances. Lower values indicate better clustering.
        - **Calinski-Harabasz Index**: Variance Ratio Criterion. Higher values indicate dense, well-separated clusters.
        """)
        
    with col_met_table:
        st.markdown("#### Multi-Metric Comparison Table")
        metric_rows = []
        for i, k in enumerate(eval_results["k_values"]):
            is_rec = " ⭐ Recommended" if k == eval_results["recommended_k"] else ""
            metric_rows.append({
                "Clusters (k)": f"k = {k}{is_rec}",
                "WCSS (Inertia)": f"{eval_results['inertias'][i]:,.0f}",
                "Silhouette Score": f"{eval_results['silhouettes'][i]:.4f}",
                "Davies-Bouldin": f"{eval_results['davies_bouldin'][i]:.4f}",
                "Calinski-Harabasz": f"{eval_results['calinski_harabasz'][i]:,.1f}"
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4: 2D & 3D VISUALIZATIONS
# ==============================================================================
with tabs[3]:
    st.markdown("### Dimensionality Reduction & Projection Space (PCA)")
    st.markdown(f"Principal Component Analysis captures **{prep_metadata['pca_2d_variance']}%** of total variance in 2D and **{prep_metadata['pca_3d_variance']}%** in 3D.")
    
    col_2d, col_3d = st.columns([1.1, 1.1])
    with col_2d:
        st.plotly_chart(plot_pca_2d(df_clustered, "Persona"), use_container_width=True)
    with col_3d:
        st.plotly_chart(plot_pca_3d(df_clustered, "Persona"), use_container_width=True)

    with st.expander("🔍 Inspect PCA Component Loadings (Feature Drivers)"):
        st.write("Loadings quantify how much each standardized customer feature contributes to each principal component:")
        loadings_df = pd.DataFrame(prep_metadata["pca_loadings"])
        st.dataframe(loadings_df.round(3), use_container_width=True)

# ==============================================================================
# TAB 5: CUSTOMER PERSONAS & PLAYBOOKS
# ==============================================================================
with tabs[4]:
    st.markdown("### Customer Segment Personas & Marketing Playbooks")
    st.markdown("Actionable profiles detailing top defining characteristics and conversion strategies.")
    
    # Multi-Cluster Radar Chart
    st.plotly_chart(plot_cluster_radar(profiles, df_clustered), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Segment Persona Deep-Dives")
    
    for p in profiles:
        with st.container():
            st.markdown(f"""
            <div class="persona-card" style="border-left-color: {p['color']};">
                <div class="persona-title">
                    {p['icon']} {p['persona_name']}
                    <span class="persona-badge" style="background-color: {p['color']}33; color: {p['color']}; border: 1px solid {p['color']};">
                        {p['badge']}
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 12px;">
                    <strong>Audience Size:</strong> {p['customer_count']:,} customers ({p['customer_pct']}%) | 
                    <strong>Total Revenue:</strong> ${p['total_revenue']:,.2f} ({p['revenue_pct']}%) | 
                    <strong>Avg Order Value:</strong> ${p['avg_aov']:.2f} | 
                    <strong>Avg Recency:</strong> {p['avg_recency']} days
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_char, c_strat = st.columns([1, 1.3])
            
            with c_char:
                st.markdown("**Top 3 Defining Characteristics**")
                for c in p["top_characteristics"]:
                    arrow = "⬆️" if c["direction"] == "higher" else "⬇️"
                    st.markdown(
                        f"- **{c['feature']}**: {arrow} **{abs(c['pct_diff'])}%** {c['direction']} than store average "
                        f"*(Segment: {c['cluster_mean']} vs Average: {c['pop_mean']})*"
                    )
            
            with c_strat:
                st.markdown(f"**Marketing Playbook — {p['playbook']['objective']}**")
                st.markdown(f"**Recommended Channels**: {', '.join(p['playbook']['channels'])}")
                sample_hook = p['playbook']['sample_campaign']
                st.markdown(f"**Campaign Hook**: *'{sample_hook}'*")
            
            st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# TAB 6: 15% CONVERSION ROI SIMULATOR
# ==============================================================================
with tabs[5]:
    st.markdown("### Targeted Marketing ROI & Conversion Lift Simulator")
    st.markdown(
        "Directly evaluates the project objective: **'Increase conversion rates through personalized marketing by 15%'**."
    )
    
    sim_col_ctrl, sim_col_out = st.columns([1, 1.4])
    
    with sim_col_ctrl:
        st.markdown("#### Campaign Simulation Parameters")
        
        target_persona_names = ["All Segments (Portfolio-wide)"] + [p["persona_name"] for p in profiles]
        selected_target = st.selectbox("Target Customer Persona", target_persona_names)
        
        base_cr = st.slider("Baseline Conversion Rate (%)", min_value=0.5, max_value=8.0, value=2.5, step=0.1) / 100.0
        lift_pct = st.slider("Target Conversion Lift (%)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
        
        # Audience selection
        if selected_target == "All Segments (Portfolio-wide)":
            audience_size = len(df_clustered)
            target_aov = kpis["avg_order_value"]
            default_budget = round(audience_size * 0.20, 2)
        else:
            chosen_p = next(p for p in profiles if p["persona_name"] == selected_target)
            audience_size = chosen_p["customer_count"]
            target_aov = chosen_p["avg_aov"] or kpis["avg_order_value"]
            default_budget = round(audience_size * 0.25, 2)

        campaign_budget = st.number_input("Campaign Spend Budget ($)", min_value=100.0, max_value=100000.0, value=float(default_budget), step=100.0)
        gross_margin = st.slider("Product Gross Margin (%)", min_value=10.0, max_value=80.0, value=40.0, step=5.0)

        sim_result = simulate_campaign_roi(
            target_segment_name=selected_target,
            target_audience_size=audience_size,
            baseline_aov=target_aov,
            baseline_conversion_rate=base_cr,
            conversion_lift_pct=lift_pct,
            campaign_budget=campaign_budget,
            gross_margin_pct=gross_margin
        )

    with sim_col_out:
        st.markdown(f"#### Projected Financial & Conversion Impact")
        
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Incremental Orders", f"+{sim_result['incremental_conversions']:,}", f"+{lift_pct}% lift")
        with s2:
            st.metric("Incremental Revenue", f"${sim_result['incremental_revenue']:,.2f}", f"${sim_result['incremental_revenue'] - campaign_budget:,.2f} net")
        with s3:
            st.metric("Projected ROAS", f"{sim_result['roas']}x", "Revenue / Spend")
        with s4:
            st.metric("Campaign ROI", f"{sim_result['roi_pct']}%", f"${sim_result['net_profit']:,.2f} profit")

        st.plotly_chart(plot_conversion_lift_waterfall(sim_result), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Full Portfolio Impact Analysis Across All Segments")
    portfolio_df = simulate_portfolio_impact(
        profiles,
        baseline_conversion_rate=base_cr,
        conversion_lift_pct=lift_pct,
        gross_margin_pct=gross_margin
    )
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 7: CUSTOMER EXPORT & SCORING
# ==============================================================================
with tabs[6]:
    st.markdown("### Export Clustered Dataset & Segment Actions")
    st.markdown("Download full customer records enriched with cluster assignments, persona labels, and RFM scores.")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        persona_filter = st.multiselect("Filter by Persona", options=list(df_clustered["Persona"].unique()), default=list(df_clustered["Persona"].unique()))
    with filter_col2:
        if "Monetary_Spend" in df_clustered.columns:
            min_spend, max_spend = float(df_clustered["Monetary_Spend"].min()), float(df_clustered["Monetary_Spend"].max())
            spend_range = st.slider("Monetary Spend Filter ($)", min_value=min_spend, max_value=max_spend, value=(min_spend, max_spend))
        else:
            spend_range = None

    filtered_df = df_clustered[df_clustered["Persona"].isin(persona_filter)]
    if spend_range and "Monetary_Spend" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["Monetary_Spend"] >= spend_range[0]) & (filtered_df["Monetary_Spend"] <= spend_range[1])]

    st.markdown(f"Displaying **{len(filtered_df):,}** matching customers:")
    
    display_cols = [c for c in ["CustomerID", "Persona", "Cluster", "Recency_Days", "Frequency", "Monetary_Spend", "Avg_Order_Value", "RFM_Segment", "RFM_Composite_Score", "Preferred_Category"] if c in filtered_df.columns]
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True, hide_index=True)

    # Download buttons
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="📥 Download Segmented Customers CSV",
            data=csv_data,
            file_name="CustomerIQ_Segmented_Customers.csv",
            mime="text/csv"
        )
    with d2:
        # Export segment playbooks
        playbook_rows = []
        for p in profiles:
            playbook_rows.append({
                "Cluster_ID": p["cluster_id"],
                "Persona": p["persona_name"],
                "Badge": p["badge"],
                "Audience_Size": p["customer_count"],
                "Total_Revenue": p["total_revenue"],
                "Avg_AOV": p["avg_aov"],
                "Objective": p["playbook"]["objective"],
                "Channels": ", ".join(p["playbook"]["channels"]),
                "Strategy": p["playbook"]["strategy"],
                "Sample_Campaign": p["playbook"]["sample_campaign"]
            })
        playbook_csv = pd.DataFrame(playbook_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Marketing Playbook & Personas CSV",
            data=playbook_csv,
            file_name="CustomerIQ_Marketing_Playbook.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "CustomerIQ — AI-Powered Customer Segmentation & Analytics Platform | Built with Python, Scikit-Learn, Plotly & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
