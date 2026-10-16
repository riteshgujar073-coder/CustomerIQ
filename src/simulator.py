"""
Marketing Campaign ROI and 15% Conversion Lift Simulator for CustomerIQ.
Directly models the business objective of achieving +15% conversion lift via personalization.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def simulate_campaign_roi(
    target_segment_name: str,
    target_audience_size: int,
    baseline_aov: float,
    baseline_conversion_rate: float = 0.025,   # e.g., 2.5% standard e-commerce conversion
    conversion_lift_pct: float = 15.0,         # Target +15% conversion lift
    campaign_budget: float = 5000.0,
    gross_margin_pct: float = 40.0             # 40% gross product margin
) -> Dict[str, Any]:
    """
    Simulate financial and conversion outcomes from implementing personalized segmentation.
    """
    baseline_rate = max(0.001, baseline_conversion_rate)
    lift_multiplier = 1.0 + (conversion_lift_pct / 100.0)
    personalized_rate = baseline_rate * lift_multiplier
    
    # Conversions
    baseline_conversions = int(round(target_audience_size * baseline_rate))
    personalized_conversions = int(round(target_audience_size * personalized_rate))
    incremental_conversions = max(0, personalized_conversions - baseline_conversions)

    # Revenue
    baseline_revenue = round(baseline_conversions * baseline_aov, 2)
    personalized_revenue = round(personalized_conversions * baseline_aov, 2)
    incremental_revenue = round(personalized_revenue - baseline_revenue, 2)

    # Profit and ROI
    gross_margin = gross_margin_pct / 100.0
    incremental_gross_profit = round(incremental_revenue * gross_margin, 2)
    net_profit = round(incremental_gross_profit - campaign_budget, 2)
    
    roas = round(incremental_revenue / max(1.0, campaign_budget), 2)
    roi_pct = round((net_profit / max(1.0, campaign_budget)) * 100, 1)

    # Customer Acquisition Cost (CAC) for incremental conversions
    cac_incremental = round(campaign_budget / max(1, incremental_conversions), 2)

    return {
        "target_segment": target_segment_name,
        "audience_size": target_audience_size,
        "baseline_rate_pct": round(baseline_rate * 100, 2),
        "personalized_rate_pct": round(personalized_rate * 100, 2),
        "conversion_lift_pct": conversion_lift_pct,
        "baseline_conversions": baseline_conversions,
        "personalized_conversions": personalized_conversions,
        "incremental_conversions": incremental_conversions,
        "baseline_revenue": baseline_revenue,
        "personalized_revenue": personalized_revenue,
        "incremental_revenue": incremental_revenue,
        "campaign_budget": campaign_budget,
        "gross_margin_pct": gross_margin_pct,
        "incremental_gross_profit": incremental_gross_profit,
        "net_profit": net_profit,
        "roas": roas,
        "roi_pct": roi_pct,
        "cac_incremental": cac_incremental
    }

def simulate_portfolio_impact(
    profiles: List[Dict[str, Any]],
    baseline_conversion_rate: float = 0.025,
    conversion_lift_pct: float = 15.0,
    cost_per_contact: float = 0.15,
    gross_margin_pct: float = 40.0
) -> pd.DataFrame:
    """
    Simulate the aggregate business impact across all identified customer segments.
    """
    rows = []
    for p in profiles:
        size = p["customer_count"]
        aov = p.get("avg_aov", 150.0) or 150.0
        budget = round(size * cost_per_contact, 2)
        
        sim = simulate_campaign_roi(
            target_segment_name=p["persona_name"],
            target_audience_size=size,
            baseline_aov=aov,
            baseline_conversion_rate=baseline_conversion_rate,
            conversion_lift_pct=conversion_lift_pct,
            campaign_budget=budget,
            gross_margin_pct=gross_margin_pct
        )
        
        rows.append({
            "Segment": sim["target_segment"],
            "Audience": sim["audience_size"],
            "Avg AOV ($)": aov,
            "Baseline Orders": sim["baseline_conversions"],
            "Target (+15%) Orders": sim["personalized_conversions"],
            "Incremental Orders": sim["incremental_conversions"],
            "Incremental Revenue ($)": sim["incremental_revenue"],
            "Campaign Spend ($)": sim["campaign_budget"],
            "Net Profit ($)": sim["net_profit"],
            "ROAS": f"{sim['roas']}x",
            "ROI (%)": f"{sim['roi_pct']}%"
        })

    return pd.DataFrame(rows)
