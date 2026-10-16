"""
Customer Persona Profiling, Top 3 Defining Characteristics Extraction,
and Actionable Marketing Strategy Recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Persona taxonomy palette
PERSONA_PALETTE = {
    "VIP Champions": {"color": "#6366F1", "icon": "💎", "badge": "High-Value Hero"},
    "Loyal Steady Spenders": {"color": "#10B981", "icon": "🌟", "badge": "Core Retention"},
    "Bargain & Deal Seekers": {"color": "#F59E0B", "icon": "🏷️", "badge": "Price Sensitive"},
    "At-Risk High-Value Lapsers": {"color": "#EF4444", "icon": "⚠️", "badge": "Urgent Win-Back"},
    "New Promising Explorers": {"color": "#06B6D4", "icon": "🚀", "badge": "Nurture & Activate"},
    "Standard Segment": {"color": "#8B5CF6", "icon": "👤", "badge": "General Segment"}
}

def identify_cluster_persona(cluster_mean: pd.Series, pop_mean: pd.Series, cluster_id: int) -> str:
    """
    Heuristic persona classification based on RFM and behavioral metrics.
    """
    rec_diff = cluster_mean.get("Recency_Days", 0) - pop_mean.get("Recency_Days", 0)
    freq_diff = cluster_mean.get("Frequency", 0) - pop_mean.get("Frequency", 0)
    mon_diff = cluster_mean.get("Monetary_Spend", 0) - pop_mean.get("Monetary_Spend", 0)
    disc_diff = cluster_mean.get("Discount_Usage_Ratio", 0) - pop_mean.get("Discount_Usage_Ratio", 0)
    tenure_diff = cluster_mean.get("Tenure_Months", 0) - pop_mean.get("Tenure_Months", 0)

    # 1. VIP Champions: High Spend, High Frequency, Fresh Recency
    if mon_diff > 0.4 * pop_mean.get("Monetary_Spend", 1) and freq_diff > 0:
        return "VIP Champions"
    # 2. At-Risk: High Recency (long absence) and previously moderate/high spend
    elif rec_diff > 0.5 * pop_mean.get("Recency_Days", 1):
        return "At-Risk High-Value Lapsers"
    # 3. Deal Seekers: High discount usage, lower-to-moderate spend
    elif disc_diff > 0.15:
        return "Bargain & Deal Seekers"
    # 4. New Explorers: Low frequency, low monetary, fresh or below average recency
    elif freq_diff < -0.4 * pop_mean.get("Frequency", 1) and mon_diff < -0.4 * pop_mean.get("Monetary_Spend", 1):
        return "New Promising Explorers"
    # 5. Loyal Steady
    elif freq_diff >= 0 and mon_diff >= 0:
        return "Loyal Steady Spenders"
    else:
        return f"Segment {cluster_id + 1}"

def extract_top_characteristics(
    cluster_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    feature_cols: List[str],
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Identify the top N defining characteristics that differentiate this segment
    from the overall customer population using normalized z-score deviations.
    """
    characteristics = []
    
    for col in feature_cols:
        if col not in cluster_df.columns or not np.issubdtype(cluster_df[col].dtype, np.number):
            continue
        
        c_mean = float(cluster_df[col].mean())
        p_mean = float(pop_df[col].mean())
        p_std = float(pop_df[col].std()) + 1e-9
        
        z_score = (c_mean - p_mean) / p_std
        pct_diff = ((c_mean - p_mean) / (p_mean + 1e-9)) * 100
        
        characteristics.append({
            "feature": col,
            "cluster_mean": round(c_mean, 2),
            "pop_mean": round(p_mean, 2),
            "pct_diff": round(pct_diff, 1),
            "z_score": round(z_score, 3),
            "abs_z": abs(z_score),
            "direction": "higher" if z_score > 0 else "lower"
        })

    # Sort by absolute deviation (most distinctive features)
    characteristics.sort(key=lambda x: x["abs_z"], reverse=True)
    return characteristics[:top_n]

def get_marketing_playbook(persona_name: str) -> Dict[str, Any]:
    """
    Actionable e-commerce personalization and marketing playbook for each persona.
    """
    playbooks = {
        "VIP Champions": {
            "objective": "Maximize Lifetime Value (LTV) & Brand Advocacy",
            "channels": ["Dedicated VIP Concierge", "Private App Early Access", "Exclusive Direct Mail"],
            "strategy": "Reward loyalty with bespoke privileges without diluting brand prestige with discounts.",
            "tactics": [
                "Invite to exclusive pre-launch product drops 48 hours prior to public.",
                "Offer zero-threshold express shipping and dedicated 24/7 priority support.",
                "Implement a tiered referral incentive (e.g. Give $50, Get $50)."
            ],
            "sample_campaign": "✨ Early Access: Exclusive preview of the Luxury Summer Collection just for you.",
            "conversion_opportunity": "High AOV upsells, premium warranty bundles, annual membership upgrades."
        },
        "Loyal Steady Spenders": {
            "objective": "Frequency Acceleration & Cross-Category Expansion",
            "channels": ["Personalized Email Drips", "Personalized Web Recommendations", "App Push Notifications"],
            "strategy": "Introduce complementary product categories based on previous purchase sequences.",
            "tactics": [
                "Deploy dynamic bundle recommendations ('Frequently Bought Together').",
                "Offer tiered milestone rewards: 'Spend $150 more this month for Silver tier'.",
                "Re-engage at average repurchase interval (every 30-45 days)."
            ],
            "sample_campaign": "⭐ We noticed you love Fashion — Check out these matching Home arrivals curated for you.",
            "conversion_opportunity": "15-20% basket size increase via algorithmically paired accessories."
        },
        "Bargain & Deal Seekers": {
            "objective": "Margin Preservation & Timed Promotion Conversion",
            "channels": ["Flash Sale SMS", "Telegram/WhatsApp Deal Alerts", "Exit-Intent Web Overlays"],
            "strategy": "Channel price sensitivity into high-margin clearance stock and time-limited deals.",
            "tactics": [
                "Deploy countdown timers on clearance/outlet collections.",
                "Condition discounts on minimum spend thresholds (e.g., 'Save 20% on orders above $120').",
                "Recommend budget bundles to elevate unit count per checkout."
            ],
            "sample_campaign": "🏷️ 24-Hour Flash Drop: Up to 40% off selected bestsellers while supplies last!",
            "conversion_opportunity": "Clear aging inventory with bounded discounts while boosting overall basket items."
        },
        "At-Risk High-Value Lapsers": {
            "objective": "Urgent Churn Prevention & Win-Back",
            "channels": ["Automated Win-Back Email Flow", "Retargeting Ads (Meta & Google)", "Feedback Survey SMS"],
            "strategy": "Diagnose disengagement, resolve friction, and provide a compelling re-activation incentive.",
            "tactics": [
                "Trigger automated Day-60/Day-90 inactivity re-engagement email sequence.",
                "Offer a high-value 'We Miss You' voucher with expiry within 7 days.",
                "Prompt customer satisfaction survey with an immediate perk for completion."
            ],
            "sample_campaign": "👋 It's been a while, Alex! Here is $25 off your next order to welcome you back.",
            "conversion_opportunity": "Re-capturing 10-15% of at-risk accounts recovers high-margin recurring spend."
        },
        "New Promising Explorers": {
            "objective": "First-to-Second Purchase Conversion (Activation)",
            "channels": ["Welcome Email Series", "In-App Onboarding Carousel", "Social Proof Showcase"],
            "strategy": "Nurture trust, educate on platform perks, and secure the vital second purchase within 30 days.",
            "tactics": [
                "Send 3-part educational welcome flow (Brand story -> Top rated picks -> Social proof/reviews).",
                "Provide a 10% coupon valid for 14 days on their second order.",
                "Personalize homepage landing based on their initial category view."
            ],
            "sample_campaign": "🎉 Welcome to the Club! Enjoy 10% off your next favorite find with code WELCOME10.",
            "conversion_opportunity": "Turning single-order buyers into repeat customers yields a 3x lifetime value jump."
        }
    }

    # Default fallback playbook for generic clusters
    default_playbook = {
        "objective": "Engagement & Behavioral Personalization",
        "channels": ["Targeted Email", "On-site Dynamic Banners"],
        "strategy": "Optimize touchpoint timing and tailor category recommendations to behavioral signals.",
        "tactics": [
            "A/B test subject lines and promotional hooks.",
            "Align promotional frequency to individual browsing velocity.",
            "Gather user preferences via interactive survey widget."
        ],
        "sample_campaign": "✨ Discover something tailored just for you today.",
        "conversion_opportunity": "Systematic 10-15% lift through algorithmic personalization."
    }

    return playbooks.get(persona_name, default_playbook)

def generate_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive profiles for all clusters including persona tags,
    top 3 defining characteristics, and marketing playbooks.
    """
    pop_mean = df[feature_cols].mean()
    total_customers = len(df)
    total_revenue = df["Monetary_Spend"].sum() if "Monetary_Spend" in df.columns else 1.0

    profiles = []
    unique_clusters = sorted(df[cluster_col].unique())

    # Map cluster ids to unique personas cleanly
    used_personas = set()
    for cid in unique_clusters:
        cdf = df[df[cluster_col] == cid]
        cmean = cdf[feature_cols].mean()
        
        # Determine persona name
        candidate_persona = identify_cluster_persona(cmean, pop_mean, cid)
        if candidate_persona in used_personas and not candidate_persona.startswith("Segment"):
            persona_name = f"{candidate_persona} (Tier {cid + 1})"
        else:
            persona_name = candidate_persona
        used_personas.add(persona_name)

        # Style metadata
        base_name = persona_name.split(" (Tier")[0]
        meta = PERSONA_PALETTE.get(base_name, PERSONA_PALETTE["Standard Segment"])

        # Top 3 distinctive characteristics
        top_chars = extract_top_characteristics(cdf, df, feature_cols, top_n=3)

        # Revenue and size stats
        c_size = len(cdf)
        c_pct = (c_size / total_customers) * 100
        c_rev = cdf["Monetary_Spend"].sum() if "Monetary_Spend" in cdf.columns else 0.0
        c_rev_pct = (c_rev / total_revenue) * 100 if total_revenue > 0 else 0.0

        playbook = get_marketing_playbook(base_name)

        profiles.append({
            "cluster_id": int(cid),
            "persona_name": persona_name,
            "badge": meta["badge"],
            "icon": meta["icon"],
            "color": meta["color"],
            "customer_count": c_size,
            "customer_pct": round(c_pct, 1),
            "total_revenue": round(c_rev, 2),
            "revenue_pct": round(c_rev_pct, 1),
            "avg_recency": round(float(cdf["Recency_Days"].mean()), 1) if "Recency_Days" in cdf.columns else None,
            "avg_frequency": round(float(cdf["Frequency"].mean()), 1) if "Frequency" in cdf.columns else None,
            "avg_monetary": round(float(cdf["Monetary_Spend"].mean()), 2) if "Monetary_Spend" in cdf.columns else None,
            "avg_aov": round(float(cdf["Avg_Order_Value"].mean()), 2) if "Avg_Order_Value" in cdf.columns else None,
            "avg_discount": round(float(cdf["Discount_Usage_Ratio"].mean() * 100), 1) if "Discount_Usage_Ratio" in cdf.columns else None,
            "avg_visits": round(float(cdf["Website_Visits"].mean()), 1) if "Website_Visits" in cdf.columns else None,
            "top_characteristics": top_chars,
            "playbook": playbook
        })

    return profiles
