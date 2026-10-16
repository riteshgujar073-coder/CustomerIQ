"""
Data loading, validation, and auditing utilities for CustomerIQ.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import io

# Canonical feature names expected by the segmentation engine
DEFAULT_FEATURE_COLUMNS = [
    "Recency_Days",
    "Frequency",
    "Monetary_Spend",
    "Avg_Order_Value",
    "Total_Items_Purchased",
    "Website_Visits",
    "Avg_Session_Duration_Min",
    "Discount_Usage_Ratio",
    "Return_Rate"
]

CORE_RFM_COLUMNS = ["Recency_Days", "Frequency", "Monetary_Spend"]

def load_data(source: Any) -> pd.DataFrame:
    """
    Load data from a file path or file-like buffer (e.g., Streamlit UploadedFile).
    """
    if isinstance(source, (str, io.BytesIO, io.StringIO)) or hasattr(source, "read"):
        df = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise ValueError("Unsupported data source format. Expected file path, buffer, or DataFrame.")
    
    return df

def validate_dataset(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Audit the dataset for missing values, shape, and numeric requirements.
    Returns (is_valid, list_of_issues, audit_summary).
    """
    issues = []
    audit = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=[np.number]).columns),
    }

    if len(df) < 20:
        issues.append(f"Dataset has only {len(df)} rows. At least 20 rows are required for clustering.")

    cols_to_check = feature_cols if feature_cols else [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    
    if not cols_to_check:
        # Fall back to any numeric columns
        cols_to_check = audit["numeric_columns"]
        if not cols_to_check:
            issues.append("No numeric columns found in dataset for clustering.")

    # Check for missing values in clustering features
    for col in cols_to_check:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} ({null_count/len(df)*100:.1f}%) missing values.")
            if not np.issubdtype(df[col].dtype, np.number):
                issues.append(f"Column '{col}' is not numeric ({df[col].dtype}).")
        else:
            issues.append(f"Required clustering column '{col}' is missing.")

    is_valid = len(issues) == 0
    return is_valid, issues, audit

def get_dataset_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate top-level executive KPI metrics from customer data.
    """
    kpis = {}
    kpis["total_customers"] = len(df)
    
    # Monetary / Revenue
    mon_col = "Monetary_Spend" if "Monetary_Spend" in df.columns else None
    if mon_col:
        kpis["total_revenue"] = float(df[mon_col].sum())
        kpis["avg_monetary"] = float(df[mon_col].mean())
        kpis["median_monetary"] = float(df[mon_col].median())
    else:
        kpis["total_revenue"] = 0.0
        kpis["avg_monetary"] = 0.0
        kpis["median_monetary"] = 0.0

    # Frequency
    freq_col = "Frequency" if "Frequency" in df.columns else None
    if freq_col:
        kpis["avg_frequency"] = float(df[freq_col].mean())
    else:
        kpis["avg_frequency"] = 0.0

    # Recency
    rec_col = "Recency_Days" if "Recency_Days" in df.columns else None
    if rec_col:
        kpis["avg_recency"] = float(df[rec_col].mean())
    else:
        kpis["avg_recency"] = 0.0

    # Avg Order Value
    aov_col = "Avg_Order_Value" if "Avg_Order_Value" in df.columns else None
    if aov_col:
        kpis["avg_order_value"] = float(df[aov_col].mean())
    elif mon_col and freq_col and (df[freq_col] > 0).all():
        kpis["avg_order_value"] = float((df[mon_col] / df[freq_col]).mean())
    else:
        kpis["avg_order_value"] = 0.0

    # Satisfaction Score
    sat_col = "Satisfaction_Score" if "Satisfaction_Score" in df.columns else None
    if sat_col:
        kpis["avg_satisfaction"] = float(df[sat_col].mean())
    else:
        kpis["avg_satisfaction"] = None

    # Return Rate
    ret_col = "Return_Rate" if "Return_Rate" in df.columns else None
    if ret_col:
        kpis["avg_return_rate"] = float(df[ret_col].mean())
    else:
        kpis["avg_return_rate"] = None

    return kpis

def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently auto-map user uploaded columns to canonical segmentation features.
    """
    mapping = {
        "customer_id": None,
        "recency": None,
        "frequency": None,
        "monetary": None,
        "avg_order_value": None,
        "visits": None,
        "session_duration": None,
        "discount_ratio": None,
        "return_rate": None,
        "tenure": None,
        "satisfaction": None
    }
    
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    
    for key, c_original in cols.items():
        if any(id_kw in key for id_kw in ["customerid", "clientid", "userid", "custid", "id"]):
            if mapping["customer_id"] is None:
                mapping["customer_id"] = c_original
        elif any(r_kw in key for r_kw in ["recency", "lastpurchase", "dayssincelast"]):
            mapping["recency"] = c_original
        elif any(f_kw in key for f_kw in ["frequency", "orders", "transactions", "ordercount", "purchases"]):
            mapping["frequency"] = c_original
        elif any(m_kw in key for m_kw in ["monetary", "revenue", "spend", "sales", "totalspend", "totalamount"]):
            mapping["monetary"] = c_original
        elif any(aov_kw in key for aov_kw in ["avgordervalue", "aov", "averageorder"]):
            mapping["avg_order_value"] = c_original
        elif any(v_kw in key for v_kw in ["visits", "views", "sessions", "websitevisits"]):
            mapping["visits"] = c_original
        elif any(d_kw in key for d_kw in ["duration", "browsingtime", "sessiontime"]):
            mapping["session_duration"] = c_original
        elif any(disc_kw in key for disc_kw in ["discount", "coupon", "promousage"]):
            mapping["discount_ratio"] = c_original
        elif any(ret_kw in key for ret_kw in ["return", "refund"]):
            mapping["return_rate"] = c_original
        elif any(t_kw in key for t_kw in ["tenure", "loyaltymonths", "membershipmonths"]):
            mapping["tenure"] = c_original
        elif any(s_kw in key for s_kw in ["satisfaction", "rating", "csat", "nps"]):
            mapping["satisfaction"] = c_original

    return mapping
