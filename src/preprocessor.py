"""
Data preprocessing, outlier handling, scaling, RFM scoring, and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA

class CustomerPreprocessor:
    """
    Comprehensive preprocessing pipeline for customer segmentation.
    Handles outliers, scaling, RFM scoring, and PCA projections.
    """
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        scaling_method: str = "standard",  # "standard", "robust", "minmax"
        outlier_method: str = "clip",      # "clip", "remove", "none"
        outlier_iqr_multiplier: float = 1.5,
        pca_components: int = 3
    ):
        self.feature_cols = feature_cols
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.pca_components = pca_components
        
        self.scaler = None
        self.pca_2d = None
        self.pca_3d = None
        self.outlier_stats: Dict[str, int] = {}
        self.fitted_columns: List[str] = []

    def _init_scaler(self):
        if self.scaling_method == "robust":
            return RobustScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def detect_and_handle_outliers(self, df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect outliers using IQR and either clip, remove, or leave them.
        """
        df_out = df.copy()
        outlier_summary = {}
        total_outlier_rows = set()

        for col in cols:
            if not np.issubdtype(df_out[col].dtype, np.number):
                continue
            
            q25 = df_out[col].quantile(0.25)
            q75 = df_out[col].quantile(0.75)
            iqr = q75 - q25
            
            lower_bound = q25 - (self.outlier_iqr_multiplier * iqr)
            upper_bound = q75 + (self.outlier_iqr_multiplier * iqr)
            
            is_outlier = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            outlier_indices = df_out[is_outlier].index
            outlier_summary[col] = {
                "count": int(is_outlier.sum()),
                "pct": float(round((is_outlier.sum() / len(df_out)) * 100, 2)),
                "lower_bound": float(round(lower_bound, 2)),
                "upper_bound": float(round(upper_bound, 2))
            }
            total_outlier_rows.update(outlier_indices)

            if self.outlier_method == "clip":
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)

        if self.outlier_method == "remove":
            df_out = df_out.drop(index=list(total_outlier_rows)).reset_index(drop=True)

        meta = {
            "method": self.outlier_method,
            "total_outlier_rows": len(total_outlier_rows),
            "total_outlier_pct": round((len(total_outlier_rows) / len(df)) * 100, 2),
            "per_feature": outlier_summary
        }
        self.outlier_stats = meta
        return df_out, meta

    def compute_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standard RFM quantile scores (1 to 5).
        For Recency: lower is better (5 = most recent).
        For Frequency & Monetary: higher is better (5 = highest).
        """
        df_rfm = df.copy()
        
        has_r = "Recency_Days" in df_rfm.columns
        has_f = "Frequency" in df_rfm.columns
        has_m = "Monetary_Spend" in df_rfm.columns
        
        if has_r and has_f and has_m:
            try:
                # Recency score (inverted: 1 = oldest, 5 = freshest)
                df_rfm["R_Score"] = pd.qcut(df_rfm["Recency_Days"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
                # Frequency score
                df_rfm["F_Score"] = pd.qcut(df_rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
                # Monetary score
                df_rfm["M_Score"] = pd.qcut(df_rfm["Monetary_Spend"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
                
                df_rfm["RFM_Segment"] = (
                    df_rfm["R_Score"].astype(str) + 
                    df_rfm["F_Score"].astype(str) + 
                    df_rfm["M_Score"].astype(str)
                )
                df_rfm["RFM_Composite_Score"] = (df_rfm["R_Score"] + df_rfm["F_Score"] + df_rfm["M_Score"]) / 3.0
            except Exception:
                # Fallback ranking if qcut fails due to unique edge cases
                df_rfm["R_Score"] = pd.cut(df_rfm["Recency_Days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
                df_rfm["F_Score"] = pd.cut(df_rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
                df_rfm["M_Score"] = pd.cut(df_rfm["Monetary_Spend"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
                df_rfm["RFM_Segment"] = df_rfm["R_Score"].astype(str) + df_rfm["F_Score"].astype(str) + df_rfm["M_Score"].astype(str)
                df_rfm["RFM_Composite_Score"] = (df_rfm["R_Score"] + df_rfm["F_Score"] + df_rfm["M_Score"]) / 3.0

        return df_rfm

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """
        Fit scaler and PCA, transform features, return scaled array, processed df, and metadata.
        """
        if feature_cols:
            self.feature_cols = feature_cols
        elif not self.feature_cols:
            # Default to all numeric columns
            self.feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]

        # Filter to columns that exist in df
        self.fitted_columns = [c for c in self.feature_cols if c in df.columns]
        
        # Outlier handling
        df_clean, outlier_meta = self.detect_and_handle_outliers(df, self.fitted_columns)
        
        # Add RFM scores
        df_clean = self.compute_rfm_scores(df_clean)

        # Scale features
        X_raw = df_clean[self.fitted_columns].values
        self.scaler = self._init_scaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # Fit PCA
        n_comp = min(len(self.fitted_columns), self.pca_components, len(df_clean))
        self.pca_2d = PCA(n_components=min(2, n_comp), random_state=42)
        X_pca_2d = self.pca_2d.fit_transform(X_scaled)
        
        df_clean["PCA1"] = X_pca_2d[:, 0]
        if X_pca_2d.shape[1] > 1:
            df_clean["PCA2"] = X_pca_2d[:, 1]
        else:
            df_clean["PCA2"] = 0.0

        pca_3d_variance = 0.0
        if n_comp >= 3:
            self.pca_3d = PCA(n_components=3, random_state=42)
            X_pca_3d = self.pca_3d.fit_transform(X_scaled)
            df_clean["PCA3"] = X_pca_3d[:, 2]
            pca_3d_variance = float(np.sum(self.pca_3d.explained_variance_ratio_))

        pca_2d_variance = float(np.sum(self.pca_2d.explained_variance_ratio_))

        # Calculate PCA Loadings (Component weights per feature)
        loadings_df = pd.DataFrame(
            self.pca_2d.components_.T,
            columns=[f"PCA{i+1}" for i in range(self.pca_2d.n_components_)],
            index=self.fitted_columns
        )

        metadata = {
            "features_used": self.fitted_columns,
            "scaling_method": self.scaling_method,
            "outlier_summary": outlier_meta,
            "pca_2d_variance": round(pca_2d_variance * 100, 2),
            "pca_3d_variance": round(pca_3d_variance * 100, 2),
            "pca_loadings": loadings_df.to_dict()
        }

        return X_scaled, df_clean, metadata

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transform new data using fitted scaler and PCA models.
        """
        if self.scaler is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
        
        df_clean = df.copy()
        # Handle outliers by clipping with saved bounds if possible
        X_raw = df_clean[self.fitted_columns].values
        X_scaled = self.scaler.transform(X_raw)
        
        X_pca_2d = self.pca_2d.transform(X_scaled)
        df_clean["PCA1"] = X_pca_2d[:, 0]
        df_clean["PCA2"] = X_pca_2d[:, 1]
        
        if self.pca_3d is not None:
            X_pca_3d = self.pca_3d.transform(X_scaled)
            df_clean["PCA3"] = X_pca_3d[:, 2]

        return X_scaled, df_clean
