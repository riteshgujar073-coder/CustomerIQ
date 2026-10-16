"""
Unit and integration tests for CustomerIQ ML pipeline.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, validate_dataset, get_dataset_kpis, auto_detect_columns, DEFAULT_FEATURE_COLUMNS
from src.preprocessor import CustomerPreprocessor
from src.model import CustomerClusterEngine
from src.persona import extract_top_characteristics, generate_cluster_profiles, get_marketing_playbook
from src.simulator import simulate_campaign_roi, simulate_portfolio_impact

class TestCustomerIQPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
        cls.df = load_data(cls.data_path)

    def test_01_data_loading_and_kpis(self):
        """Test dataset loading and KPI calculations."""
        self.assertGreater(len(self.df), 100)
        self.assertIn("CustomerID", self.df.columns)
        self.assertIn("Monetary_Spend", self.df.columns)
        
        kpis = get_dataset_kpis(self.df)
        self.assertEqual(kpis["total_customers"], len(self.df))
        self.assertGreater(kpis["total_revenue"], 0)
        self.assertGreater(kpis["avg_order_value"], 0)

    def test_02_validation(self):
        """Test dataset validation and schema checks."""
        is_valid, issues, audit = validate_dataset(self.df, DEFAULT_FEATURE_COLUMNS)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        self.assertEqual(audit["total_nulls"], 0)

    def test_03_column_auto_mapping(self):
        """Test intelligent column auto-detection."""
        synthetic_df = pd.DataFrame({
            "User_ID": [1, 2],
            "Days_Since_Last": [10, 20],
            "Order_Count": [5, 10],
            "Total_Spend": [100.0, 250.0]
        })
        mapping = auto_detect_columns(synthetic_df)
        self.assertEqual(mapping["customer_id"], "User_ID")
        self.assertEqual(mapping["recency"], "Days_Since_Last")
        self.assertEqual(mapping["frequency"], "Order_Count")
        self.assertEqual(mapping["monetary"], "Total_Spend")

    def test_04_preprocessor(self):
        """Test outlier clipping, RFM score computation, scaling, and PCA."""
        prep = CustomerPreprocessor(
            feature_cols=DEFAULT_FEATURE_COLUMNS,
            scaling_method="standard",
            outlier_method="clip",
            pca_components=3
        )
        X_scaled, df_proc, meta = prep.fit_transform(self.df)
        
        self.assertEqual(X_scaled.shape[0], len(self.df))
        self.assertEqual(X_scaled.shape[1], len(DEFAULT_FEATURE_COLUMNS))
        self.assertIn("PCA1", df_proc.columns)
        self.assertIn("PCA2", df_proc.columns)
        self.assertIn("PCA3", df_proc.columns)
        self.assertIn("RFM_Composite_Score", df_proc.columns)
        self.assertGreater(meta["pca_2d_variance"], 50.0)

    def test_05_cluster_range_evaluation(self):
        """Test Elbow & Silhouette evaluation over range."""
        prep = CustomerPreprocessor(feature_cols=DEFAULT_FEATURE_COLUMNS)
        X_scaled, _, _ = prep.fit_transform(self.df.head(300))
        eval_res = CustomerClusterEngine.evaluate_cluster_range(X_scaled, k_min=2, k_max=6)
        
        self.assertEqual(len(eval_res["k_values"]), 5)
        self.assertEqual(len(eval_res["silhouettes"]), 5)
        self.assertIn(eval_res["recommended_k"], [2, 3, 4, 5, 6])

    def test_06_model_fit_and_persona_generation(self):
        """Test K-Means training, persona identification, and top characteristics."""
        prep = CustomerPreprocessor(feature_cols=DEFAULT_FEATURE_COLUMNS)
        X_scaled, df_proc, _ = prep.fit_transform(self.df)
        
        engine = CustomerClusterEngine(n_clusters=5)
        engine.fit(X_scaled)
        
        self.assertGreater(engine.metrics_["silhouette_score"], 0.30)
        self.assertGreater(engine.metrics_["calinski_harabasz_score"], 100)
        
        df_proc["Cluster"] = engine.cluster_labels_
        profiles = generate_cluster_profiles(df_proc, "Cluster", DEFAULT_FEATURE_COLUMNS)
        self.assertEqual(len(profiles), 5)
        
        for p in profiles:
            self.assertIn("persona_name", p)
            self.assertEqual(len(p["top_characteristics"]), 3)
            self.assertIn("playbook", p)
            self.assertIn("objective", p["playbook"])

    def test_07_simulator(self):
        """Test 15% conversion lift simulation and portfolio ROI calculation."""
        sim = simulate_campaign_roi(
            target_segment_name="VIP Champions",
            target_audience_size=500,
            baseline_aov=250.0,
            baseline_conversion_rate=0.03,
            conversion_lift_pct=15.0,
            campaign_budget=1000.0,
            gross_margin_pct=40.0
        )
        self.assertGreater(sim["personalized_conversions"], sim["baseline_conversions"])
        self.assertGreater(sim["incremental_revenue"], 0)
        self.assertGreater(sim["roas"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
