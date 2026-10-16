"""
Model training and artifact serialization script for CustomerIQ.
Trains K-Means model, evaluates cluster metrics, and saves all artifacts.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, validate_dataset, DEFAULT_FEATURE_COLUMNS
from src.preprocessor import CustomerPreprocessor
from src.model import CustomerClusterEngine
from src.persona import generate_cluster_profiles

def main():
    print("=" * 70)
    print(" CustomerIQ — Model Training & Evaluation Pipeline")
    print("=" * 70)

    data_path = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/6] Loading customer dataset from: {data_path}")
    df = load_data(data_path)
    print(f"      Loaded {len(df)} records with {len(df.columns)} columns.")

    # 2. Validation
    print("\n[2/6] Validating dataset schema and data quality...")
    is_valid, issues, audit = validate_dataset(df, DEFAULT_FEATURE_COLUMNS)
    if not is_valid:
        print(f"      [WARNING] Validation issues found: {issues}")
    else:
        print("      [PASSED] Dataset is clean with zero missing values.")

    # 3. Preprocessing
    print("\n[3/6] Preprocessing features (Outlier Handling, RFM Scoring, Scaling, PCA)...")
    preprocessor = CustomerPreprocessor(
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        scaling_method="standard",
        outlier_method="clip",
        pca_components=3
    )
    X_scaled, df_processed, prep_metadata = preprocessor.fit_transform(df)
    print(f"      Fitted {len(prep_metadata['features_used'])} features.")
    print(f"      PCA 2D explained variance: {prep_metadata['pca_2d_variance']}%")
    print(f"      PCA 3D explained variance: {prep_metadata['pca_3d_variance']}%")

    # 4. Cluster Range Evaluation
    print("\n[4/6] Evaluating cluster count k in range [2, 9] (Elbow & Silhouette)...")
    eval_results = CustomerClusterEngine.evaluate_cluster_range(X_scaled, k_min=2, k_max=9)
    print(f"      Optimal K (Silhouette): {eval_results['optimal_k_silhouette']} (Score: {max(eval_results['silhouettes'])})")
    print(f"      Optimal K (Elbow Knee): {eval_results['optimal_k_elbow']}")
    print(f"      Recommended K: {eval_results['recommended_k']}")

    # 5. Fit Final Model
    best_k = 5  # Optimal 5-segment behavioral taxonomy
    print(f"\n[5/6] Fitting K-Means with k={best_k}...")
    engine = CustomerClusterEngine(n_clusters=best_k)
    engine.fit(X_scaled)
    df_processed["Cluster"] = engine.cluster_labels_

    # Generate Personas and Top Characteristics
    profiles = generate_cluster_profiles(df_processed, "Cluster", DEFAULT_FEATURE_COLUMNS)
    cluster_to_persona = {p["cluster_id"]: p["persona_name"] for p in profiles}
    df_processed["Persona"] = df_processed["Cluster"].map(cluster_to_persona)

    print("\n      Cluster Profiles Identified:")
    for p in profiles:
        print(f"      - Cluster {p['cluster_id']}: {p['persona_name']} | Count: {p['customer_count']} ({p['customer_pct']}%) | Rev: ${p['total_revenue']:,.2f}")
        for c in p["top_characteristics"]:
            print(f"          * {c['feature']}: {c['direction']} by {abs(c['pct_diff'])}% vs average")

    # 6. Save Artifacts
    print("\n[6/6] Saving trained models and serializing artifacts...")
    model_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    meta_path = os.path.join(models_dir, "training_metadata.json")
    output_csv = os.path.join(PROJECT_ROOT, "data", "customer_segments_output.csv")

    joblib.dump(engine.model, model_path)
    joblib.dump(preprocessor.scaler, scaler_path)
    joblib.dump(preprocessor.pca_3d, pca_path)

    metadata = {
        "features": DEFAULT_FEATURE_COLUMNS,
        "n_clusters": best_k,
        "metrics": engine.metrics_,
        "evaluation_range": eval_results,
        "preprocessing": {
            "scaling_method": "standard",
            "outlier_method": "clip",
            "pca_2d_variance": prep_metadata["pca_2d_variance"],
            "pca_3d_variance": prep_metadata["pca_3d_variance"]
        },
        "profiles": profiles
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    df_processed.to_csv(output_csv, index=False)

    print(f"      [OK] Saved model to: {model_path}")
    print(f"      [OK] Saved scaler to: {scaler_path}")
    print(f"      [OK] Saved PCA to: {pca_path}")
    print(f"      [OK] Saved metadata to: {meta_path}")
    print(f"      [OK] Saved clustered dataset to: {output_csv}")
    print("\n" + "=" * 70)
    print(" Pipeline Execution Successfully Completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
