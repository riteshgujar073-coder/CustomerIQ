"""
Machine Learning clustering engine using K-Means, Elbow method, Silhouette score,
Davies-Bouldin index, and Calinski-Harabasz evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os

class CustomerClusterEngine:
    """
    K-Means clustering and evaluation engine.
    """
    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        
        self.model: Optional[KMeans] = None
        self.metrics_: Dict[str, float] = {}
        self.centroids_scaled_: Optional[np.ndarray] = None
        self.cluster_labels_: Optional[np.ndarray] = None

    @staticmethod
    def evaluate_cluster_range(
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 9,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluate WCSS (Elbow), Silhouette, Davies-Bouldin, and Calinski-Harabasz
        for a range of cluster values k.
        """
        k_values = list(range(k_min, k_max + 1))
        inertias = []
        silhouettes = []
        davies_bouldin_scores = []
        calinski_scores = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
            labels = km.fit_predict(X)
            
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X, labels))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            
            silhouettes.append(round(sil, 4))
            davies_bouldin_scores.append(round(db, 4))
            calinski_scores.append(round(ch, 2))

        # Determine optimal k based on max silhouette score
        best_sil_idx = int(np.argmax(silhouettes))
        optimal_k_sil = k_values[best_sil_idx]

        # Determine elbow point using 2D point-to-line perpendicular distance
        # Normalize inertia to [0, 1]
        norm_inertias = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias) + 1e-9)
        x1, y1 = float(k_values[0]), float(norm_inertias[0])
        x2, y2 = float(k_values[-1]), float(norm_inertias[-1])
        line_len = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-9

        distances = []
        for i, k in enumerate(k_values):
            x0, y0 = float(k), float(norm_inertias[i])
            dist = np.abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1) / line_len
            distances.append(dist)
        optimal_k_elbow = k_values[int(np.argmax(distances))]

        return {
            "k_values": k_values,
            "inertias": [round(x, 2) for x in inertias],
            "silhouettes": silhouettes,
            "davies_bouldin": davies_bouldin_scores,
            "calinski_harabasz": calinski_scores,
            "optimal_k_silhouette": optimal_k_sil,
            "optimal_k_elbow": optimal_k_elbow,
            "recommended_k": optimal_k_sil if silhouettes[best_sil_idx] >= 0.35 else optimal_k_elbow
        }

    def fit(self, X: np.ndarray) -> "CustomerClusterEngine":
        """
        Fit K-Means model on preprocessed feature array.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.cluster_labels_ = self.model.fit_predict(X)
        self.centroids_scaled_ = self.model.cluster_centers_

        # Calculate evaluation metrics
        sil = float(silhouette_score(X, self.cluster_labels_))
        db = float(davies_bouldin_score(X, self.cluster_labels_))
        ch = float(calinski_harabasz_score(X, self.cluster_labels_))

        self.metrics_ = {
            "inertia": round(float(self.model.inertia_), 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_score": round(db, 4),
            "calinski_harabasz_score": round(ch, 2),
            "n_clusters": self.n_clusters
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for preprocessed data.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def save(self, filepath: str):
        """
        Serialize model and parameters to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "n_clusters": self.n_clusters,
            "metrics": self.metrics_,
            "centroids_scaled": self.centroids_scaled_
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerClusterEngine":
        """
        Load model artifact from disk.
        """
        data = joblib.load(filepath)
        engine = cls(n_clusters=data["n_clusters"])
        engine.model = data["model"]
        engine.metrics_ = data["metrics"]
        engine.centroids_scaled_ = data["centroids_scaled"]
        return engine
