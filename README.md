# CustomerIQ — AI-Powered Customer Segmentation & Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Machine Learning analytics platform and interactive web dashboard designed to personalize e-commerce customer experiences. **CustomerIQ** solves the problem of generic, one-size-fits-all marketing by transforming customer transaction and interaction data into high-affinity behavioral clusters, extracting defining segment characteristics, and quantifying financial return through an interactive **+15% conversion lift ROI simulator**.

---

## 📌 Problem Statement & Objectives

> **Context**: An e-commerce company wants to personalize its user experience by implementing a customer segmentation model to categorize customers based on preferences and behaviors, enabling targeted marketing, personalized recommendations, and reduced churn.

### Core Objectives
1. **Develop an Unsupervised ML Segmentation Engine**: Implement K-Means clustering with optimal cluster validation using the **Elbow Method** (WCSS) and **Silhouette Score Analysis**.
2. **Understand Distinct Customer Segments & Characteristics**: Automatically identify the **top three defining characteristics** that statistically distinguish each customer segment from the store-wide baseline.
3. **Actionable Marketing Playbooks & Revenue Impact**: Formulate personalized omnichannel strategies for each segment and simulate the business goal of **increasing conversion rates by 15%**.

---

## 🌟 Key Features

- **Real Customer Data Preprocessing Pipeline**:
  - Robust handling of outliers via **IQR-based Winsorization** or threshold filtering.
  - Multi-method feature scaling (**StandardScaler**, **RobustScaler**, **MinMaxScaler**).
  - Automated **RFM Quantile Scoring** (Recency, Frequency, Monetary composite indices).
  - **PCA Dimensionality Reduction** (2D & 3D projections retaining >85% cumulative variance).
- **Mathematical Cluster Optimization**:
  - Dual-metric cluster validation across $k \in [2, 9]$ displaying WCSS (Inertia) and Silhouette Scores.
  - Multi-criterion scoring: **Davies-Bouldin Index** (cohesion vs separation) and **Calinski-Harabasz Index** (variance ratio).
  - Automated Knee/Elbow detection using perpendicular 2D distance algorithms.
- **Automated Persona Profiling**:
  - Real-time extraction of top 3 defining characteristics via normalized $z$-score deviations.
  - Intuitive personas: **VIP Champions**, **Loyal Steady Spenders**, **Bargain & Deal Seekers**, **At-Risk High-Value Lapsers**, and **New Promising Explorers**.
  - Tailored marketing strategies, campaign copy hooks, and recommended messaging channels.
- **Interactive Visualizations (Plotly)**:
  - Dual-axis Elbow & Silhouette curve with optimal $k$ highlight.
  - Interactive 2D & 3D PCA scatter plots with centroid overlays and detailed customer hover cards.
  - Multi-cluster Radar / Spider chart comparing normalized behavioral dimensions.
  - Feature distribution boxplots and interactive correlation matrix heatmap.
- **Business Impact & +15% Conversion ROI Simulator**:
  - Directly simulates financial outcomes for target campaigns (Incremental Orders, Revenue Lift, Net Profit, ROAS, and ROI %).
  - Full portfolio impact breakdown table across all segments.
- **CSV Data Ingestion & Batch Export**:
  - Ingest custom customer transaction CSV files with intelligent column auto-mapping.
  - 1-click export of clustered datasets with assigned personas and actionable marketing playbooks.

---

## 🏗️ System Architecture

```
ML-Project/
├── .streamlit/
│   └── config.toml                  # Dark slate theme and server configuration
├── data/
│   ├── generate_data.py             # Realistic e-commerce customer transaction generator
│   ├── customer_data.csv            # 2,500 customer records (RFM, visits, returns, spend)
│   └── customer_segments_output.csv # Pre-computed customer clusters with persona labels
├── models/
│   ├── train_model.py               # Standalone training & evaluation pipeline
│   ├── kmeans_model.pkl             # Trained K-Means model artifact
│   ├── scaler.pkl                   # Trained feature scaler artifact
│   ├── pca_model.pkl                # Trained 3D PCA dimensionality reducer
│   └── training_metadata.json       # Serialized metrics, loadings, and cluster profiles
├── src/
│   ├── __init__.py                  # Package init
│   ├── data_loader.py               # Data loading, validation, and schema auditing
│   ├── preprocessor.py              # Outlier handling, scaling, RFM scoring, PCA
│   ├── model.py                     # K-Means clustering, Elbow method, Silhouette scoring
│   ├── persona.py                   # Automated persona generation & top 3 characteristics
│   ├── simulator.py                 # +15% conversion lift & campaign ROI financial engine
│   └── visualizer.py                # Modern Plotly charts (Elbow, 2D/3D PCA, Radar, Heatmaps)
├── tests/
│   └── test_pipeline.py             # Complete unit and integration test suite
├── app.py                           # Full-featured Streamlit production dashboard
├── requirements.txt                 # Pinned dependencies
├── project_report.md                # Formal Machine Learning Project Report
├── run_app.bat                      # Windows 1-click launcher
└── README.md                        # Project documentation
```

---

## 🔬 Mathematical Formulations

### 1. K-Means Objective Function (Inertia / WCSS)
Given a dataset $X = \{x_1, x_2, \dots, x_n\} \subset \mathbb{R}^d$, K-Means partitions $X$ into $K$ disjoint clusters $C = \{C_1, C_2, \dots, C_K\}$ minimizing the Within-Cluster Sum of Squares (WCSS):
$$\min_{C, \mu} J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
where $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the centroid of cluster $C_k$.

### 2. Silhouette Coefficient
For sample $i$, let $a(i)$ be the mean intra-cluster distance and $b(i)$ be the lowest mean distance to any other cluster:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
The overall score is $\bar{s} = \frac{1}{N} \sum_{i=1}^N s(i)$.

### 3. Davies-Bouldin Index
$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \quad DB = \frac{1}{K}\sum_{i=1}^K \max_{j \ne i} R_{ij}$$
where $s_i$ is average distance from points in cluster $i$ to centroid $\mu_i$. Lower is better.

### 4. Principal Component Analysis (PCA)
Features are standardized ($Z = \frac{X - \mu}{\sigma}$), covariance matrix $\Sigma = \frac{1}{n-1}Z^T Z$ is computed, and eigenvectors $V$ are extracted:
$$\Sigma v_i = \lambda_i v_i$$
Principal components are obtained via projection: $P = Z V_k$.

---

## 👥 Customer Persona Breakdown

| Segment Persona | Audience Share | Revenue Share | Top 3 Defining Characteristics | Primary Business Playbook |
| :--- | :---: | :---: | :--- | :--- |
| **💎 VIP Champions** | 18.0% | **44.8%** | ⬆️ Spend (+154%), ⬆️ Frequency (+107%), ⬆️ Visits (+88%) | Exclusive VIP pre-access drops, zero-threshold shipping, concierge support |
| **🌟 Loyal Steady** | 28.0% | **30.2%** | ⬇️ Returns (-26%), ⬇️ Discount usage (-23%), ⬆️ Frequency (+18%) | Category cross-selling, dynamic bundles, milestone frequency rewards |
| **⚠️ At-Risk High-Value** | 16.0% | **12.0%** | ⬆️ Recency (+151%), ⬆️ Returns (+94%), ⬇️ Session time (-53%) | Automated Day-60/90 win-back flows, high-value voucher, exit surveys |
| **🏷️ Bargain & Deal Seekers** | 24.0% | **9.1%** | ⬆️ Discount (+111%), ⬇️ AOV (-53%), ⬇️ Spend (-61%) | Flash sales, minimum-spend coupons ($120+), clearance bundles |
| **🚀 New Promising Explorers** | 14.0% | **3.9%** | ⬇️ Frequency (-81%), ⬇️ Items (-81%), ⬇️ Spend (-86%) | 3-part educational welcome drip, 10% second-order coupon within 14 days |

---

## 🚀 Quickstart & Installation

### 1. Clone or Navigate to the Workspace
```bash
cd c:\Users\HP\OneDrive\My_Projects\ML-Project
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train ML Model & Regenerate Artifacts
```bash
python models/train_model.py
```

### 4. Run the Unit & Integration Tests
```bash
python tests/test_pipeline.py
```

### 5. Launch the Interactive Streamlit Platform
```bash
streamlit run app.py
```
Or simply double-click `run_app.bat` on Windows. The application will be accessible at:
```
http://localhost:8501
```

---

## ☁️ Production Deployment Guide

### Deploy to Streamlit Community Cloud (Free & 1-Click)
1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "CustomerIQ: Production commit"
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, specify `app.py` as the entrypoint, and deploy.

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t customeriq:latest .
docker run -p 8501:8501 customeriq:latest
```

---

## 🧪 Automated Testing

CustomerIQ includes a comprehensive test suite covering schema integrity, KPI computations, preprocessing pipelines, elbow/silhouette calculations, K-Means clustering, persona classification, and ROI financial simulators.

Run the test suite:
```bash
python tests/test_pipeline.py
```
Output:
```
test_01_data_loading_and_kpis ... ok
test_02_validation ... ok
test_03_column_auto_mapping ... ok
test_04_preprocessor ... ok
test_05_cluster_range_evaluation ... ok
test_06_model_fit_and_persona_generation ... ok
test_07_simulator ... ok
----------------------------------------------------------------------
Ran 7 tests in 0.52s

OK
```

---

## 📄 License
This project is open-source and distributed under the [MIT License](LICENSE).
