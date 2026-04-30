# AI-Based Spending Behavior Segmentation

A machine learning project that segments credit card customers into distinct behavioral groups using clustering algorithms, with an interactive Streamlit dashboard for exploration and prediction.

## Author
**Anika Varsha Shekar** — [@AnikaShekar](https://github.com/AnikaShekar)

## Project Overview

This project analyzes credit card customer data to identify spending patterns and group customers into 5 meaningful segments using unsupervised machine learning. Three clustering algorithms are compared — K-Means, Agglomerative, and BIRCH — with K-Means selected as the final model based on silhouette score.

- **Dataset:** cc_general — Credit Card Customer Dataset (Kaggle)
- **Primary Algorithm:** K-Means Clustering
- **Algorithms Compared:** K-Means, Agglomerative, BIRCH, Spectral
- **Segments Found:** 5
- **Total Customers:** 8,950
- **Features Used:** 17 (13 after dropping redundant columns)

## Customer Segments Identified

| Segment | Customers | Description |
|---------|-----------|-------------|
| 🔴 High Risk — Cash Advance & Debt Users | 987 | High balance, high cash advance, barely pays back |
| 🟡 Premium High Spenders | 395 | Highest purchases and credit limit |
| 🔵 Active Moderate Spenders | 3,164 | Largest group, moderate spending behavior |
| 🟠 Revolvers — Minimum Payers | 3,047 | Carry debt, very low full payment rate |
| 🟢 Inactive / Dormant Users | 1,357 | Lowest balance and purchases |

## Algorithm Comparison

| Algorithm | Silhouette Score |
|-----------|-----------------|
| K-Means | 0.242 |
| Agglomerative | 0.217 |
| BIRCH | 0.169 |

K-Means was selected as the final model due to the highest silhouette score, interpretable cluster profiles, and efficient scalability on 8,950 customers.

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Clustering, PCA, StandardScaler, Silhouette |
| Matplotlib | Data visualisation |
| Seaborn | Statistical plots |
| Joblib | Model serialisation |
| Streamlit | Interactive dashboard |

## Project Structure
spending-segmentation/
│
├── data/
│   ├── cc_general.csv               ← Raw dataset (Kaggle)
│   ├── cleaned_data.csv             ← After preprocessing
│   ├── clustered_data.csv           ← With K-Means labels
│   ├── cluster_profile.csv          ← Mean KPIs per cluster
│   ├── hierarchical_data.csv        ← With Agglomerative labels
│   ├── birch_data.csv               ← With BIRCH labels
│   └── spectral_data.csv            ← With Spectral labels
│
├── models/
│   ├── scaler.pkl                   ← Fitted StandardScaler
│   ├── pca.pkl                      ← Fitted PCA
│   ├── kmeans.pkl                   ← Fitted K-Means model
│   └── cluster_names.json           ← Cluster ID to name mapping
│
├── src/
│   ├── main.py                      ← Data loading, cleaning, scaling
│   ├── eda.py                       ← Exploratory data analysis plots
│   ├── kmeans.py                    ← K-Means training + model saving
│   ├── visualization.py             ← Cluster visualisations
│   ├── hierarchical.py              ← Agglomerative clustering
│   ├── birch.py                     ← BIRCH clustering
│   ├── spectral.py                  ← Spectral clustering
│   ├── silhouette.py                ← Silhouette analysis (all algorithms)
│   └── predict.py                   ← CLI customer segment predictor
│
├── dashboard/
│   └── app.py                       ← Streamlit interactive dashboard
│
├── reports/
│   ├── distributions.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── elbow_curve.png
│   ├── cluster_scatter.png
│   ├── cluster_sizes.png
│   ├── cluster_heatmap.png
│   ├── silhouette_plot.png
│   ├── comparison_plot.png
│   ├── algorithm_comparison.png
│   ├── algorithm_comparison_3.png
│   ├── birch_comparison_plot.png
│   └── spectral_comparison_plot.png
│
├── requirements.txt
├── runtime.txt
└── README.md

## How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Run scripts in order
```bash
python src/main.py
python src/eda.py
python src/kmeans.py
python src/visualization.py
python src/hierarchical.py
python src/birch.py
python src/spectral.py
python src/silhouette.py
python src/predict.py
```

> Always run from the project root `spending-segmentation/`, not from inside `src/`

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Dashboard Features

**Tab 1 — Overview**
- Total customers, features, clusters, and silhouette scores
- Segment distribution table with percentages
- Pie chart of segment split
- Raw data sample

**Tab 2 — Cluster Explorer**
- Drill down into any segment
- KPI tiles: balance, purchases, cash advance, credit limit, full pay %
- Grouped bar chart: segment average vs overall average
- Segment sample data table

**Tab 3 — Customer Lookup**
- Look up any customer by index (0 to 8949)
- See their segment, KPIs, and profile
- Enter custom values to predict segment for a new customer
- Comparison table: your customer vs segment average

---

## Results

| Metric | Value |
|--------|-------|
| Optimal K | 5 |
| K-Means Silhouette Score | 0.242 |
| PCA Components (85% variance) | 7 |
| Largest Segment | Active Moderate Spenders (35.4%) |
| Smallest Segment | Premium High Spenders (4.4%) |
