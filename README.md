# AI-Based Spending Behavior Segmentation
A machine learning project that segments credit card customers into distinct behavioral groups using K-Means clustering.

# Author
Anika Varsha Shekar -@AnikaShekar

# Project Overview
This project analyzes credit card customer data to identify spending patterns and group customers into meaningful segments. The results are visualized through an interactive Streamlit dashboard.
- Dataset: cc_general - Credit Card Customer Dataset (Kaggle)
- Algorithm: K-Means Clustering
- Segments Found: 5
- Total Customers: 8,950
- Features Used: 17

# Customer Segments Identified
| Segment | Customers | Description |
|---|---|---|
| 🔴 High Risk — Cash Advance & Debt Users | 987 | High balance, high cash advance, barely pays back |
| 🟡 Premium High Spenders | 395 | Highest purchases and credit limit |
| 🔵 Active Moderate Spenders | 3164 | Largest group, moderate spending behavior |
| 🟠 Revolvers — Minimum Payers | 3047 | Carry debt, very low full payment rate |
| 🟢 Inactive / Dormant Users | 1357 | Lowest balance and purchases |

---

# Tech Stack
- Python - Core programming language
- Pandas - Data loading and manipulation
- NumPy - Numerical operations
- Scikit-learn - K-Means clustering, PCA, StandardScaler
- Matplotlib - Data visualization
- Seaborn - Statistical plots
- Streamlit - Interactive dashboard

# Project Structure
```
spending-segmentation/
│
├── data/
│   ├── cc_general.csv
│   ├── cleaned_data.csv
│   ├── clustered_data.csv
│   └── cluster_profile.csv
│
├── src/
│   ├── main.py
│   ├── eda.py
│   ├── kmeans.py
│   ├── visualization.py
│   └── silhouette.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── distributions.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── elbow_curve.png
│   ├── cluster_scatter.png
│   ├── cluster_sizes.png
│   ├── cluster_heatmap.png
│   └── silhouette_plot.png
│
└── README.md
```

# Results
- Optimal K (clusters) - 5
- Silhouette Score - 0.193
- PCA Variance Explained - 47.6%
- Largest Segment - Active Moderate Spenders (35.4%)
- Smallest Segment - Premium High Spenders (4.4%)
