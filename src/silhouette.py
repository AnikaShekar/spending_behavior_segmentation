import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

df = pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/clustered_data.csv')

X = df.drop(columns=['Cluster'])
labels = df['Cluster']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Silhouette Score: It measures how well each customer fits in their assigned cluster.

For every customer it asks two things:
A = How close am I to others in MY cluster?
B = How far am I from the NEAREST other cluster?
Score = (B - A) / max(A, B)

Close to +1 -> Customer fits perfectly in their cluster
Close to 0 -> Customer is on the border between two clusters
Negative -> Customer probably belongs to a different cluster
"""
score = silhouette_score(X_scaled, labels)
print(f"Overall Silhouette Score: {score:.3f}")

"""
Overall Score: 0.193

1.0  → Perfect
0.5+ → Strong clusters
0.3+ → Reasonable clusters
0.2  → Weak but acceptable ← You are here
0.0  → No structure

0.193 is weak but acceptable for a real world dataset with 17 features and overlapping behaviors.
"""
# Per Cluster Silhouette Score
sample_scores = silhouette_samples(X_scaled, labels)

cluster_names = {
    0: 'High Risk',
    1: 'Premium Spenders',
    2: 'Active Moderate',
    3: 'Revolvers',
    4: 'Inactive Users'
}

print("\nPer Cluster Silhouette Score:")
for cluster in range(5):
    mask = labels == cluster
    cluster_score = sample_scores[mask].mean()
    print(f"Cluster {cluster} ({cluster_names[cluster]}): {cluster_score:.3f}")

"""
Per Cluster Silhouette Score:
Cluster 0 (High Risk): 0.002
Cluster 1 (Premium Spenders): -0.033
Cluster 2 (Active Moderate): 0.140
Cluster 3 (Revolvers): 0.341
Cluster 4 (Inactive Users): 0.186

Per Cluster Breakdown:
Revolvers -> 0.341 Best defined cluster — customers clearly belong here
Active Moderate -> 0.140 Moderate fit — some overlap with revolvers
Inactive Users -> 0.186 Decent fit — fairly well separated
High Risk -> 0.002 Almost on the border — barely fits
Premium Spenders -> -0.033 Negative — some customers may belong elsewhere
"""


# Silhouette Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['red', 'gold', 'steelblue', 'orange', 'green']

y_lower = 10
for cluster in range(5):
    cluster_scores = sample_scores[labels == cluster]
    cluster_scores.sort()

    size = cluster_scores.shape[0]
    y_upper = y_lower + size

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_scores,
        alpha=0.7,
        color=colors[cluster],
        label=cluster_names[cluster]
    )

    y_lower = y_upper + 10

ax.axvline(x=score, color='black', linestyle='--', label=f'Avg Score: {score:.3f}')
ax.set_title('Silhouette Plot — Customer Segments', fontsize=14)
ax.set_xlabel('Silhouette Score')
ax.set_ylabel('Customers grouped by Cluster')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/silhouette_plot.png')
plt.show()

"""
Each colored shape represents one cluster:
Width of shape = number of customers at that score

Wide shape on right side → many customers have HIGH score → good cluster
Wide shape on left side  → many customers have LOW/negative score → poor cluster

What you see in YOUR plot:
    Orange (Revolvers)  biggest shape, extends to 0.5
                        Widest and tallest shape — most customers score well — best defined cluster

    Blue (Active Moderate)  large shape, moderate scores
                            Most customers between 0.0 and 0.3 — acceptable fit

    Green (Inactive Users)  small but decent shape
                            Compact and mostly positive — well separated from others

    Yellow (Premium Spenders)   thin shape, extends left
                                Thin shape = small cluster (only 395 customers)
                                Extends to negative scores = some premium customers overlap with others

Red (High Risk) — shape extends far left
Many customers with negative scores = significant overlap with other clusters
Makes sense — high risk customers share some features with revolvers
"""

"""
Why are High Risk and Premium Spenders weak?
High Risk (0.002):
These customers have high balance AND high cash advance — but some also have moderate purchases. They sit between Revolvers and Premium Spenders in feature space. Hard boundary.

Premium Spenders (-0.033):
Only 395 customers — small cluster. Some heavy spenders also carry high balance, making them similar to High Risk customers in some dimensions.

Is this a problem for your project?
No — and here's why:
Real world customer data is never perfectly separable. Human behavior overlaps. A score of 0.193 with meaningful business segments is completely acceptable.
What matters more is business interpretability — and your clusters make perfect business sense.
"""