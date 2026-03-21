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

score = silhouette_score(X_scaled, labels)
print(f"Overall Silhouette Score: {score:.3f}")

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