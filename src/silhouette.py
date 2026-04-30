import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

#Load saved outputs from each clustering script
kmeans_df = pd.read_csv('data/clustered_data.csv')
birch_df = pd.read_csv('data/birch_data.csv')
hier_df = pd.read_csv('data/hierarchical_data.csv')

redundant_cols = ['PURCHASES_TRX', 'CASH_ADVANCE_TRX','ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']
label_cols = ['Cluster', 'Agg_Cluster', 'Birch_Cluster']

cols_to_drop = [c for c in label_cols + redundant_cols if c in kmeans_df.columns]
X_cluster = kmeans_df.drop(columns=cols_to_drop)

print(f"Feature matrix shape: {X_cluster.shape}")
print(f"Features used: {X_cluster.columns.tolist()}")

scaler2 = StandardScaler()
X_scaled2 = scaler2.fit_transform(X_cluster)

#PCA - reduce to 85% variance threshold
pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_scaled2)

print(f"PCA components: {pca.n_components_}  |  "f"Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%")

#Extract cluster label arrays from each DataFrame
kmeans_labels = kmeans_df['Cluster'].values
agg_labels = hier_df['Agg_Cluster'].values
birch_labels = birch_df['Birch_Cluster'].values

# Overall silhouette scores (one score per algorithm)
kmeans_score = silhouette_score(X_pca, kmeans_labels)
agg_score = silhouette_score(X_pca, agg_labels)
birch_score = silhouette_score(X_pca, birch_labels)

print(f"\nK-Means Silhouette Score: {kmeans_score:.3f}")
print(f"Agglomerative Silhouette Score: {agg_score:.3f}")
print(f"BIRCH Silhouette Score: {birch_score:.3f}")

#Per-cluster silhouette scores
cluster_names = {
    0: 'High Risk - Cash Advance & Debt Users',
    1: 'Premium High Spenders',
    2: 'Active Moderate Spenders',
    3: 'Revolvers - Minimum Payers',
    4: 'Inactive / Dormant Users',
}

algo_data = [
    ('K-Means', kmeans_labels, kmeans_score),
    ('Agglomerative', agg_labels,    agg_score),
    ('BIRCH', birch_labels,  birch_score),
]

for algo_name, lbls, score in algo_data:
    print(f"\nPer-Cluster Silhouette Scores ({algo_name}):")
    samples = silhouette_samples(X_pca, lbls)

    for cluster in range(5):
        mask = lbls == cluster
        print(f"Cluster {cluster} ({cluster_names[cluster]}): {samples[mask].mean():.3f}")

#Silhouette subplot
colors = ['red', 'gold', 'steelblue', 'orange', 'green']

fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#0e1117')
fig.suptitle('Silhouette Plot Comparison — K-Means vs Agglomerative vs BIRCH',color='white', fontsize=14, y=1.02)

for ax, (algo_name, lbls, score) in zip(axes, algo_data):
    ax.set_facecolor('#1e2130')

    sample_scores = silhouette_samples(X_pca, lbls)
    y_lower = 10

    for cluster in range(5):
        cluster_scores = sample_scores[lbls == cluster].copy()
        cluster_scores.sort()
        size = len(cluster_scores)
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

    ax.axvline(x=score, color='white', linestyle='--', label=f'Avg: {score:.3f}')

    ax.set_title(f'{algo_name}\nSilhouette Score: {score:.3f}', color='white', fontsize=12)
    ax.set_xlabel('Silhouette Score', color='white')
    ax.set_ylabel('Customers grouped by Cluster', color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', facecolor='#1e2130', labelcolor='white', fontsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor('#3d4470')

plt.tight_layout()
plt.savefig('reports/silhouette_plot.png', bbox_inches='tight', facecolor='#0e1117')
plt.show()
plt.close()

#Summary bar chart
fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0e1117')
ax.set_facecolor('#1e2130')

algorithms = ['K-Means', 'Agglomerative', 'BIRCH']
scores = [kmeans_score, agg_score, birch_score]
bar_colors = ['steelblue', 'coral', 'mediumpurple']

bars = ax.bar(algorithms, scores, color=bar_colors, width=0.4)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f'{score:.3f}',
        ha='center', color='white', fontsize=12
    )

ax.set_ylabel('Silhouette Score', color='white')
ax.set_title('3-Algorithm Comparison - Silhouette Score', color='white', fontsize=12)
ax.tick_params(colors='white')
ax.set_ylim(0, max(scores) + 0.05)

for spine in ax.spines.values():
    spine.set_edgecolor('#3d4470')

plt.tight_layout()
plt.savefig('reports/algorithm_comparison_3.png', facecolor='#0e1117')
plt.show()
plt.close()