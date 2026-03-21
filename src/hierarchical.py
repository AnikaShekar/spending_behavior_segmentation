import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
agg_labels = agg.fit_predict(df_scaled)

df['Agg_Cluster'] = agg_labels

agg_score = silhouette_score(df_scaled, agg_labels)
print(f"Agglomerative Silhouette Score: {agg_score:.3f}")
print(f"K-Means Silhouette Score:       0.193")
print(f"Difference: {abs(agg_score - 0.193):.3f}")

print("\nAgglomerative Cluster Sizes:")
print(pd.Series(agg_labels).value_counts().sort_index())

key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE',
            'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']

agg_profile = df.groupby('Agg_Cluster')[key_cols].mean().round(2)
print("\nAgglomerative Cluster Profile:")
print(agg_profile)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0e1117')
colors = ['red', 'gold', 'steelblue', 'orange', 'green']

kmeans_df = pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/clustered_data.csv')
kmeans_labels = kmeans_df['Cluster'].values

axes[0].set_facecolor('#1e2130')
for cluster in range(5):
    mask = kmeans_labels == cluster
    axes[0].scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[cluster], alpha=0.5, s=15,
        label=f'Cluster {cluster}'
    )
axes[0].set_title('K-Means Clustering', color='white', fontsize=13)
axes[0].set_xlabel('PC1', color='white')
axes[0].set_ylabel('PC2', color='white')
axes[0].tick_params(colors='white')
axes[0].legend(facecolor='#1e2130', labelcolor='white', fontsize=8)
for spine in axes[0].spines.values():
    spine.set_edgecolor('#3d4470')

axes[1].set_facecolor('#1e2130')
for cluster in range(5):
    mask = agg_labels == cluster
    axes[1].scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[cluster], alpha=0.5, s=15,
        label=f'Cluster {cluster}'
    )
axes[1].set_title('Agglomerative Clustering', color='white', fontsize=13)
axes[1].set_xlabel('PC1', color='white')
axes[1].set_ylabel('PC2', color='white')
axes[1].tick_params(colors='white')
axes[1].legend(facecolor='#1e2130', labelcolor='white', fontsize=8)
for spine in axes[1].spines.values():
    spine.set_edgecolor('#3d4470')

plt.suptitle('K-Means vs Agglomerative Clustering Comparison',
             color='white', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/comparison_plot.png',
            bbox_inches='tight', facecolor='#0e1117')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0e1117')
ax.set_facecolor('#1e2130')

algorithms = ['K-Means', 'Agglomerative']
scores = [0.193, agg_score]
bar_colors = ['steelblue', 'coral']

bars = ax.bar(algorithms, scores, color=bar_colors, width=0.4)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{score:.3f}',
        ha='center', color='white', fontsize=12
    )

ax.set_ylabel('Silhouette Score', color='white')
ax.set_title('Algorithm Comparison — Silhouette Score',
             color='white', fontsize=12)
ax.tick_params(colors='white')
ax.set_ylim(0, max(scores) + 0.05)
for spine in ax.spines.values():
    spine.set_edgecolor('#3d4470')

plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/algorithm_comparison.png',facecolor='#0e1117')
plt.show()

df.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/hierarchical_data.csv', index=False)