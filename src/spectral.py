import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#Load data
df = pd.read_csv('data/cleaned_data.csv')

redundant_cols = ['PURCHASES_TRX', 'CASH_ADVANCE_TRX','ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']
df_cluster = df.drop(columns=redundant_cols)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

#PCA before clustering (same 85% variance threshold as other scripts)
pca_cluster = PCA(n_components=0.85, random_state=42)
df_pca = pca_cluster.fit_transform(df_scaled)

#Spectral Clustering
spectral = SpectralClustering(
    n_clusters=5,
    affinity='nearest_neighbors',
    n_neighbors=10,
    assign_labels='kmeans',
    n_init=10,
    random_state=42,
)

spectral_labels = spectral.fit_predict(df_pca)
df['Spectral_Cluster'] = spectral_labels

#Silhouette scores
kmeans_score = 0.242
agg_score = 0.217
spectral_score = silhouette_score(df_pca, spectral_labels)

print(f"K-Means Silhouette Score: {kmeans_score:.3f}")
print(f"Agglomerative Silhouette Score: {agg_score:.3f}")
print(f"Spectral Silhouette Score: {spectral_score:.3f}")
print(f"Difference (KM vs Spectral): {abs(spectral_score - kmeans_score):.3f}")

print("\nSpectral Cluster Sizes:")
print(pd.Series(spectral_labels).value_counts().sort_index())

key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE','CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']
spectral_profile = df.groupby('Spectral_Cluster')[key_cols].mean().round(2)
print("\nSpectral Cluster Profile:")
print(spectral_profile)

#PCA to 2 dimensions for visualisation
pca_viz = PCA(n_components=2)
X_pca = pca_viz.fit_transform(df_scaled)
kmeans_df = pd.read_csv('data/clustered_data.csv')
kmeans_labels = kmeans_df['Cluster'].values

colors = ['red', 'gold', 'steelblue', 'orange', 'green']

#Side-by-side scatter: K-Means vs Spectral
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0e1117')

for ax, lbls, title in zip(
    axes,
    [kmeans_labels, spectral_labels],
    ['K-Means Clustering', 'Spectral Clustering']
):
    ax.set_facecolor('#1e2130')

    for cluster in range(5):
        mask = lbls == cluster
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[cluster], alpha=0.5, s=15,
            label=f'Cluster {cluster}'
        )

    ax.set_title(title, color='white', fontsize=13)
    ax.set_xlabel('PC1', color='white')
    ax.set_ylabel('PC2', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor('#3d4470')

plt.suptitle('K-Means vs Spectral Clustering Comparison',
             color='white', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('reports/spectral_comparison_plot.png',
            bbox_inches='tight', facecolor='#0e1117')
plt.show()
plt.close()

#Bar chart - 3-algorithm silhouette comparison
fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0e1117')
ax.set_facecolor('#1e2130')

algorithms = ['K-Means', 'Agglomerative', 'Spectral']
scores = [kmeans_score, agg_score, spectral_score]
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

df.to_csv('data/spectral_data.csv', index=False)