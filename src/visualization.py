import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/clustered_data.csv')

X = df.drop(columns=['Cluster'])
labels = df['Cluster']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance explained by each component:")
print(pca.explained_variance_ratio_.round(3))
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels

cluster_names = {
    0: 'High Risk',
    1: 'Premium Spenders',
    2: 'Active Moderate',
    3: 'Revolvers',
    4: 'Inactive Users'
}

plt.figure(figsize=(10, 7))
colors = ['red', 'gold', 'steelblue', 'orange', 'green']

for cluster in range(5):
    mask = pca_df['Cluster'] == cluster
    plt.scatter(
        pca_df[mask]['PC1'],
        pca_df[mask]['PC2'],
        c=colors[cluster],
        label=cluster_names[cluster],
        alpha=0.5,
        s=20
    )

plt.title('Customer Segments — PCA Visualization', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/cluster_scatter.png')
plt.show()

plt.figure(figsize=(8, 5))
cluster_counts = df['Cluster'].value_counts().sort_index()
bars = plt.bar(
    [cluster_names[i] for i in cluster_counts.index],
    cluster_counts.values,
    color=colors
)

for bar, count in zip(bars, cluster_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 30,
        str(count),
        ha='center',
        fontsize=10
    )

plt.title('Customer Count per Segment', fontsize=14)
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/cluster_sizes.png')
plt.show()

key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE','CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']

cluster_profile = df.groupby('Cluster')[key_cols].mean().round(2)
cluster_profile.index = [cluster_names[i] for i in cluster_profile.index]

plt.figure(figsize=(10, 5))
sns.heatmap(
    cluster_profile.T,
    annot=True,
    fmt='.0f',
    cmap='YlOrRd',
    linewidths=0.5
)
plt.title('Cluster Feature Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/cluster_heatmap.png')
plt.show()