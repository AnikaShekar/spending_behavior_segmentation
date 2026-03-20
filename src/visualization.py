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

# 17 features → PCA → 2 numbers (PC1, PC2)
#                          ↓
#               Now you can plot on X and Y axis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance explained by each component:")
print(pca.explained_variance_ratio_.round(3))
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
"""
PC1 (X axis):27.3% -> Captures 27.3% of all information
PC2 (Y axis):20.3% -> Captures 20.3% of all information
Total:47.6% -> Together they capture 47.6%

Above 70% → Excellent
50% - 70% → Good
40% - 50% → Acceptable ← You are here, so its moderate(not gud but not bad)
Below 40% → Poor

47.6% is acceptable for a dataset with 17 features. 
It means your 2D plot shows roughly half the story - clusters may overlap slightly but overall structure is visible.
The other 52.4% of information is hidden in dimensions you can't see - clusters slightly overlap.
"""
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

"""
Cluster plot
Red cluster (High Risk) — top left area:
Spreads upward on PC2 axis
Many outlier dots floating high up
= These customers are extreme in their behavior
= High cash advance pulls them away from everyone else

Yellow cluster (Premium Spenders) — right side:
Spreads far right on PC1 axis
Some dots go very far right (up to 30!)
= High purchases pull them to the right
= PC1 is clearly capturing purchase behavior

Blue cluster (Active Moderate) — center bottom:
Tight group near the origin (0,0)
= Average customers, nothing extreme
= Sit in the middle because no feature dominates

Orange cluster (Revolvers) — bottom left:
Overlaps slightly with blue and green
= Similar to moderate customers but with debt behavior
= Overlap makes sense — they're not drastically different

Green cluster (Inactive) — far left bottom:
Tight small group on the left
= Very low values on everything
= PC1 negative = low spending behavior confirmed
"""

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

"""
Barplot
Active Moderate - 3164 (35%) ← majority
Revolvers 3047 - (34%) ← second
Inactive 1357 - (15%)
High Risk 987 - (11%)
Premium 395 - (4%) ← smallest but most valuable

69% of your customers are either Active Moderate or Revolvers — your bank's core customer base
Premium Spenders are only 4% — but generate the most revenue — most important to retain
High Risk is 11% — significant enough to worry about — default risk
"""

# ── Plot 3 — Feature Heatmap per Cluster ────────────────────────

key_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE',
            'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']

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

"""
What PC1 and PC2 Actually Represent:
PC1 (X axis) -> Overall spending activity — high purchases push right, low push left
PC2 (Y axis) -> Cash advance & balance behavior — high cash advance pushes up

This is why:
Yellow (high purchases) is far right
Red (high cash advance) is far up
Green (low everything) is far left and down
"""