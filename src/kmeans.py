import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method — Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/elbow_curve.png')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_
print("Customer count per cluster:")
print(df['Cluster'].value_counts().sort_index())
df.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/clustered_data.csv', index=False)

cluster_profile = df.groupby('Cluster').mean().round(2)
key_cols = ['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','PRC_FULL_PAYMENT']
print("\nCluster Profile:")
print(cluster_profile[key_cols])
cluster_profile.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cluster_profile.csv')

print(cluster_profile[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']].to_string())

cluster_names = {
    0: 'High Risk — Cash Advance & Debt Users',
    1: 'Premium High Spenders',
    2: 'Active Moderate Spenders',
    3: 'Revolvers — Minimum Payers',
    4: 'Inactive / Dormant Users'
}