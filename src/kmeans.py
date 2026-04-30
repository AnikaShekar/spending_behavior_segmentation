import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#Load data
df = pd.read_csv('data/cleaned_data.csv')

redundant_cols = ['PURCHASES_TRX','CASH_ADVANCE_TRX','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']
df_cluster = df.drop(columns=redundant_cols)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

#PCA - dimensionality reduction before clustering
pca = PCA(n_components=0.85, random_state=42)
df_pca = pca.fit_transform(df_scaled)

print(f"PCA - Components kept: {pca.n_components_}  |  "
      f"Variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

#Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_temp.fit(df_pca)
    inertia.append(km_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method - Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('reports/elbow_curve.png')
plt.show()

#Train K-Means model (K = 5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
kmeans.fit(df_pca)

df['Cluster'] = kmeans.labels_

score = silhouette_score(df_pca, kmeans.labels_)
print(f"Silhouette Score for K=5: {score:.3f}")

print("\nCustomer count per cluster:")
print(df['Cluster'].value_counts().sort_index())

key_cols = ['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','PRC_FULL_PAYMENT',]

profile = df.groupby('Cluster')[key_cols].mean().round(2)

#Assign business-meaningful names via rule-based logic
unassigned = list(range(5))
cluster_names = {}

#Rule 1 - Highest CASH_ADVANCE -> cash-heavy / high-risk users
high_risk_id = profile['CASH_ADVANCE'].idxmax()
cluster_names[high_risk_id] = 'High Risk - Cash Advance & Debt Users'
unassigned.remove(high_risk_id)

#Rule 2 - Highest PURCHASES among remaining -> big spenders
premium_id = profile.loc[unassigned, 'PURCHASES'].idxmax()
cluster_names[premium_id] = 'Premium High Spenders'
unassigned.remove(premium_id)

#Rule 3 - Lowest PURCHASES among remaining -> barely use the card
inactive_id = profile.loc[unassigned, 'PURCHASES'].idxmin()
cluster_names[inactive_id] = 'Inactive / Dormant Users'
unassigned.remove(inactive_id)

# Rule 4 - Highest BALANCE among remaining -> carry revolving debt
revolver_id = profile.loc[unassigned, 'BALANCE'].idxmax()
cluster_names[revolver_id] = 'Revolvers - Minimum Payers'
unassigned.remove(revolver_id)

# Rule 5 - Only cluster left -> moderate everyday spenders
moderate_id = unassigned[0]
cluster_names[moderate_id] = 'Active Moderate Spenders'

print("\nFINAL CLUSTER NAME ASSIGNMENTS")
for cid in sorted(cluster_names.keys()):
    print(f"\nCluster {cid} - {cluster_names[cid]}")
    print(f"Balance : {profile.loc[cid, 'BALANCE']}")
    print(f"Purchases : {profile.loc[cid, 'PURCHASES']}")
    print(f"Cash Advance : {profile.loc[cid, 'CASH_ADVANCE']}")
    print(f"Credit Limit : {profile.loc[cid, 'CREDIT_LIMIT']}")
    print(f"Payments : {profile.loc[cid, 'PAYMENTS']}")
    print(f"Full Pay % : {profile.loc[cid, 'PRC_FULL_PAYMENT']}")
cluster_profile = df.groupby('Cluster').mean().round(2)
cluster_profile.to_csv('data/cluster_profile.csv')

df.to_csv('data/clustered_data.csv', index=False)

os.makedirs('models', exist_ok=True)

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca.pkl')
joblib.dump(kmeans, 'models/kmeans.pkl')

cluster_names_str_keys = {str(k): v for k, v in cluster_names.items()}
with open('models/cluster_names.json', 'w') as f:
    json.dump(cluster_names_str_keys, f, indent=2)