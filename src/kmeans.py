"""
How K-Means Works
Super simple explanation:
Step 1: You tell it K=5 (number of clusters)
Step 2: It randomly places 5 center points
Step 3: Every customer gets assigned to nearest center
Step 4: Centers move to the middle of their group
Step 5: Repeat steps 3-4 until nothing changes
Done! 5 groups formed.
The algorithm is just answering one question repeatedly:
"Which center point is this customer closest to?"
"""
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

"""
What is Inertia? Inertia measures how spread out customers are within their cluster.
Specifically — it calculates:For every customer, how far are they from their cluster's center point? Then adds all those distances together.
    Very high --> Clusters are loose, customers spread far from center
    Very low --> Clusters are tight, customers close to their center
    After elbow point --> Adding more clusters doesn't improve much
            Elbow Method:   Runs K-Means for K=1 to K=10, plots inertia for each K, and you look for the bend in the curve.
                            The point where it goes from dropping sharply to almost flat = elbow = best K
"""

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method — Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/elbow_curve.png')
plt.show()

"""
Reading Your Elbow Plot
What I see in your plot:
K=1  → 152000  (drop of ~23000)
K=2  → 129000  (drop of ~17000)
K=3  → 112000  (drop of ~7000)
K=4  → 105000  (drop of ~13000) ← still dropping
K=5  →  92000  (drop of ~5000)  ← curve starts flattening
K=6  →  87000  (drop of ~4000)
K=7  →  83000  (drop of ~3000)
K=8  →  76000  (drop of ~7000)
K=9  →  73000  (drop of ~3000)
K=10 →  67000

The Honest Truth About Your Plot:
Your curve is gradually decreasing without a super sharp elbow — this is actually common with real world datasets.
But look carefully:
K=1 to K=4 → drops steeply
K=4 to K=5 → noticeable drop still
K=5 onwards → starts getting flatter

The bend begins around K=4 to K=5 — after K=5 the improvements become smaller and smaller.

So What K Should You Use?
K=4:Mathematically where curve starts bending
K=5:Matches your earlier logical thinking + still meaningful drop
K=6:Diminishing returns begin here

Use k=5
"""


kmeans = KMeans(n_clusters=5, random_state=42) #Creates a K-Means model with 5 clusters, Fixes the random starting points(42 is just convention)
kmeans.fit(df_scaled) #Actually runs the algorithm on your scaled data — finds the 5 group
df['Cluster'] = kmeans.labels_ #A list of 8950 numbers (0,1,2,3,4) — one for each customer
# Before this line your df had 17 columns, after this line it has 18 columns — last one being [Cluster].
print("Customer count per cluster:")
print(df['Cluster'].value_counts().sort_index()) #Count customers per cluster
df.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/clustered_data.csv', index=False)
# Customer count per cluster:
# Cluster
# 0     987
# 1     395     Cluster 1 = only 395 customers - small but important
# 2    3164
# 3    3047     Clusters 2 and 3 together=70% of your customers
# 4    1357
# Name: count, dtype: int64


cluster_profile = df.groupby('Cluster').mean().round(2) #groupby('Cluster') = separates customers into 5 rooms//.mean() = calculates average feature values in each room//.round(2) = rounds to 2 decimal places
key_cols = ['BALANCE', #How much debt they carry
            'PURCHASES', #How much they shop
            'CASH_ADVANCE', #Do they withdraw cash from card
            'CREDIT_LIMIT', #How much the bank trusts them
            'PAYMENTS', #How much they actually pay back
            'PRC_FULL_PAYMENT'] #Do they pay fully or partially
print("\nCluster Profile:")
print(cluster_profile[key_cols]) #Displays only those 6 columns for each cluster
cluster_profile.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cluster_profile.csv')

print(cluster_profile[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']].to_string())

"""
         BALANCE  PURCHASES  CASH_ADVANCE  CREDIT_LIMIT  PAYMENTS  PRC_FULL_PAYMENT
Cluster                                                                            
0        4903.43     553.10       4983.09       8062.62   3858.81              0.04
1        3588.99    7815.73        661.79       9769.62   7409.10              0.29
2         930.36    1299.71        226.75       4272.48   1388.75              0.26
3        1526.12     255.46        794.78       3244.23    958.64              0.02
4         111.26     335.22        325.75       3687.35   1076.94              0.23

Cluster 0:High Risk -> This customer is withdrawing cash constantly, accumulating huge debt and barely paying it back. **Bank's most risky segment.**
Cluster 1:Premium Spender -> This customer shops heavily, has highest credit limit, barely uses cash advance. **Bank's most valuable segment.**
Cluster 2:Active Moderate -> Everyday normal customer. Shops regularly, pays moderately. **Largest segment — 3164 customers.**
Cluster 3:Revolvers -> Barely shops but carries debt and almost never pays full amount. Keeps rolling balance month to month. **Second largest segment — 3047 customers.**
Cluster 4:Inactive -> Card is barely used. Low everything. **Dormant customers — bank wants to re-engage these.**
"""

cluster_names = {
    0: 'High Risk — Cash Advance & Debt Users',
    1: 'Premium High Spenders',
    2: 'Active Moderate Spenders',
    3: 'Revolvers — Minimum Payers',
    4: 'Inactive / Dormant Users'
}