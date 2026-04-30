import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')

#Histogram - distribution of every feature
df.hist(bins=30, figsize=(20, 15), color='steelblue', edgecolor='black')
plt.suptitle('Feature Distribution', fontsize=16)
plt.tight_layout()

plt.savefig('reports/distributions.png')
plt.show()

#Heatmap - pairwise Pearson correlations
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

plt.title('Features Correlation Heatmap')
plt.tight_layout()
plt.savefig('reports/correlation_heatmap.png')
plt.show()

#4. Box-plots - outlier detection for key columns
cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(f'Outliers in {col}')

plt.tight_layout()
plt.savefig('reports/boxplots.png')
plt.show()