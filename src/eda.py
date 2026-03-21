import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv')

df.hist(bins=30,figsize=(20,15),color='steelblue',edgecolor='black')
plt.suptitle('Feature Distribution',fontsize=16)
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/distributions.png')
plt.show()

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap='coolwarm',linewidths=0.5)
plt.title('Features Correlation Heatmap')
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/correlation_heatmap.png')
plt.show()

plt.figure(figsize=(12, 5))
cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(f'Outliers in {col}')
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/boxplots.png')
plt.show()