import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv')

df.hist(bins=30,figsize=(20,15),color='steelblue',edgecolor='black')
plt.suptitle('Feature Distribution',fontsize=16) #super title — one main title for the entire figure
plt.tight_layout() #automatically adjusts spacing between subplots
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/distributions.png')
plt.show()

"""Histogram: It shows how many customers fall in each value range for a feature.
X axis = value range of the feature (e.g. BALANCE from 0 to 19000)
Y axis = number of customers in that range

Most of the data values are clustered on the left {meaning it is right-skewed}"""


plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap='coolwarm',linewidths=0.5)
plt.title('Features Correlation Heatmap')
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/correlation_heatmap.png')
plt.show()


"""
df.corr() - The correlation matrix.this is the data being plotted
annot=True - Shows the **actual number** inside each cell
fmt='.2f' - Formats numbers to **2 decimal places** e.g. 0.87
cmap='coolwarm' - Color scheme — **red = high correlation, blue = low/negative**
linewidths=0.5 - Adds thin lines between cells — easier to read

Dark Red pairs = redundant features = candidates for removal in PCA later
White/Blue pairs = independent features = most valuable for clustering

PURCHASES & PURCHASES_TRXMore transactions = more spending
PURCHASES & ONEOFF_PURCHASESOne-off purchases make up most of total
CASH_ADVANCE & CASH_ADVANCE_TRXMore cash advance transactions = higher cash advance amount
CASH_ADVANCE_FREQUENCY & CASH_ADVANCE_TRXSame logic
"""


plt.figure(figsize=(12, 5))
cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(f'Outliers in {col}')
plt.tight_layout()
plt.savefig('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/reports/boxplots.png')
plt.show()

"""
boxplot tells you something different — Where are the outliers? And how spread out is the data?
This matters for clustering because outliers can pull your cluster centers in the wrong direction — messing up your segments

Lots of dots above the whisker = heavy right skew = confirms what your histogram showed
Box sitting near the bottom = most customers are low value = a clear majority segment exists

Balance has most outliers
You're right — and here's why it makes sense:

Most people maintain a low credit card balance, but a small group of customers carry extremely high balances month after month — possibly because they can't pay it off fully.
This group will likely become its own cluster — "High Risk / High Balance customers" 

median sitting closer to the bottom means : Majority of customers have low values, but the box stretches upward because some customers have moderately high values too
"""




"""
What observations to draw — specifically:
From Distribution Plots:

Almost every feature is right skewed
Meaning majority of customers are low activity — low balance, low purchases, low cash advance
Very few customers are high activity
PURCHASES_FREQUENCY likely shows two peaks — customers either buy frequently or rarely, nothing in between

From Correlation Heatmap:

PURCHASES and PURCHASES_TRX are strongly correlated — both measure purchase behavior
ONEOFF_PURCHASES and PURCHASES are strongly correlated — one-off purchases dominate total purchases
CASH_ADVANCE and CASH_ADVANCE_TRX are strongly correlated — obvious relationship
These correlated pairs tell you: you don't need all 17 features — PCA later will handle this

From Boxplots:

BALANCE has the most outliers — small group of very high balance customers
PURCHASES has extreme outliers — few customers are very heavy shoppers
CREDIT_LIMIT is more evenly spread — fewer outliers
Median sitting near the bottom in all 3 — confirms majority are low-value customers
"""


"""
Your 5 Customer Segments:
Inactive Users
Active Low Spenders
Big Ticket Shoppers
Cash Advance Dependents
Responsible Spenders
"""