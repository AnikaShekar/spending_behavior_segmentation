import pandas as pd

df=pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cc_general.csv')

print(f"Shape:{df.shape}")
print(f"Columns:{df.columns.tolist()}")
print(df.head())
print(df.info())
print(df.describe())

df.drop(columns='CUST_ID',inplace=True)
print(f"Shape after dropping cust_id:{df.shape}")

df['CREDIT_LIMIT']=df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
df['MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("Scaled data sample:")
print(df_scaled.head())

df.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv',index=False)