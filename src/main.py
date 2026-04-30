import pandas as pd
from sklearn.preprocessing import StandardScaler

#Load raw dataset
df = pd.read_csv('data/cc_general.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
print(df.info())
print(df.describe())

#Drop identifier column
df.drop(columns='CUST_ID', inplace=True)
print(f"Shape after dropping CUST_ID: {df.shape}")

#Impute missing values
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

print(df.isnull().sum())

#Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print("Scaled data sample:")
print(df_scaled.head())
df.to_csv('data/cleaned_data.csv', index=False)