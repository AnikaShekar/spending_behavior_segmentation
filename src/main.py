import pandas as pd

df=pd.read_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cc_general.csv')

print(f"Shape:{df.shape}")
print(f"Columns:{df.columns.tolist()}")
print(df.head())
print(df.info())
print(df.describe())

#handling missing values
#column 0 can be dropped bcuz it is not neede in ml for clustering
df.drop(columns='CUST_ID',inplace=True)
print(f"Shape after dropping cust_id:{df.shape}")    #shape:(8950,17)

#column 13 has 1 null values... fill it with median
df['CREDIT_LIMIT']=df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
#column 15 has 313 null values... fill it with median
df['MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
print(df.isnull().sum())

#Scaling the Data
#This is crucial for clustering — all features must be on the same scale, otherwise large values like [BALANCE](ranges between 0-19000) will dominate over small values like [PRC_FULL_PAYMENT](ranges between 0-1).
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #transforms all the values to same scale so there is no domination {z = (x - mean) / standard_deviation}
df_scaled = scaler.fit_transform(df) #Learns the mean & std from data, then scales it
# Convert back to dataframe
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("Scaled data sample:")
print(df_scaled.head())

#save the scaled data
df.to_csv('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/data/cleaned_data.csv',index=False)
print("Cleaned and Scaled data saved")