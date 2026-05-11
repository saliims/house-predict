import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')
print(df.head())
print(f"Shape of the df: {df.shape}")
print(df.dtypes.value_counts())
print(f"Price range: {df['SalePrice'].describe()}")
print(df.isnull().sum().sort_values(ascending=False).head(10))

df_numeric = df.select_dtypes(include=[np.number])

threshold = 0.05 * len(df_numeric)
df_clean = df_numeric.dropna(axis=1, thresh=len(df_numeric) - threshold)

df_clean = df_clean.dropna(axis=0)

print(f"Shape after cleaning : {df_clean.shape}")
print(f"Remaining features: {df_clean.columns.tolist()}")