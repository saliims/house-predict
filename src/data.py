import numpy as np
import pandas as pd
from src.utils import add_bias, z_score_normalize
from sklearn.model_selection import train_test_split

def load_and_prepare(path):
    df = pd.read_csv(path)

    df_numeric = df.select_dtypes(include=[np.number])
    threshold = 0.05 * len(df_numeric)
    df_clean = df_numeric.dropna(axis=1, thresh=len(df_numeric) - threshold)
    df_clean = df_clean.dropna(axis=0)

    y = df_clean["SalePrice"].values
    X = df_clean.drop(columns=["SalePrice"]).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_n, X_val_n, mu, sigma = z_score_normalize(X_train, X_val)

    X_train_b = add_bias(X_train_n)
    X_val_b   = add_bias(X_val_n)

    y_mu    = y_train.mean()
    y_sigma = y_train.std()
    y_train_n = (y_train - y_mu) / y_sigma

    return X_train_b, X_val_b,X_train_n, X_val_n, y_train, y_val, y_train_n, y_mu, y_sigma