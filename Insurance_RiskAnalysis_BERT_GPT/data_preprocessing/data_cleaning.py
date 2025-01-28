import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df