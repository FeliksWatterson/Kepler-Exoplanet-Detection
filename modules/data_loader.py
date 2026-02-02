import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        exit()

    # Select features
    features = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr'
    ]
    target = 'koi_disposition'

    # Filter
    df = df[df[target].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    # Label
    df['label'] = df[target].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    
    # Fill nan
    X = df[features].fillna(0)
    y = df['label']

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features, df, scaler