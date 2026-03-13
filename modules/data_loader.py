import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    features = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
                'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr']
    target = 'koi_disposition'

    df = df[df[target].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df['label'] = df[target].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    
    X_raw = df[features]
    y = df['label']

    return X_raw, y, features, df