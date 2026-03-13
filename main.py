import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modules.data_loader import load_data
from modules.trainer import run_models
from modules.visualizer import plot_corr, plot_acc, plot_features, plot_cm

DATA_FILE = 'dataset/cumulative.csv'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("[INFO] Loading data...")
    X, y, feats, df = load_data(DATA_FILE) 

    plot_corr(df, feats, OUT_DIR)

    # 1. Chia Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Scale 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw) # Lưu ý: chỉ dùng transform cho tập test

    print("[INFO] Training models...")
    results, best_model = run_models(X_train, X_test, y_train, y_test, OUT_DIR)

    print("[INFO] Saving plots...")
    plot_acc(results, OUT_DIR)
    plot_features(best_model, feats, OUT_DIR)
    plot_cm(best_model, X_test, y_test, OUT_DIR)
    print("[INFO] Execution complete.")