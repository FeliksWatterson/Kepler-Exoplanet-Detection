import os
from sklearn.model_selection import train_test_split
from modules.data_loader import load_data
from modules.trainer import run_models
from modules.visualizer import plot_corr, plot_acc, plot_features, plot_cm

# Config
DATA_FILE = 'dataset/cumulative.csv'
OUT_DIR = 'outputs'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if __name__ == "__main__":
    # 1. Load
    print("Loading data...")
    X, y, feats, df, scaler = load_data(DATA_FILE) 

    # 2. EDA
    plot_corr(df, feats, OUT_DIR)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train & Compare
    results, best_model = run_models(X_train, X_test, y_train, y_test, OUT_DIR)

    # 5. Visualize
    print("Saving plots...")
    plot_acc(results, OUT_DIR)
    plot_features(best_model, feats, OUT_DIR)
    plot_cm(best_model, X_test, y_test, OUT_DIR)

    print("Done.")