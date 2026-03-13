import os
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from modules.data_loader import load_data
from modules.trainer import run_models
from modules.visualizer import plot_corr, plot_acc, plot_features, plot_cm

DATA_FILE = 'dataset/cumulative.csv'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("[INFO] Loading raw data...")
    X_raw, y, feats, df = load_data(DATA_FILE) 

    plot_corr(df, feats, OUT_DIR)

    # 1. Chia Train/Test 
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Xử lý Missing Data
    print("[INFO] Applying Imputation and Standardization...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # 3. Standard Scale 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_test = scaler.transform(X_test_imputed) 

    print("[INFO] Training models...")
    results, best_model = run_models(X_train, X_test, y_train, y_test, OUT_DIR)

    # 4. Saving
    print("[INFO] Exporting inference artifacts (.pkl)...")
    joblib.dump(imputer, os.path.join(OUT_DIR, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))
    joblib.dump(best_model, os.path.join(OUT_DIR, 'best_model.pkl'))

    print("[INFO] Generating analytical plots...")
    plot_acc(results, OUT_DIR)
    plot_features(best_model, feats, OUT_DIR)
    plot_cm(best_model, X_test, y_test, OUT_DIR)
    
    print("[INFO] Pipeline execution complete. Ready for inference.")