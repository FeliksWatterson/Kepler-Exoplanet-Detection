import os
import joblib
import numpy as np
import pandas as pd
from modules.data_loader import load_data
from sklearn.model_selection import train_test_split

# Config
DATA_FILE = 'dataset/cumulative.csv'
OUT_DIR = 'outputs'

def print_result(sample_data, prediction, actual, proba):
    print("-" * 60)
    print(f"[INFERENCE] Analyzing Kepler Object of Interest...")
    print(f"  |- Orbital Period : {sample_data[0]:.4f} days")
    print(f"  |- Transit Depth  : {sample_data[4]:.4f} ppm")
    print(f"  |- Planetary Rad  : {sample_data[5]:.4f} Earth radii")
    print(f"  |- Equil. Temp    : {sample_data[6]:.2f} K")
    
    pred_text = "CONFIRMED" if prediction == 1 else "FALSE POSITIVE"
    act_text = "CONFIRMED" if actual == 1 else "FALSE POSITIVE"
    
    print(f"\n[OUTPUT] Model Prediction : {pred_text} (Confidence: {proba/100:.4f})")
    print(f"[GROUND TRUTH] Actual     : {act_text}")
    
    if prediction == actual:
        print("[EVALUATION] Status: Correct Match")
    else:
        print("[EVALUATION] Status: Misclassification")
    print("-" * 60)

if __name__ == "__main__":
    print("[INFO] Initializing Inference Engine...")
    if not os.path.exists(os.path.join(OUT_DIR, 'best_model.pkl')):
        print("[ERROR] Model artifacts not found! Vui lòng chạy 'python main.py' trước để train và xuất model.")
        exit(1)
    print("[INFO] Loading pre-trained artifacts (.pkl)...")
    imputer = joblib.load(os.path.join(OUT_DIR, 'imputer.pkl'))
    scaler = joblib.load(os.path.join(OUT_DIR, 'scaler.pkl'))
    model = joblib.load(os.path.join(OUT_DIR, 'best_model.pkl'))

    print("[INFO] Loading dataset for sampling...")
    X_raw, y, feats, df = load_data(DATA_FILE)
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
    print("\n[SIMULATION STARTED] Scanning for random targets (UNSEEN DATA ONLY)...\n")
    
    # Chọn ngẫu nhiên các index nằm trong tập test
    random_indices = np.random.choice(len(X_test_raw), 5, replace=False)
    
    for i in random_indices:
        raw_sample = X_test_raw.iloc[i].values
        actual_label = y_test.iloc[i]
        
        input_vector = raw_sample.reshape(1, -1)
        input_imputed = imputer.transform(input_vector)
        input_scaled = scaler.transform(input_imputed)
        
        pred_label = model.predict(input_scaled)[0]
        pred_prob = model.predict_proba(input_scaled)[0][pred_label] * 100
        
        print_result(raw_sample, pred_label, actual_label, pred_prob)
        
        input("Press Enter to scan next target...")