import numpy as np
import pandas as pd
from modules.data_loader import load_data
from modules.trainer import run_models
from sklearn.model_selection import train_test_split

# Config
DATA_FILE = 'dataset/cumulative.csv'
OUT_DIR = 'outputs'

def print_result(sample_data, prediction, actual, proba):
    print("-" * 50)
    print(f"INPUT SIGNAL:")
    print(f" - Period: {sample_data[0]:.2f} days") 
    print(f" - Radius: {sample_data[5]:.2f} R_earth") 
    print(f" - Depth : {sample_data[4]:.2f} ppm") 
    print(f" - Temp  : {sample_data[6]:.2f} K")
    
    pred_text = "CONFIRMED PLANET" if prediction == 1 else "FALSE POSITIVE"
    act_text = "CONFIRMED PLANET" if actual == 1 else "FALSE POSITIVE"
    
    print(f"\nModel Prediction : [{pred_text}] ({proba:.1f}%)")
    print(f"Actual Truth : [{act_text}]")
    
    if prediction == actual:
        print("Verdict: Correct")
    else:
        print("Verdict: Incorrect")
    print("-" * 50)

if __name__ == "__main__":
    # 1. Setup
    print("Initializing System & Loading Models...")
    X, y, feats, df, scaler = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, model = run_models(X_train, X_test, y_train, y_test, OUT_DIR)

    # 2. Pick random samples for testing
    print("\nPicking random targets...\n")
    
    # Pick 5 random indices from Test set
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i in random_indices:
        original_idx = y_test.index[i]
        raw_sample = df.loc[original_idx, feats].values
        
        # Get processed data for model
        input_vector = X_test[i].reshape(1, -1)
        actual_label = y_test.iloc[i]
        
        # Predict
        pred_label = model.predict(input_vector)[0]
        pred_prob = model.predict_proba(input_vector)[0][pred_label] * 100
        
        # Show result
        print_result(raw_sample, pred_label, actual_label, pred_prob)
        
        input("Press Enter to scan next target...")