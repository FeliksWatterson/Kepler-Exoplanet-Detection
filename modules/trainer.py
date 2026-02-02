import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

def run_models(X_train, X_test, y_train, y_test, out_dir):
    # Init models
    models = {
        'LogReg': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    
    # ROC Plot setup
    plt.figure(figsize=(10, 8))
    
    print("Training models...")
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Score
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f" - {name}: {acc*100:.2f}%")
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

    # Save ROC
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()
    
    return results, models['RandomForest']