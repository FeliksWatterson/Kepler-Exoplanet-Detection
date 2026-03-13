import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def run_models(X_train, X_test, y_train, y_test, out_dir):
    models_and_params = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            {'C': [0.1, 1.0, 10.0]}
        ),
        'SVM_Linear': (
            SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
            {'C': [0.1, 1.0, 10.0]}
        ),
        'RandomForest': (
            RandomForestClassifier(class_weight='balanced', random_state=42),
            {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    plt.figure(figsize=(10, 8))
    
    print("[INFO] Initiating model training and hyperparameter optimization phase...")
    best_overall_model = None
    best_overall_metric = 0.0
    best_model_name = ""

    for name, (model, params) in models_and_params.items():
        print(f"\n[INFO] Evaluating candidate architecture: {name}")
        
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=params, 
            cv=cv, 
            scoring='f1', 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"[RESULT] Optimal hyperparameters for {name}: {grid_search.best_params_}")
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Đánh giá dựa trên Accuracy (để biểu đồ)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print("[METRIC] Detailed Classification Report:")
        # Trích xuất toàn bộ report dưới dạng dict để lấy F1-score của lớp CONFIRMED (1)
        report_dict = classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED'], output_dict=True)
        f1_score_confirmed = report_dict['CONFIRMED']['f1-score']
        
        print(classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED'], digits=4))
        
        # Identify globally optimal model dựa trên F1-score của lớp thiểu số
        if f1_score_confirmed > best_overall_metric:
            best_overall_metric = f1_score_confirmed
            best_overall_model = best_model
            best_model_name = name

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    print(f"\n[SUMMARY] Globally optimal model selected: {best_model_name} (F1-Score CONFIRMED: {best_overall_metric:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Analysis')
    plt.legend(loc='lower right')
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight') 
    plt.close()
    
    return results, best_overall_model