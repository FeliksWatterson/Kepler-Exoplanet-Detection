import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_corr(df, features, out_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(out_dir, 'corr_matrix.png'))
    plt.close()

def plot_acc(results, out_dir):
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=values, y=names, palette='viridis')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.savefig(os.path.join(out_dir, 'model_acc.png'))
    plt.close()

def plot_features(model, names, out_dir):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp[idx], y=np.array(names)[idx], palette='magma')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(out_dir, 'feat_imp.png'))
    plt.close()

def plot_cm(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FALSE', 'CONFIRMED'],
                yticklabels=['FALSE', 'CONFIRMED'])
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()