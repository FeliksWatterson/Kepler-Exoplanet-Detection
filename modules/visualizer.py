import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
    # Lấy trọng số đặc trưng tùy theo loại kiến trúc thuật toán
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0]) # Lấy giá trị tuyệt đối của trọng số
    else:
        print("[WARNING] Model does not support feature importance visualization.")
        return

    idx = np.argsort(imp)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp[idx], y=np.array(names)[idx], palette='magma')
    plt.title('Feature Importance Analysis')
    plt.xlabel('Relative Importance / Absolute Coefficient')
    plt.savefig(os.path.join(out_dir, 'feat_imp.png'), dpi=300)
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

def plot_target_dist(df, out_dir):
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x='koi_disposition', palette='Set2')
    plt.title('Target Variable Distribution (Class Imbalance Check)')
    plt.ylabel('Count')
    plt.xlabel('Disposition')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
    plt.savefig(os.path.join(out_dir, 'eda_1_target_dist.png'), dpi=300)
    plt.close()

def plot_feature_dist(df, features, out_dir):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        sns.histplot(data=df, x=feat, hue='koi_disposition', kde=True, 
                     ax=axes[i], palette='Set1', element='step', bins=30)
        axes[i].set_title(f'Distribution of {feat}')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eda_2_feature_dist.png'), dpi=300)
    plt.close()

def plot_boxplots(df, features, out_dir):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        sns.boxplot(data=df, x='koi_disposition', y=feat, ax=axes[i], palette='pastel')
        axes[i].set_title(f'Boxplot of {feat}')
        
        if df[feat].min() > 0: 
            axes[i].set_yscale('log')
            axes[i].set_ylabel(f'{feat} (Log Scale)')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eda_3_boxplots.png'), dpi=300)
    plt.close()

def plot_missing_values(df, features, out_dir):
    missing_pct = df[features].isnull().sum() / len(df) * 100
    
    plt.figure(figsize=(8, 6))
    if missing_pct.sum() > 0:
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='Reds_r')
        plt.title('Percentage of Missing Values per Feature')
        plt.xlabel('% Missing')
    else:
        plt.text(0.5, 0.5, 'No Missing Values Detected', 
                 ha='center', va='center', fontsize=15, color='green')
        plt.title('Missing Values Analysis')
        plt.axis('off')
        
    plt.savefig(os.path.join(out_dir, 'eda_4_missing_values.png'), dpi=300)
    plt.close()

