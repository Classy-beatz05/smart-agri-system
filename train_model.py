"""
train_model.py — Multilingual Smart Agriculture Recommendation System
Trains crop & fertilizer models with full research output generation.

Usage:
    python train_model.py

Outputs:
    models/  — all .pkl files + meta.json
    research/ — confusion matrix, ROC, feature importance, heatmap, histogram, accuracy graph
"""

from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd
import numpy as np
import os
import json
import warnings
import time
warnings.filterwarnings("ignore")

matplotlib.use("Agg")


# ── Directories ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"
RESEARCH_DIR = "research"
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(RESEARCH_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = sorted([
    'Agra', 'Ahmedabad', 'Allahabad', 'Amritsar', 'Aurangabad', 'Bangalore', 'Bhopal',
    'Bhubaneswar', 'Chennai', 'Coimbatore', 'Dehradun', 'Delhi', 'Dharwad', 'Guwahati',
    'Gwalior', 'Hubli', 'Hyderabad', 'Indore', 'Jabalpur', 'Jaipur', 'Jammu', 'Jodhpur',
    'Kanpur', 'Kochi', 'Kolkata', 'Lucknow', 'Ludhiana', 'Madurai', 'Mangalore', 'Mumbai',
    'Mysore', 'Nagpur', 'Nashik', 'Patna', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Shimla',
    'Siliguri', 'Solapur', 'Srinagar', 'Surat', 'Thiruvananthapuram', 'Tiruchirappalli',
    'Udaipur', 'Vadodara', 'Varanasi', 'Vijayawada', 'Visakhapatnam'
])

KHARIF = ['rice', 'maize', 'cotton', 'jute',
          'pigeonpeas', 'mungbean', 'mothbeans', 'blackgram']
RABI = ['chickpea', 'lentil', 'kidneybeans']
ZAID = ['watermelon', 'muskmelon']

CROP_FERT = {
    'rice':       ('Urea', 'DAP'),        'maize':      ('Urea', 'NPK 10-26-26'),
    'cotton':     ('Urea', 'DAP'),        'jute':       ('Urea', 'SSP'),
    'banana':     ('Urea', 'MOP'),        'papaya':     ('Urea', 'MOP'),
    'watermelon': ('Urea', 'MOP'),        'muskmelon':  ('Urea', 'MOP'),
    'orange':     ('Urea', 'NPK 10-26-26'), 'chickpea': ('SSP', 'NPK 12-32-16'),
    'kidneybeans': ('SSP', 'NPK 12-32-16'), 'lentil':    ('SSP', 'NPK 12-32-16'),
    'mungbean':   ('SSP', 'DAP'),         'blackgram':  ('SSP', 'DAP'),
    'mothbeans':  ('SSP', 'MOP'),         'pigeonpeas': ('SSP', 'DAP'),
    'pomegranate': ('NPK 10-26-26', 'MOP'), 'mango':     ('NPK 10-26-26', 'MOP'),
    'grapes':     ('NPK 10-26-26', 'MOP'), 'apple':     ('NPK 10-26-26', 'MOP'),
    'coconut':    ('MOP', 'NPK 10-26-26'), 'coffee':    ('NPK 10-26-26', 'DAP'),
}

# ── Synthetic dataset generation (replace with real CSV if available) ─────────


def generate_dataset(n_samples=5000):
    """Generate realistic synthetic agricultural data."""
    print("📊 Generating synthetic dataset...")
    np.random.seed(42)

    CROP_PARAMS = {
        'rice':       dict(N=(60, 100), P=(30, 60),  K=(30, 60),  temp=(22, 35), hum=(70, 90), ph=(5.5, 7.0), rain=(150, 300), ec=(0.2, 0.8)),
        'maize':      dict(N=(60, 100), P=(30, 50),  K=(30, 50),  temp=(18, 30), hum=(55, 75), ph=(5.8, 7.5), rain=(60, 120),  ec=(0.2, 0.6)),
        'chickpea':   dict(N=(10, 30),  P=(40, 80),  K=(20, 50),  temp=(10, 25), hum=(40, 65), ph=(6.0, 8.0), rain=(30, 100),  ec=(0.1, 0.4)),
        'kidneybeans': dict(N=(15, 30),  P=(50, 80),  K=(15, 40),  temp=(12, 25), hum=(45, 70), ph=(6.0, 7.5), rain=(40, 120),  ec=(0.1, 0.4)),
        'pigeonpeas': dict(N=(10, 25),  P=(30, 60),  K=(25, 50),  temp=(25, 35), hum=(60, 80), ph=(5.5, 7.5), rain=(60, 150),  ec=(0.1, 0.5)),
        'mungbean':   dict(N=(10, 25),  P=(30, 50),  K=(20, 40),  temp=(25, 35), hum=(60, 80), ph=(6.0, 7.5), rain=(60, 120),  ec=(0.1, 0.4)),
        'blackgram':  dict(N=(20, 40),  P=(40, 70),  K=(20, 40),  temp=(25, 35), hum=(65, 85), ph=(5.5, 7.5), rain=(60, 150),  ec=(0.1, 0.5)),
        'lentil':     dict(N=(10, 25),  P=(30, 60),  K=(15, 35),  temp=(10, 25), hum=(40, 70), ph=(6.0, 8.0), rain=(25, 80),   ec=(0.1, 0.3)),
        'pomegranate': dict(N=(20, 40),  P=(40, 70),  K=(40, 80),  temp=(20, 35), hum=(40, 70), ph=(5.5, 7.5), rain=(50, 150),  ec=(0.2, 0.8)),
        'banana':     dict(N=(80, 120), P=(30, 50),  K=(80, 120), temp=(25, 35), hum=(70, 90), ph=(5.5, 7.0), rain=(100, 300), ec=(0.3, 1.0)),
        'mango':      dict(N=(20, 40),  P=(20, 40),  K=(30, 60),  temp=(25, 38), hum=(50, 80), ph=(5.5, 7.5), rain=(60, 200),  ec=(0.2, 0.7)),
        'grapes':     dict(N=(20, 40),  P=(30, 60),  K=(40, 80),  temp=(15, 35), hum=(50, 75), ph=(5.5, 7.0), rain=(50, 150),  ec=(0.2, 0.8)),
        'watermelon': dict(N=(80, 120), P=(30, 50),  K=(40, 70),  temp=(28, 38), hum=(60, 80), ph=(5.8, 7.5), rain=(40, 100),  ec=(0.2, 0.6)),
        'muskmelon':  dict(N=(70, 100), P=(30, 50),  K=(40, 70),  temp=(28, 38), hum=(55, 75), ph=(6.0, 7.5), rain=(35, 90),   ec=(0.2, 0.6)),
        'apple':      dict(N=(20, 40),  P=(30, 60),  K=(30, 60),  temp=(5, 20),  hum=(60, 80), ph=(5.5, 7.0), rain=(100, 200), ec=(0.1, 0.4)),
        'orange':     dict(N=(30, 60),  P=(20, 40),  K=(30, 60),  temp=(15, 30), hum=(55, 80), ph=(5.5, 7.5), rain=(60, 180),  ec=(0.2, 0.6)),
        'papaya':     dict(N=(50, 90),  P=(30, 50),  K=(40, 70),  temp=(25, 40), hum=(65, 85), ph=(5.5, 7.5), rain=(100, 200), ec=(0.2, 0.8)),
        'coconut':    dict(N=(30, 60),  P=(20, 40),  K=(60, 100), temp=(25, 38), hum=(70, 90), ph=(5.0, 8.0), rain=(150, 300), ec=(0.2, 0.8)),
        'cotton':     dict(N=(60, 100), P=(30, 60),  K=(30, 60),  temp=(25, 38), hum=(50, 75), ph=(5.8, 8.0), rain=(60, 150),  ec=(0.2, 0.8)),
        'jute':       dict(N=(60, 80),  P=(40, 70),  K=(30, 50),  temp=(25, 38), hum=(75, 90), ph=(6.0, 7.5), rain=(150, 300), ec=(0.2, 0.6)),
        'coffee':     dict(N=(80, 120), P=(40, 70),  K=(30, 60),  temp=(15, 28), hum=(65, 90), ph=(5.0, 6.5), rain=(150, 300), ec=(0.2, 0.6)),
        'mothbeans':  dict(N=(10, 25),  P=(25, 50),  K=(15, 35),  temp=(28, 40), hum=(40, 65), ph=(6.0, 8.0), rain=(20, 70),   ec=(0.1, 0.4)),
    }

    rows = []
    crops = list(CROP_PARAMS.keys())
    per_crop = n_samples // len(crops)

    for crop, p in CROP_PARAMS.items():
        n = per_crop
        def noise(arr, s): return arr + np.random.normal(0, s, n)
        N_vals = np.random.uniform(*p['N'],    n)
        P_vals = np.random.uniform(*p['P'],    n)
        K_vals = np.random.uniform(*p['K'],    n)
        t_vals = np.random.uniform(*p['temp'], n)
        h_vals = np.random.uniform(*p['hum'],  n)
        ph_vals = np.random.uniform(*p['ph'],   n)
        r_vals = np.random.uniform(*p['rain'], n)
        ec_vals = np.random.uniform(*p['ec'],   n)
        cities = np.random.choice(CITIES, n)

        if crop in KHARIF:
            months = np.random.choice([6, 7, 8, 9, 10], n)
        elif crop in RABI:
            months = np.random.choice([11, 12, 1, 2, 3], n)
        elif crop in ZAID:
            months = np.random.choice([3, 4, 5, 6], n)
        else:
            months = np.random.randint(1, 13, n)

        for i in range(n):
            rows.append({
                'N': round(N_vals[i], 2),   'P': round(P_vals[i], 2),
                'K': round(K_vals[i], 2),   'temperature': round(t_vals[i], 2),
                'humidity': round(h_vals[i], 2), 'ph': round(ph_vals[i], 2),
                'rainfall': round(r_vals[i], 2), 'ec': round(ec_vals[i], 3),
                'city': cities[i], 'month': int(months[i]), 'label': crop,
            })

    df = pd.DataFrame(rows).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"   ✅ {len(df)} rows | {df['label'].nunique()} crops")
    return df


# ── Research Plots ────────────────────────────────────────────────────────────
PLOT_STYLE = {
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4a',
    'text.color':       '#e8e8e8',
    'axes.labelcolor':  '#e8e8e8',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'grid.color':       '#2a2d3a',
    'grid.alpha':       0.4,
}


def save_confusion_matrix(y_true, y_pred, classes, fname):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d27')
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, linecolor='#0f1117',
                    annot_kws={'size': 7})
        ax.set_title('Confusion Matrix — Crop Prediction',
                     fontsize=14, color='#7dd87d', pad=15)
        ax.set_xlabel('Predicted Label', color='#a0b4c0')
        ax.set_ylabel('True Label', color='#a0b4c0')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


def save_roc_curve(y_test_bin, y_score, n_classes, class_names, fname):
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d27')
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        for i, (color, cname) in enumerate(zip(colors, class_names)):
            if y_test_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.2, alpha=0.8,
                    label=f'{cname} (AUC={roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], '--', color='#666', lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve — Multi-class Crop Prediction',
                     fontsize=13, color='#7dd87d')
        ax.legend(loc='lower right', fontsize=7, ncol=2,
                  facecolor='#1a1d27', edgecolor='#3a3d4a', labelcolor='#d0d0d0')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


def save_feature_importance(model, feature_names, fname):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d27')
        bars = ax.barh([feature_names[i] for i in idx[::-1]],
                       importances[idx[::-1]],
                       color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(idx))))
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title('Feature Importance — Random Forest',
                     fontsize=13, color='#7dd87d')
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, importances[idx[::-1]]):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8, color='#c0c0d0')
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


def save_heatmap(df, fname):
    cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'ec']
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d27')
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={'size': 10})
        ax.set_title('Feature Correlation Heatmap',
                     fontsize=13, color='#7dd87d')
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


def save_histograms(df, fname):
    cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'ec']
    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor('#0f1117')
        colors = ['#4fc3f7', '#81c784', '#ffb74d', '#f06292',
                  '#ba68c8', '#4db6ac', '#fff176', '#ff8a65']
        for ax, col, color in zip(axes.flat, cols, colors):
            ax.set_facecolor('#1a1d27')
            ax.hist(df[col], bins=40, color=color,
                    alpha=0.85, edgecolor='none')
            ax.set_title(col, fontsize=11, color='#e0e0e0')
            ax.set_xlabel('Value', fontsize=8, color='#a0a0b0')
            ax.set_ylabel('Frequency', fontsize=8, color='#a0a0b0')
            ax.grid(alpha=0.2)
        fig.suptitle('Feature Distribution Histograms',
                     fontsize=14, color='#7dd87d', y=1.01)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


def save_accuracy_graph(rf_scores, gb_scores, rf_acc, gb_acc, fname):
    with plt.style.context('dark_background'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0f1117')
        for ax in (ax1, ax2):
            ax.set_facecolor('#1a1d27')

        # Cross-validation scores
        folds = range(1, len(rf_scores)+1)
        ax1.plot(folds, rf_scores,  'o-', color='#4fc3f7',
                 lw=2, label='Random Forest', markersize=7)
        ax1.plot(folds, gb_scores, 's-', color='#ffb74d',
                 lw=2, label='Gradient Boost', markersize=7)
        ax1.axhline(rf_scores.mean(), color='#4fc3f7',
                    ls='--', alpha=0.5, lw=1)
        ax1.axhline(gb_scores.mean(), color='#ffb74d',
                    ls='--', alpha=0.5, lw=1)
        ax1.set_xlabel('CV Fold', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('5-Fold Cross-Validation Accuracy',
                      fontsize=12, color='#7dd87d')
        ax1.legend(facecolor='#1a1d27', edgecolor='#3a3d4a')
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0.85, 1.01])

        # Bar comparison
        models = ['Random Forest', 'Gradient Boost']
        accs = [rf_acc, gb_acc]
        bars = ax2.bar(models, accs, color=['#4fc3f7', '#ffb74d'], width=0.4,
                       edgecolor='none', zorder=3)
        ax2.set_ylim([0.9, 1.0])
        ax2.set_ylabel('Test Accuracy', fontsize=11)
        ax2.set_title('Model Comparison — Test Set Accuracy',
                      fontsize=12, color='#7dd87d')
        ax2.grid(axis='y', alpha=0.3, zorder=0)
        for bar, val in zip(bars, accs):
            ax2.text(bar.get_x()+bar.get_width()/2, val+0.001,
                     f'{val*100:.2f}%', ha='center', fontsize=12, color='#e0e0e0', fontweight='bold')

        plt.suptitle('Model Performance Analysis',
                     fontsize=14, color='#7dd87d', y=1.02)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    print(f"   📊 Saved {fname}")


# ── Main Training ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("  🌾 MULTILINGUAL SMART AGRICULTURE — MODEL TRAINING")
    print("=" * 60)

    # 1. Dataset
    df = generate_dataset(n_samples=5500)

    # 2. Feature engineering
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    le_city = LabelEncoder()
    df['city_enc'] = le_city.fit_transform(df['city'])

    le_crop = LabelEncoder()
    df['crop_enc'] = le_crop.fit_transform(df['label'])

    # Research: heatmap & histograms (before encoding)
    print("\n📈 Generating research plots...")
    save_heatmap(df, f"{RESEARCH_DIR}/heatmap.png")
    save_histograms(df, f"{RESEARCH_DIR}/histograms.png")

    # 3. Crop model features
    CROP_FEAT = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'ec',
                 'month_sin', 'month_cos', 'city_enc']
    X = df[CROP_FEAT].values
    y = df['label'].values
    classes = sorted(df['label'].unique())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    ms = MinMaxScaler()
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(ms.fit_transform(X_tr))
    X_te_s = sc.transform(ms.transform(X_te))

    # 4. Train & compare models
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te_s))
    print(f"   RF Accuracy: {rf_acc*100:.2f}%")

    print("🚀 Training Gradient Boosting (optimized)...")

    gb = GradientBoostingClassifier(
        n_estimators=25,      # reduced from 100
        max_depth=3,          # reduced from 5
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_tr_s, y_tr)
    gb_acc = accuracy_score(y_te, gb.predict(X_te_s))
    print(f"   GB Accuracy: {gb_acc*100:.2f}%")

    # Cross-validation
    print("🔁 Running 5-fold cross-validation...")
    rf_cv = cross_val_score(rf, X_tr_s, y_tr, cv=5, scoring='accuracy')
    gb_cv = cross_val_score(gb, X_tr_s, y_tr, cv=5, scoring='accuracy')
    print(f"   RF CV: {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")
    print(f"   GB CV: {gb_cv.mean()*100:.2f}% ± {gb_cv.std()*100:.2f}%")

    # Best model = RF (generally higher, plus faster predict)
    best_model = rf
    best_name = "Random Forest"
    print(f"\n🏆 Best model: {best_name}")

    # 5. Research outputs
    print("\n📊 Generating research visualizations...")
    save_confusion_matrix(y_te, best_model.predict(X_te_s), classes,
                          f"{RESEARCH_DIR}/confusion_matrix.png")

    y_te_bin = label_binarize(y_te, classes=classes)
    y_score = best_model.predict_proba(X_te_s)
    save_roc_curve(y_te_bin, y_score, len(classes), classes,
                   f"{RESEARCH_DIR}/roc_curve.png")

    save_feature_importance(best_model, CROP_FEAT,
                            f"{RESEARCH_DIR}/feature_importance.png")

    save_accuracy_graph(rf_cv, gb_cv, rf_acc, gb_acc,
                        f"{RESEARCH_DIR}/accuracy_graph.png")

    print("\n📋 Classification Report:")
    print(classification_report(y_te, best_model.predict(X_te_s), zero_division=0))

    # 6. Fertilizer model
    print("🧪 Training Fertilizer Model...")

    def assign_fert(row):
        f_low, f_high = CROP_FERT[row['label']]
        if row['N'] < 40:
            return f_low
        elif row['P'] < 30:
            return 'DAP'
        elif row['K'] < 20:
            return 'MOP'
        else:
            return f_high

    df['fertilizer'] = df.apply(assign_fert, axis=1)

    FERT_FEAT = ['N', 'P', 'K', 'temperature',
                 'humidity', 'ph', 'rainfall', 'crop_enc']
    Xf = df[FERT_FEAT].values
    yf = df['fertilizer'].values

    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
        Xf, yf, test_size=0.2, random_state=42, stratify=yf)

    mf = MinMaxScaler()
    sf = StandardScaler()
    Xf_tr_s = sf.fit_transform(mf.fit_transform(Xf_tr))
    Xf_te_s = sf.transform(mf.transform(Xf_te))

    fert_model = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1)
    fert_model.fit(Xf_tr_s, yf_tr)
    fert_acc = accuracy_score(yf_te, fert_model.predict(Xf_te_s))
    print(f"   Fertilizer Accuracy: {fert_acc*100:.2f}%")

    # 7. Save models
    print("\n💾 Saving models...")
    saves = {
        'crop_model': best_model,   'crop_ms': ms,    'crop_sc': sc,
        'fert_model': fert_model,   'fert_ms': mf,    'fert_sc': sf,
        'le_city':    le_city,      'le_crop': le_crop,
    }
    for name, obj in saves.items():
        p = f"{MODEL_DIR}/{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        print(f"   ✅ {p}")

    meta = {
        'crop_accuracy':  round(rf_acc*100, 2),
        'fert_accuracy':  round(fert_acc*100, 2),
        'cv_accuracy':    round(rf_cv.mean()*100, 2),
        'cv_std':         round(rf_cv.std()*100, 2),
        'gb_accuracy':    round(gb_acc*100, 2),
        'best_model':     best_name,
        'n_crops':        int(df['label'].nunique()),
        'n_cities':       len(le_city.classes_),
        'cities':         sorted(CITIES),
        'crops':          sorted(le_crop.classes_.tolist()),
        'fertilizers':    sorted(fert_model.classes_.tolist()),
        'crop_features':  CROP_FEAT,
        'fert_features':  FERT_FEAT,
    }
    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   ✅ {MODEL_DIR}/meta.json")

    elapsed = time.time() - t0
    print(f"\n🎉 Training complete in {elapsed:.1f}s")
    print(f"   Crop model  : {rf_acc*100:.2f}%  (CV: {rf_cv.mean()*100:.2f}%)")
    print(f"   Fert model  : {fert_acc*100:.2f}%")
    print(f"   Research     : {RESEARCH_DIR}/")
    print(f"   Models       : {MODEL_DIR}/")


if __name__ == "__main__":
    main()
