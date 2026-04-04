# ============================================================
#  FINAL PROJECT - Bank Note Authentication (Binary Classification)
#  Dataset: banknote_authentication.csv
#  - 1372 samples, no header, no missing values
#  - Features: variance, skewness, curtosis, entropy (float64)
#  - Label: class  →  0 = genuine (762), 1 = forged (610)
#  Author: [Your Name] - [Student ID]
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, StackingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_SEED  = 42
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]
COLORS        = {0: "#4C72B0", 1: "#DD8452"}   # blue=genuine, orange=forged

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 65)
print("STEP 1: LOAD DATA")
print("=" * 65)

# File KHÔNG có header → tự đặt tên cột
df = pd.read_csv(
    "banknote_authentication.csv",
    header=None,
    names=FEATURE_NAMES + ["class"]
)

print(f"Shape   : {df.shape}")          # (1372, 5)
print(f"Columns : {list(df.columns)}")
print(f"\n{df.head()}")

# ============================================================
# STEP 2: INITIAL DATA ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("STEP 2: INITIAL DATA ANALYSIS")
print("=" * 65)

print("\n--- Missing Values ---")
print(df.isnull().sum())
print("→ No missing values.\n")

vc = df["class"].value_counts().sort_index()
print("--- Class Distribution ---")
print(f"  Class 0 (Genuine): {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  Class 1 (Forged) : {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")
print("→ Dataset fairly balanced (55.5% vs 44.5%)\n")

print("--- Descriptive Statistics ---")
print(df.describe().round(3))

# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: EDA")
print("=" * 65)

# 3a. Feature distributions by class
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold")
for i, feat in enumerate(FEATURE_NAMES):
    ax = axes[i // 2][i % 2]
    for cls in [0, 1]:
        ax.hist(df[df["class"] == cls][feat], bins=35, alpha=0.6,
                color=COLORS[cls], label=f"Class {cls}",
                edgecolor="white", linewidth=0.3)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Value"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.show()
print("[Saved] eda_distributions.png")

# 3b. Boxplot
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle("Boxplot — Features by Class", fontsize=13, fontweight="bold")
for i, feat in enumerate(FEATURE_NAMES):
    bp = axes[i].boxplot(
        [df[df["class"] == 0][feat], df[df["class"] == 1][feat]],
        patch_artist=True, labels=["Genuine", "Forged"]
    )
    bp["boxes"][0].set_facecolor(COLORS[0])
    bp["boxes"][1].set_facecolor(COLORS[1])
    axes[i].set_title(feat, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_boxplot.png", dpi=150)
plt.show()
print("[Saved] eda_boxplot.png")

# 3c. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()
print("[Saved] eda_correlation.png")

# 3d. Pairplot
g = sns.pairplot(df, hue="class", palette=COLORS,
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
g.figure.suptitle("Pairplot — All Features", y=1.02,
                   fontsize=13, fontweight="bold")
g.figure.savefig("eda_pairplot.png", dpi=150)
plt.show()
print("[Saved] eda_pairplot.png")

# ============================================================
# STEP 4: PREPROCESS DATA
# ============================================================
print("\n" + "=" * 65)
print("STEP 4: PREPROCESS DATA")
print("=" * 65)

X = df[FEATURE_NAMES].values
y = df["class"].values

# 70 / 30 split, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train class → 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")
print(f"Test  class → 0:{(y_test==0).sum()}   1:{(y_test==1).sum()}")

# StandardScaler — fit on train ONLY
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("✓ StandardScaler applied.")

# ============================================================
# STEP 5: BASELINE — 10 CLASSIFIERS
# ============================================================
print("\n" + "=" * 65)
print("STEP 5: 10 BASELINE CLASSIFIERS")
print("=" * 65)

classifiers = {
    "kNN":                 KNeighborsClassifier(),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(probability=True, random_state=RANDOM_SEED),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                  random_state=RANDOM_SEED),
    "AdaBoost":            AdaBoostClassifier(random_state=RANDOM_SEED),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=RANDOM_SEED),
    "LDA":                 LinearDiscriminantAnalysis(),
    "MLP":                 MLPClassifier(max_iter=500, random_state=RANDOM_SEED),
    "Logistic Regression": LogisticRegression(max_iter=500,
                                              random_state=RANDOM_SEED),
}

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
results = []

for name, clf in classifiers.items():
    t0     = time.time()
    cv_acc = cross_val_score(clf, X_train_sc, y_train, cv=cv,
                             scoring="accuracy")
    clf.fit(X_train_sc, y_train)
    elapsed = time.time() - t0

    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    results.append({
        "Model":         name,
        "CV Acc (mean)": round(cv_acc.mean(), 4),
        "CV Acc (std)":  round(cv_acc.std(),  4),
        "Accuracy":      round(accuracy_score(y_test, y_pred),           4),
        "Precision":     round(precision_score(y_test, y_pred,
                                               zero_division=0),         4),
        "Recall":        round(recall_score(y_test, y_pred,
                                            zero_division=0),            4),
        "F1-Score":      round(f1_score(y_test, y_pred,
                                        zero_division=0),                4),
        "AUC":           round(roc_auc_score(y_test, y_prob),            4),
        "Time (s)":      round(elapsed, 3),
    })
    print(f"  [{name:22s}]  Acc={results[-1]['Accuracy']:.4f}"
          f"  F1={results[-1]['F1-Score']:.4f}"
          f"  AUC={results[-1]['AUC']:.4f}"
          f"  t={elapsed:.2f}s")

results_df = pd.DataFrame(results).set_index("Model")
print("\n--- BASELINE RESULTS TABLE ---")
print(results_df.to_string())
results_df.to_csv("baseline_results.csv")
print("[Saved] baseline_results.csv")

# Top-2 per metric
print("\n--- TOP 2 MODELS PER METRIC ---")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
    top2 = results_df[metric].nlargest(2)
    print(f"  {metric:12s}: 1. {top2.index[0]} ({top2.iloc[0]:.4f})"
          f"  |  2. {top2.index[1]} ({top2.iloc[1]:.4f})")

# Bar chart
metrics_bar = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle("Baseline Model Comparison", fontsize=13, fontweight="bold")
for i, m in enumerate(metrics_bar):
    sv = results_df[m].sort_values()
    bars = axes[i].barh(sv.index, sv.values, color="#4C72B0", edgecolor="white")
    axes[i].set_title(m, fontweight="bold")
    axes[i].set_xlim(sv.min() - 0.02, 1.02)
    for bar, val in zip(bars, sv.values):
        axes[i].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("[Saved] model_comparison.png")

# ============================================================
# STEP 6: CONFUSION MATRIX & ROC CURVE
# ============================================================
print("\n" + "=" * 65)
print("STEP 6: CONFUSION MATRIX & ROC CURVE")
print("=" * 65)

top3 = results_df["F1-Score"].nlargest(3).index.tolist()
print(f"Top 3 by F1: {top3}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Confusion Matrices — Top 3 Models", fontsize=13, fontweight="bold")
for ax, name in zip(axes, top3):
    cm = confusion_matrix(y_test, classifiers[name].predict(X_test_sc))
    ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Forged"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("[Saved] confusion_matrices.png")

plt.figure(figsize=(8, 6))
for name in top3:
    y_prob = classifiers[name].predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2,
             label=f"{name}  (AUC={roc_auc_score(y_test, y_prob):.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Top 3 Models", fontsize=13, fontweight="bold")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150); plt.show()
print("[Saved] roc_curve.png")

# ============================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 65)
print("STEP 7: HYPERPARAMETER TUNING (GridSearchCV, 5-fold)")
print("=" * 65)

tuned = {}

def tune(name, base_clf, param_grid):
    print(f"\n[Tuning] {name} ...")
    gs = GridSearchCV(base_clf, param_grid, cv=5,
                      scoring="f1", n_jobs=-1)
    gs.fit(X_train_sc, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")
    tuned[f"{name} (Tuned)"] = gs.best_estimator_

tune("RF", RandomForestClassifier(random_state=RANDOM_SEED),
     {"n_estimators": [100, 200, 300],
      "max_depth": [None, 5, 10],
      "min_samples_split": [2, 5]})

tune("SVM", SVC(probability=True, random_state=RANDOM_SEED),
     {"C": [0.1, 1, 10, 100],
      "kernel": ["rbf", "linear"],
      "gamma": ["scale", "auto"]})

tune("GB", GradientBoostingClassifier(random_state=RANDOM_SEED),
     {"n_estimators": [100, 200],
      "learning_rate": [0.05, 0.1, 0.2],
      "max_depth": [3, 5]})

tune("kNN", KNeighborsClassifier(),
     {"n_neighbors": [3, 5, 7, 11, 15],
      "weights": ["uniform", "distance"],
      "metric": ["euclidean", "manhattan"]})

tune("MLP", MLPClassifier(max_iter=500, random_state=RANDOM_SEED),
     {"hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
      "activation": ["relu", "tanh"],
      "learning_rate_init": [0.001, 0.01]})

# ============================================================
# STEP 8: STACKING ENSEMBLE
# ============================================================
print("\n" + "=" * 65)
print("STEP 8: STACKING ENSEMBLE")
print("=" * 65)

stacking = StackingClassifier(
    estimators=[
        ("rf",  tuned["RF (Tuned)"]),
        ("svm", tuned["SVM (Tuned)"]),
        ("gb",  tuned["GB (Tuned)"]),
        ("knn", tuned["kNN (Tuned)"]),
    ],
    final_estimator=LogisticRegression(max_iter=500),
    cv=5,
)
stacking.fit(X_train_sc, y_train)
tuned["Stacking Ensemble"] = stacking
print("✓ Stacking Ensemble fitted.")

# ============================================================
# STEP 9: FINAL COMPARISON
# ============================================================
print("\n" + "=" * 65)
print("STEP 9: FINAL COMPARISON")
print("=" * 65)

def eval_clf(name, clf):
    yp   = clf.predict(X_test_sc)
    yprb = clf.predict_proba(X_test_sc)[:, 1]
    return {"Model":     name,
            "Accuracy":  round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp,  zero_division=0), 4),
            "Recall":    round(recall_score(y_test, yp,     zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, yp,         zero_division=0), 4),
            "AUC":       round(roc_auc_score(y_test, yprb), 4)}

tuned_df = pd.DataFrame([eval_clf(n, c) for n, c in tuned.items()]).set_index("Model")
print(tuned_df.to_string())
tuned_df.to_csv("tuned_results.csv")
print("[Saved] tuned_results.csv")

# Final F1 bar chart (baseline + tuned)
all_f1 = pd.concat([results_df["F1-Score"],
                    tuned_df["F1-Score"]]).sort_values()
bar_colors = ["#DD8452" if "(Tuned)" in n or "Stacking" in n
              else "#4C72B0" for n in all_f1.index]
plt.figure(figsize=(10, 7))
bars = plt.barh(all_f1.index, all_f1.values,
                color=bar_colors, edgecolor="white")
for bar, val in zip(bars, all_f1.values):
    plt.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=8)
plt.xlabel("F1-Score")
plt.title("F1 Comparison — Blue=Baseline  Orange=Tuned/Stacking",
          fontsize=11, fontweight="bold")
plt.xlim(all_f1.min() - 0.03, 1.02)
plt.tight_layout()
plt.savefig("final_comparison_f1.png", dpi=150)
plt.show()
print("[Saved] final_comparison_f1.png")

print("\n" + "=" * 65)
print("  ALL DONE!")
print("  baseline_results.csv    — bảng 10 baseline models")
print("  tuned_results.csv       — kết quả tuned & stacking")
print("  eda_distributions.png   — phân phối features")
print("  eda_boxplot.png         — boxplot")
print("  eda_correlation.png     — heatmap tương quan")
print("  eda_pairplot.png        — pairplot")
print("  model_comparison.png    — so sánh 10 models")
print("  confusion_matrices.png  — top 3 models")
print("  roc_curve.png           — ROC curve top 3")
print("  final_comparison_f1.png — F1 tất cả models")
print("=" * 65)