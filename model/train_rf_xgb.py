# model/train_rf_xgb.py
# Trains and evaluates Random Forest and XGBoost models to predict which
# compiler transformation produces the fastest runtime for a given C function.
#
# Multi-class problem (4 classes):
#   0 = original is fastest
#   1 = unrolled is fastest
#   2 = inlined is fastest
#   3 = unrolled_inlined is fastest
#
# Validation strategy:
#   - Stratified 80/20 train/test split (preserves class distribution)
#   - Stratified 5-fold cross-validation on training set
#   - Training F1 vs CV F1 gap to detect overfitting
#   - Majority-class baseline to detect if model is just predicting majority
#   - Macro F1 and per-class F1 to check all classes are learned
#   - Learning curves to detect underfitting vs overfitting
#
# GPU note (AMD RX 9060 XT):
#   CPU is faster at this dataset size due to transfer overhead.
#   Set GPU_DEVICE = "cuda" once ROCm is installed and dataset grows.
#
# Output:
#   Console: full validation report for both models
#   model/random_forest.pkl
#   model/xgboost.pkl
#   model/learning_curves.png

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import csv
import math
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve,
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.dummy   import DummyClassifier
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
MODEL_DIR    = os.path.dirname(__file__)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

GPU_DEVICE   = "cpu"   # set to "cuda" for AMD GPU with ROCm

# Class labels for display.
LABEL_NAMES = {0: "original", 1: "unrolled", 2: "inlined", 3: "unrolled_inlined"}
N_CLASSES   = 4

FEATURE_COLUMNS = [
    # Phase 1 — loop structure
    "loop_count",
    "total_loop_count",
    "max_nesting_depth",
    "max_iteration_count",
    "avg_iteration_count",
    "total_iter_product",       # log-transformed below
    # Phase 2 — body level
    "has_reduction",
    "array_reads_per_iter",
    "array_writes_per_iter",
    "has_multiply",
    "total_body_stmts",
    # Phase 2 — inlining
    "has_function_call",
    "num_helpers",
    "helper_body_ops",
    "call_count",
]
LABEL_COLUMN = "best_transformation"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset.csv, apply log1p to total_iter_product, return (X, y, names)."""
    rows = list(csv.DictReader(open(path)))
    X, y = [], []
    for row in rows:
        features = []
        for col in FEATURE_COLUMNS:
            val = float(row[col])
            if col == "total_iter_product":
                val = math.log1p(val)
            features.append(val)
        X.append(features)
        y.append(int(row[LABEL_COLUMN]))

    feature_names = [
        "log_total_iter_product" if c == "total_iter_product" else c
        for c in FEATURE_COLUMNS
    ]
    return np.array(X), np.array(y), feature_names


# ── Model builders ────────────────────────────────────────────────────────────

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 10,   # limited to prevent memorisation
        min_samples_leaf = 5,
        class_weight     = "balanced",   # compensates for class imbalance
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )


def build_xgboost(n_classes: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "multi:softmax",  # multi-class classification
        num_class        = n_classes,
        eval_metric      = "mlogloss",       # multi-class log loss
        random_state     = RANDOM_STATE,
        device           = GPU_DEVICE,
        verbosity        = 0,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    name:          str,
    model,
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    y_train:       np.ndarray,
    y_test:        np.ndarray,
    feature_names: list[str],
    cv:            StratifiedKFold,
) -> dict:
    """Full multi-class evaluation: overfitting check, per-class F1, importances."""

    y_pred = model.predict(X_test)

    # ── Overfitting check ─────────────────────────────────────────────────────
    train_f1 = f1_score(y_train, model.predict(X_train), average="macro")
    cv_f1s   = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="f1_macro"
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    test_acc    = accuracy_score(y_test, y_pred)
    test_f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    test_f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    gap     = train_f1 - cv_f1s.mean()
    verdict = "OK" if gap < 0.05 else "POSSIBLE OVERFIT"
    print(f"\n  Bias-Variance check (macro F1):")
    print(f"    Training F1:         {train_f1:.3f}")
    print(f"    CV F1 ({CV_FOLDS}-fold):      {cv_f1s.mean():.3f} ± {cv_f1s.std():.3f}")
    print(f"    Gap (train - CV):    {gap:.3f}  [{verdict}]")

    print(f"\n  Test metrics:")
    print(f"    Accuracy:            {test_acc:.3f}")
    print(f"    Macro F1:            {test_f1_mac:.3f}  (each class weighted equally)")
    print(f"    Weighted F1:         {test_f1_wt:.3f}  (weighted by class frequency)")

    # Per-class F1 — shows which transformation classes the model struggles with.
    print(f"\n  Per-class F1:")
    per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, score in enumerate(per_class):
        label_name = LABEL_NAMES.get(i, str(i))
        bar        = "█" * int(score * 30)
        count      = int((y_test == i).sum())
        print(f"    {i} {label_name:<20} F1={score:.3f}  n={count:<5} {bar}")

    # Confusion matrix.
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    cm     = confusion_matrix(y_test, y_pred)
    header = "         " + "  ".join(f"{LABEL_NAMES[i][:6]:>6}" for i in range(N_CLASSES))
    print(f"    {header}")
    for i in range(N_CLASSES):
        row_label = f"{LABEL_NAMES[i][:6]:>6}"
        row_vals  = "  ".join(f"{cm[i,j]:>6}" for j in range(N_CLASSES))
        print(f"    {row_label}  {row_vals}")

    # Feature importances.
    print(f"\n  Feature importances (descending):")
    ranked = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<32} {imp:.3f}  {bar}")

    return {
        "name":    name,
        "model":   model,
        "cv_f1":   cv_f1s.mean(),
        "test_f1": test_f1_mac,
        "acc":     test_acc,
    }


# ── Learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(
    models:      list[tuple[str, object]],
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    cv:          StratifiedKFold,
    output_path: str,
) -> None:
    """Plot training vs CV macro-F1 as training set size increases."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    train_sizes = np.linspace(0.1, 1.0, 10)

    for ax, (name, model) in zip(axes, models):
        sizes, train_scores, cv_scores = learning_curve(
            model, X_train, y_train,
            train_sizes = train_sizes,
            cv          = cv,
            scoring     = "f1_macro",
            n_jobs      = -1,
        )
        ax.plot(sizes, train_scores.mean(axis=1), label="Training macro-F1", color="steelblue")
        ax.fill_between(sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.1, color="steelblue")
        ax.plot(sizes, cv_scores.mean(axis=1), label="CV macro-F1", color="darkorange")
        ax.fill_between(sizes,
            cv_scores.mean(axis=1) - cv_scores.std(axis=1),
            cv_scores.mean(axis=1) + cv_scores.std(axis=1),
            alpha=0.1, color="darkorange")
        ax.set_title(name)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Learning Curves — compiler-opt (multi-class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nLearning curves saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading dataset from {DATASET_PATH}...")
    X, y, feature_names = load_dataset(DATASET_PATH)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    # Print class distribution.
    print(f"  Class distribution:")
    for i in range(N_CLASSES):
        n = int((y == i).sum())
        print(f"    {i} {LABEL_NAMES[i]:<20} {n:>5} ({100*n/len(y):.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Majority-class baseline ───────────────────────────────────────────────
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
    baseline_f1  = f1_score(y_test, dummy.predict(X_test), average="macro", zero_division=0)
    print(f"\nMajority-class baseline (always predict most frequent class):")
    print(f"  Accuracy: {baseline_acc:.3f}   Macro F1: {baseline_f1:.3f}")
    print(f"  Our models must beat these to be doing anything useful.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = build_xgboost(N_CLASSES)
    xgb_model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rf_results  = evaluate("Random Forest", rf,        X_train, X_test, y_train, y_test, feature_names, cv)
    xgb_results = evaluate("XGBoost",       xgb_model, X_train, X_test, y_train, y_test, feature_names, cv)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'CV Macro F1':>12} {'Test Macro F1':>14} {'Accuracy':>9}")
    print(f"  {'-'*62}")
    print(f"  {'Baseline':<25} {'N/A':>12} {baseline_f1:>14.3f} {baseline_acc:>9.3f}")
    for r in [rf_results, xgb_results]:
        print(f"  {r['name']:<25} {r['cv_f1']:>12.3f} {r['test_f1']:>14.3f} {r['acc']:>9.3f}")

    # ── Learning curves ───────────────────────────────────────────────────────
    curves_path = os.path.join(MODEL_DIR, "learning_curves.png")
    plot_learning_curves(
        [("Random Forest", rf), ("XGBoost", xgb_model)],
        X_train, y_train, cv, curves_path,
    )

    # ── Save models ───────────────────────────────────────────────────────────
    for filename, model in [("random_forest.pkl", rf), ("xgboost.pkl", xgb_model)]:
        path = os.path.join(MODEL_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
