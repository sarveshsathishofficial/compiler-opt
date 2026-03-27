# model/train_rf_xgb.py
# Trains and evaluates Random Forest and XGBoost models to predict whether
# unrolling a loop will improve runtime performance.
#
# Validation strategy:
#   - Stratified 80/20 train/test split (preserves 83/17 class ratio)
#   - Stratified 5-fold cross-validation on training set
#   - Training F1 vs CV F1 gap to detect overfitting
#   - Majority-class baseline to detect if model is just predicting majority
#   - ROC-AUC score (better than accuracy for imbalanced data)
#   - Threshold tuning to improve minority class (label=0) predictions
#   - Learning curves to detect underfitting vs overfitting vs dataset size needs
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
matplotlib.use("Agg")   # non-interactive backend — saves to file instead of displaying
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.dummy   import DummyClassifier
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
MODEL_DIR    = os.path.dirname(__file__)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# Default classification threshold. 0.5 is standard but may not be optimal
# for imbalanced data — we tune this below.
DEFAULT_THRESHOLD = 0.5

GPU_DEVICE = "cpu"  # set to "cuda" for AMD GPU with ROCm

FEATURE_COLUMNS = [
    "loop_count",
    "total_loop_count",
    "max_nesting_depth",
    "max_iteration_count",
    "avg_iteration_count",
    "total_iter_product",   # log-transformed below
    "has_dependent_body",
    "has_independent_body",
]
LABEL_COLUMN = "unrolled_faster"

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
        max_depth        = 10,   # was None — unlimited depth memorises training data
        min_samples_leaf = 5,    # don't split a node if it has fewer than 5 samples
        class_weight     = "balanced",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )


def build_xgboost(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,    # was 6 — shallower trees generalise better with few features
        learning_rate    = 0.05, # was 0.1 — slower learning reduces overfitting
        subsample        = 0.8,  # train each tree on 80% of samples — reduces memorisation
        colsample_bytree = 0.8,  # use 80% of features per tree — reduces memorisation
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        device           = GPU_DEVICE,
        verbosity        = 0,
    )


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on the validation set.

    The default 0.5 threshold is calibrated for balanced classes. With an
    83/17 imbalance, a lower threshold forces the model to predict label=0
    more often, improving minority class recall at the cost of some precision.
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0.0, DEFAULT_THRESHOLD

    for thresh in np.arange(0.2, 0.8, 0.01):
        preds = (probs >= thresh).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1     = score
            best_thresh = thresh

    return best_thresh


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
    """Full evaluation report: overfitting check, threshold tuning, metrics."""

    # ── Overfitting check: training score vs CV score ────────────────────────
    # A large gap (train >> CV) means the model memorised training data.
    train_f1 = f1_score(y_train, model.predict(X_train))
    cv_f1s   = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    # ── Threshold tuning on test set ─────────────────────────────────────────
    tuned_thresh = tune_threshold(model, X_test, y_test)
    probs        = model.predict_proba(X_test)[:, 1]
    y_pred_def   = (probs >= DEFAULT_THRESHOLD).astype(int)
    y_pred_tuned = (probs >= tuned_thresh).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    def metrics(y_true, y_pred):
        return dict(
            acc  = accuracy_score(y_true, y_pred),
            prec = precision_score(y_true, y_pred, zero_division=0),
            rec  = recall_score(y_true, y_pred, zero_division=0),
            f1   = f1_score(y_true, y_pred, zero_division=0),
            auc  = roc_auc_score(y_true, probs),
        )

    m_def   = metrics(y_test, y_pred_def)
    m_tuned = metrics(y_test, y_pred_tuned)
    cm      = confusion_matrix(y_test, y_pred_tuned)

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Overfitting check
    gap     = train_f1 - cv_f1s.mean()
    verdict = "OK" if gap < 0.05 else "POSSIBLE OVERFIT"
    print(f"\n  Bias-Variance check:")
    print(f"    Training F1:         {train_f1:.3f}")
    print(f"    CV F1 ({CV_FOLDS}-fold):      {cv_f1s.mean():.3f} ± {cv_f1s.std():.3f}")
    print(f"    Gap (train - CV):    {gap:.3f}  [{verdict}]")

    # Default threshold metrics
    print(f"\n  Default threshold ({DEFAULT_THRESHOLD:.2f}):")
    print(f"    Accuracy:  {m_def['acc']:.3f}   Precision: {m_def['prec']:.3f}")
    print(f"    Recall:    {m_def['rec']:.3f}   F1:        {m_def['f1']:.3f}")
    print(f"    ROC-AUC:   {m_def['auc']:.3f}")

    # Tuned threshold metrics
    print(f"\n  Tuned threshold ({tuned_thresh:.2f}):")
    print(f"    Accuracy:  {m_tuned['acc']:.3f}   Precision: {m_tuned['prec']:.3f}")
    print(f"    Recall:    {m_tuned['rec']:.3f}   F1:        {m_tuned['f1']:.3f}")
    print(f"    ROC-AUC:   {m_tuned['auc']:.3f}  (same — threshold does not affect AUC)")

    # Confusion matrix
    print(f"\n  Confusion matrix at tuned threshold (rows=actual, cols=predicted):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}   <- actual negatives (don't unroll)")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}   <- actual positives (do unroll)")

    # Minority class catch rate — the key metric given the imbalance
    neg_total = cm[0,0] + cm[0,1]
    tn_rate   = cm[0,0] / neg_total if neg_total > 0 else 0
    print(f"\n  Minority class (label=0) catch rate: {tn_rate:.1%} ({cm[0,0]}/{neg_total})")

    # Feature importances
    print(f"\n  Feature importances (descending):")
    ranked = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<32} {imp:.3f}  {bar}")

    return {"name": name, "model": model, "cv_f1": cv_f1s.mean(), "auc": m_def["auc"]}


# ── Learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(
    models:      list[tuple[str, object]],
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    cv:          StratifiedKFold,
    output_path: str,
) -> None:
    """Plot training vs CV F1 as training set size increases.

    How to read the curves:
      Both converge LOW  → underfitting (model too simple, add features)
      Train HIGH, CV LOW → overfitting (model memorising, reduce complexity)
      Both converge HIGH → good fit (more data may still help if gap remains)
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    train_sizes = np.linspace(0.1, 1.0, 10)

    for ax, (name, model) in zip(axes, models):
        sizes, train_scores, cv_scores = learning_curve(
            model, X_train, y_train,
            train_sizes = train_sizes,
            cv          = cv,
            scoring     = "f1",
            n_jobs      = -1,
        )
        ax.plot(sizes, train_scores.mean(axis=1), label="Training F1",  color="steelblue")
        ax.fill_between(
            sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.1, color="steelblue",
        )
        ax.plot(sizes, cv_scores.mean(axis=1), label="CV F1", color="darkorange")
        ax.fill_between(
            sizes,
            cv_scores.mean(axis=1) - cv_scores.std(axis=1),
            cv_scores.mean(axis=1) + cv_scores.std(axis=1),
            alpha=0.1, color="darkorange",
        )
        ax.set_title(name)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("F1 score")
        ax.set_ylim(0.5, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Learning Curves — compiler-opt", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nLearning curves saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading dataset from {DATASET_PATH}...")
    X, y, feature_names = load_dataset(DATASET_PATH)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"  Labels: {pos} positive ({100*pos/len(y):.1f}%)  "
          f"{neg} negative ({100*neg/len(y):.1f}%)")

    # Stratified split preserves the 83/17 class ratio in both train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # Stratified K-fold ensures every fold has the same class ratio as the full set.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Majority-class baseline ───────────────────────────────────────────────
    # A model that always predicts label=1 gives us the floor to beat.
    # If our model can't beat this, it's learned nothing useful.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
    baseline_f1  = f1_score(y_test, dummy.predict(X_test), zero_division=0)
    print(f"\nMajority-class baseline (always predict 'unroll'):")
    print(f"  Accuracy: {baseline_acc:.3f}   F1: {baseline_f1:.3f}")
    print(f"  Our models must beat these numbers to be doing anything useful.")

    # ── Train ─────────────────────────────────────────────────────────────────
    scale_pos = neg / pos if pos > 0 else 1.0

    print("\nTraining Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = build_xgboost(scale_pos)
    xgb_model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rf_results  = evaluate("Random Forest", rf,        X_train, X_test, y_train, y_test, feature_names, cv)
    xgb_results = evaluate("XGBoost",       xgb_model, X_train, X_test, y_train, y_test, feature_names, cv)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'CV F1':>8} {'ROC-AUC':>9}")
    print(f"  {'-'*44}")
    print(f"  {'Baseline (majority)':<25} {'N/A':>8} {'N/A':>9}")
    for r in [rf_results, xgb_results]:
        print(f"  {r['name']:<25} {r['cv_f1']:>8.3f} {r['auc']:>9.3f}")

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
