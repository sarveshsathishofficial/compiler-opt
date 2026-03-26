# model/train.py
# Trains and evaluates ML models to predict whether unrolling a loop
# will improve runtime performance.
#
# Models trained:
#   - Random Forest  (baseline, good on small tabular data)
#   - XGBoost        (usually stronger on tabular data, compare against RF)
#
# GPU note (AMD RX 9060 XT):
#   At the current dataset size (1000 rows, 8 features), CPU training is faster
#   than GPU due to data transfer overhead. GPU acceleration is relevant in
#   Phase 3 when the dataset grows and we move to neural networks.
#   XGBoost supports AMD GPUs via ROCm — see GPU_DEVICE below to enable.
#
# Output:
#   - Console: metrics table and feature importances
#   - model/random_forest.pkl  — saved Random Forest model
#   - model/xgboost.pkl        — saved XGBoost model

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import sys
import csv
import math
import pickle
import numpy as np

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing   import StandardScaler
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
MODEL_DIR    = os.path.dirname(__file__)

RANDOM_STATE = 42       # fixed seed for reproducible train/test split
TEST_SIZE    = 0.2      # 80% train, 20% test
CV_FOLDS     = 5        # cross-validation folds for more reliable estimates

# XGBoost GPU device — set to "cuda" if ROCm is installed and you want GPU.
# Stays on CPU by default since dataset is small.
# For AMD GPU with ROCm: GPU_DEVICE = "cuda"
GPU_DEVICE = "cpu"

# Features to use from dataset.csv.
# total_iter_product is log-transformed below due to its extreme range.
FEATURE_COLUMNS = [
    "loop_count",
    "total_loop_count",
    "max_nesting_depth",
    "max_iteration_count",
    "avg_iteration_count",
    "total_iter_product",   # will be log-transformed
    "has_dependent_body",
    "has_independent_body",
]

LABEL_COLUMN = "unrolled_faster"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset.csv and return (X, y, feature_names).

    Applies log1p transform to total_iter_product to compress its extreme
    range (2 to 46 trillion) into a scale the model can work with.
    Without this, total_iter_product dominates all other features numerically.
    """
    rows = list(csv.DictReader(open(path)))

    X, y = [], []
    for row in rows:
        features = []
        for col in FEATURE_COLUMNS:
            val = float(row[col])
            # Log-transform to compress the range.
            if col == "total_iter_product":
                val = math.log1p(val)
            features.append(val)
        X.append(features)
        y.append(int(row[LABEL_COLUMN]))

    feature_names = [
        col if col != "total_iter_product" else "log_total_iter_product"
        for col in FEATURE_COLUMNS
    ]

    return np.array(X), np.array(y), feature_names


# ── Model training ────────────────────────────────────────────────────────────

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    n_estimators=200 gives stable feature importances without overfitting.
    class_weight="balanced" compensates for the 83/17 label imbalance.
    """
    model = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = None,   # grow full trees, rely on ensemble for regularisation
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,     # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """Train an XGBoost classifier.

    scale_pos_weight compensates for class imbalance by weighting the
    minority class (label=0) proportionally to its under-representation.
    eval_metric="logloss" is standard for binary classification.
    """
    # Weight minority class to handle 83/17 imbalance.
    neg_count  = int((y_train == 0).sum())
    pos_count  = int((y_train == 1).sum())
    scale_pos  = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 6,
        learning_rate     = 0.1,
        scale_pos_weight  = scale_pos,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        device            = GPU_DEVICE,
        verbosity         = 0,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Print a full evaluation report for a trained model."""
    y_pred = model.predict(X_test)

    # Cross-validation on training set for a more reliable accuracy estimate.
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Test accuracy:   {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Test precision:  {precision_score(y_test, y_pred):.3f}")
    print(f"  Test recall:     {recall_score(y_test, y_pred):.3f}")
    print(f"  Test F1:         {f1_score(y_test, y_pred):.3f}")
    print(f"  CV F1 ({CV_FOLDS}-fold):  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\n  Feature importances (descending):")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<30} {imp:.3f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading dataset from {DATASET_PATH}...")
    X, y, feature_names = load_dataset(DATASET_PATH)
    print(f"  {len(X)} samples, {X.shape[1]} features")
    print(f"  Label distribution: {int(y.sum())} positive ({100*y.mean():.1f}%), "
          f"{int((y==0).sum())} negative ({100*(1-y.mean()):.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # Train both models.
    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate and print reports.
    evaluate("Random Forest",  rf,        X_train, X_test, y_train, y_test, feature_names)
    evaluate("XGBoost",        xgb_model, X_train, X_test, y_train, y_test, feature_names)

    # Save models to disk.
    rf_path  = os.path.join(MODEL_DIR, "random_forest.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgboost.pkl")

    with open(rf_path,  "wb") as f: pickle.dump(rf,        f)
    with open(xgb_path, "wb") as f: pickle.dump(xgb_model, f)

    print(f"\nModels saved:")
    print(f"  {rf_path}")
    print(f"  {xgb_path}")


if __name__ == "__main__":
    main()
