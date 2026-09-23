import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import build_feature_table, TRAIN_FS
from features import FEATURE_NAMES

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = "data/merged_eeg_datas_50_subjects.csv"
MODELS_DIR = "models"
N_TEST_SUBJECTS = 10
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("STEP 1: Building feature table from raw EEG")
    print("=" * 70)
    X, y, groups = build_feature_table(DATA_PATH)
    print()
    print("Class balance:")
    print(y.value_counts())
    print()

    print("=" * 70)
    print("STEP 2: GroupKFold cross-validation (baseline comparison)")
    print("=" * 70)
    gkf = GroupKFold(n_splits=5)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    )

    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="f1_macro")
        print(f"{name:22s} macro-F1 per fold: {np.round(scores, 3)}  "
              f"mean={scores.mean():.3f}  std={scores.std():.3f}")
    print()

    print("=" * 70)
    print("STEP 3: Final subject-holdout evaluation (Random Forest)")
    print("=" * 70)
    gss = GroupShuffleSplit(n_splits=1, test_size=N_TEST_SUBJECTS / groups.nunique(),
                             random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_subjects = groups.iloc[train_idx].nunique()
    test_subjects = groups.iloc[test_idx].nunique()
    print(f"Train subjects: {train_subjects}  ({len(X_train)} rows)")
    print(f"Test subjects:  {test_subjects}  ({len(X_test)} rows, never seen in training)")
    print()

    final_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=predicted), labels order:",
          sorted(y.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))
    print()

    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, final_model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:20s} {imp:.3f}")
    print()

    print("=" * 70)
    print("STEP 4: Refitting on ALL data and saving artifacts")
    print("=" * 70)
    deployed_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    deployed_model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(deployed_model, os.path.join(MODELS_DIR, "workload_classifier.joblib"))

    metadata = {
        "feature_names": FEATURE_NAMES,
        "labels": sorted(y.unique().tolist()),
        "train_fs_hz": TRAIN_FS,
        "held_out_subject_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODELS_DIR}/workload_classifier.joblib")
    print(f"Saved metadata to {MODELS_DIR}/metadata.json")
    print(f"Held-out subject macro-F1: {metadata['held_out_subject_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
