"""
dataset.py

Loads merged_eeg_datas_50_subjects.csv, filters bad-signal rows, and
converts each row's raw_eeg array into a feature vector via features.py
-- the SAME extraction function the live app uses.

Deliberately excluded from features (and why):
    task_id                -> directly determines the label (leakage)
    student_id              -> not a feature, used only as the GroupKFold group key
    created_at               -> not physiologically meaningful
    is_correct                -> not the target; workload_score is
    neuro_raw_eeg (as tabular) -> we extract features from it, don't feed it raw
    neuro_delta / theta / loAlpha / hiAlpha / loBeta / hiBeta / loGamma / midGamma
                               -> chip-computed bands; BioAmp+Arduino can't reproduce
                                  this proprietary processing live
    neuro_attention / neuro_meditation -> proprietary chip-derived, same reason
    neuro_dominan_rhythm       -> proprietary chip-derived, same reason
    neuro_relative_*            -> also chip-computed (kept ONLY for the quick
                                  sanity-check baseline in the notebook, never
                                  in the real model)
"""

import json
import numpy as np
import pandas as pd

from features import extract_features, FEATURE_NAMES

# The dataset's column names match NeuroSky ThinkGear/TGAM chip fields,
# whose raw EEG output rate is 512Hz -- confirmed by raw_eeg array lengths
# in this file (modal length 512). This is NOT the same rate as the live
# BioAmp+Arduino (~250Hz); features.py handles that via the fs parameter.
TRAIN_FS = 512

LABEL_MAP = {0.0: "low", 0.5: "moderate", 1.0: "high"}


def load_raw(csv_path):
    return pd.read_csv(csv_path)


def build_feature_table(csv_path, max_poor_signal=0, verbose=True):
    """
    Returns (X, y, groups) ready for GroupKFold / model training:
        X       -- DataFrame[FEATURE_NAMES]
        y       -- Series of string labels ("low"/"moderate"/"high")
        groups  -- Series of student_id, for subject-independent splitting
    """
    df = load_raw(csv_path)
    n_total = len(df)

    df = df[df["neuro_poor_signal_strength"] <= max_poor_signal].copy()
    n_after_signal_filter = len(df)

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            raw = json.loads(row["neuro_raw_eeg"])
        except (TypeError, ValueError):
            skipped += 1
            continue

        feats = extract_features(raw, fs=TRAIN_FS)
        if feats is None:
            skipped += 1
            continue

        feats["workload_score"] = row["workload_score"]
        feats["student_id"] = row["student_id"]
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    if verbose:
        print(f"Total rows in CSV:              {n_total}")
        print(f"After poor_signal filter:       {n_after_signal_filter}")
        print(f"Skipped (bad/short raw_eeg):     {skipped}")
        print(f"Final feature rows:              {len(feat_df)}")
        print(f"Unique subjects remaining:       {feat_df['student_id'].nunique()}")

    X = feat_df[FEATURE_NAMES]
    y = feat_df["workload_score"].map(LABEL_MAP)
    groups = feat_df["student_id"]

    return X, y, groups
