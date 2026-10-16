from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


METADATA_COLUMNS = {"sample_id", "diagnosis"}


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the supplied CSV and return X, y, and the original diagnosis labels"""
    data = pd.read_csv(csv_path)

    required = {"sample_id", "diagnosis"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    if data["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")

    diagnosis = data["diagnosis"].astype(str).str.lower()
    allowed = {"benign", "malignant"}
    if not set(diagnosis.unique()).issubset(allowed):
        raise ValueError(
            "diagnosis must contain only 'benign' and 'malignant'; "
            f"found {sorted(diagnosis.unique())}"
        )

    y = diagnosis.map({"benign": 0, "malignant": 1}).astype(int)
    X = data.drop(columns=list(METADATA_COLUMNS)).copy()

    X = X.apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("The dataset contains missing feature values")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("The dataset contains infinite feature values")

    return X, y, diagnosis

def descriptive_exploration(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    """Create descriptive summaries"""
    labels = y.map({0: "benign", 1: "malignant"}).rename("diagnosis")
    summary = X.describe().T
    summary.index.name = "feature"
    class_means = X.groupby(labels).mean().T
    class_means.index.name = "feature"
    class_means["malignant_minus_benign"] = (
        class_means["malignant"] - class_means["benign"]
    )
    correlations = X.assign(diagnosis=y).corr(numeric_only=True)["diagnosis"]
    correlations = correlations.drop("diagnosis").rename("diagnosis_correlation").to_frame()
    correlations.index.name = "feature"
    class_counts = y.map({0: "benign", 1: "malignant"}).value_counts().rename_axis(
        "diagnosis"
    ).to_frame("count")
    return {
        "feature_summary": summary,
        "class_means": class_means,
        "feature_diagnosis_correlations": correlations,
        "class_counts": class_counts,
    }


def weighted_voting_fit(
    X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 10
) -> dict[str, Any]:
    """
    arbitrary choice of 10 features
    w_j = (mean_malignant - mean_benign) /
          (std_malignant + std_benign)
    """
    if n_features < 1 or n_features > X_train.shape[1]:
        raise ValueError("n_features must be between 1 and the number of columns")

    malignant = X_train.loc[y_train == 1]
    benign = X_train.loc[y_train == 0]
    if malignant.empty or benign.empty:
        raise ValueError("Both classes must be present in each training fold")

    mu_m = malignant.mean()
    mu_b = benign.mean()
    sd_m = malignant.std(ddof=1)
    sd_b = benign.std(ddof=1)
    denominator = sd_m + sd_b

    weights = (mu_m - mu_b) / denominator.replace(0, np.nan)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Stable sorting makes tied weights reproducible.
    ranked = weights.abs().sort_values(ascending=False, kind="mergesort")
    selected = ranked.index[:n_features].tolist()
    midpoints = (mu_m + mu_b) / 2.0

    return {
        "weights": weights,
        "selected_features": selected,
        "mean_malignant": mu_m,
        "mean_benign": mu_b,
        "midpoints": midpoints,
    }


def weighted_voting_score(X: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    """Calculate V(x) using the selected training-fold weights and midpoints"""
    selected = fitted["selected_features"]
    weights = fitted["weights"].loc[selected].to_numpy(dtype=float)
    midpoints = fitted["midpoints"].loc[selected].to_numpy(dtype=float)
    values = X.loc[:, selected].to_numpy(dtype=float)
    return np.sum(weights * (values - midpoints), axis=1)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Calculate the required metrics"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, scores),
        "tn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 1]),
    }


def run_cross_validation(
    X: pd.DataFrame, y: pd.Series, seed: int, n_features: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Evaluate weighted voting and logistic regression on identical outer splits
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    example: dict[str, Any] | None = None

    for split_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Method 1: weighted voting. All fitted quantities come from X_train/y_train
        fitted = weighted_voting_fit(X_train, y_train, n_features=n_features)
        vote_scores = weighted_voting_score(X_valid, fitted)
        vote_pred = (vote_scores >= 0).astype(int)
        vote_metrics = evaluate_predictions(y_valid, vote_pred, vote_scores)
        metric_rows.append({
            "split": split_number,
            "repeat": (split_number - 1) // 5 + 1,
            "fold": (split_number - 1) % 5 + 1,
            "model": "weighted_voting",
            **vote_metrics,
        })
        selection_rows.append({
            "split": split_number,
            "repeat": (split_number - 1) // 5 + 1,
            "fold": (split_number - 1) % 5 + 1,
            "selected_features": ";".join(fitted["selected_features"]),
        })

        # Save the first outer fold as the required example confusion matrix
        if example is None:
            example = {
                "split": split_number,
                "y_true": y_valid.to_numpy(),
                "vote_pred": vote_pred,
                "vote_scores": vote_scores,
                "vote_metrics": vote_metrics,
            }

        # Method 2: logistic regression with all features; scale only on training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        logistic = LogisticRegression(
            solver="liblinear", C=1.0, max_iter=1000
        )
        logistic.fit(X_train_scaled, y_train)
        log_scores = logistic.predict_proba(X_valid_scaled)[:, 1]
        log_pred = logistic.predict(X_valid_scaled)
        log_metrics = evaluate_predictions(y_valid, log_pred, log_scores)
        metric_rows.append({
            "split": split_number,
            "repeat": (split_number - 1) // 5 + 1,
            "fold": (split_number - 1) % 5 + 1,
            "model": "logistic_regression",
            **log_metrics,
        })

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    if example is None:
        raise RuntimeError("Cross-validation produced no folds")
    return metrics, selections, example


def feature_selection_stability(
    selections: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Compute feature selection frequencies and mean pairwise Jaccard similarity"""
    selected_sets = [
        set(value.split(";")) for value in selections["selected_features"].tolist()
    ]
    frequency = {
        feature: sum(feature in selected for selected in selected_sets)
        for feature in feature_names
    }
    frequency_table = pd.DataFrame({
        "feature": list(frequency.keys()),
        "selected_folds": list(frequency.values()),
    })
    frequency_table["selection_frequency"] = (
        frequency_table["selected_folds"] / len(selected_sets)
    )
    frequency_table = frequency_table.sort_values(
        ["selected_folds", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    similarities = []
    for left, right in combinations(selected_sets, 2):
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 1.0)
    mean_jaccard = float(np.mean(similarities)) if similarities else float("nan")

    per_fold = selections.copy()
    per_fold["n_selected"] = per_fold["selected_features"].apply(
        lambda value: len(value.split(";")) if value else 0
    )
    return frequency_table, mean_jaccard, per_fold


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean and standard deviation across the 15 outer folds"""
    metric_names = ["accuracy", "balanced_accuracy", "auc"]
    summary = metrics.groupby("model")[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
