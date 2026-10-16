from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_exploration_figures(
    X: pd.DataFrame,
    y: pd.Series,
    exploration: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save descriptive plots. These plots are not used for model feature selection"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = y.map({0: "Benign", 1: "Malignant"}).value_counts().reindex(
        ["Benign", "Malignant"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Diagnosis class counts")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of patients")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "class_counts.png", dpi=180)
    plt.close(fig)

    corr = exploration["feature_diagnosis_correlations"][
        "diagnosis_correlation"
    ].sort_values()
    fig, ax = plt.subplots(figsize=(9, 10))
    corr.plot(kind="barh", ax=ax)
    ax.set_title("Feature correlation with diagnosis (descriptive only)")
    ax.set_xlabel("Pearson correlation (malignant = 1)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_diagnosis_correlations.png", dpi=180)
    plt.close(fig)

    # Display class-wise means after z-scoring each feature for visualization only
    # This transformation is descriptive and is not used by either predictive model
    class_means = exploration["class_means"][["benign", "malignant"]]
    pooled = X.std(ddof=1).replace(0, np.nan)
    standardized_means = (class_means - X.mean(axis=0).to_numpy()[:, None]) if False else None
    # Use per-feature min-max scaling so heterogeneous units do not dominate the plot
    minimum = X.min()
    span = (X.max() - minimum).replace(0, np.nan)
    X_scaled = (X - minimum) / span
    plot_data = X_scaled.assign(diagnosis=y.map({0: "Benign", 1: "Malignant"}))
    means = plot_data.groupby("diagnosis").mean().T
    fig, ax = plt.subplots(figsize=(8, 10))
    means.plot(kind="barh", ax=ax)
    ax.set_title("Class-wise feature means (min-max scaled for display)")
    ax.set_xlabel("Scaled mean (visualization only)")
    fig.tight_layout()
    fig.savefig(output_dir / "classwise_feature_means.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path
) -> None:
    """Save a confusion matrix for the first weighted-voting validation fold"""
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Benign (0)", "Malignant (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    display.ax_.set_title("Weighted voting: first validation fold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metric_comparison(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the fold-wise distributions of the required evaluation metrics"""
    long = metrics.melt(
        id_vars=["split", "model"],
        value_vars=["balanced_accuracy", "auc"],
        var_name="metric",
        value_name="score",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["balanced_accuracy", "auc"]):
        subset = long[long["metric"] == metric]
        data = [
            subset.loc[subset["model"] == model, "score"].to_numpy()
            for model in ["weighted_voting", "logistic_regression"]
        ]
        ax.boxplot(data, tick_labels=["Weighted voting", "Logistic regression"])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Outer-fold performance comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_selection_frequency(
    frequency_table: pd.DataFrame, output_path: str | Path
) -> None:
    """Plot all features by number of outer folds in which they were selected"""
    plot_data = frequency_table.sort_values("selected_folds", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(plot_data["feature"], plot_data["selected_folds"])
    ax.set_xlabel("Number of folds selected (out of 15)")
    ax.set_title("Weighted-voting feature-selection frequency")
    ax.set_xlim(0, 15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
