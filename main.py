from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis import (
    descriptive_exploration,
    feature_selection_stability,
    load_dataset,
    run_cross_validation,
    summarize_metrics,
)
from visualization import (
    save_confusion_matrix,
    save_exploration_figures,
    save_metric_comparison,
    save_selection_frequency,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "wdbc.csv"
DEFAULT_RESULTS = PROJECT_DIR / "results"


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of a file"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_path: Path, manifest_path: Path) -> None:
    """Verify the frozen CSV against its manifest when the manifest is present"""
    if not manifest_path.exists():
        print(f"Warning: manifest not found at {manifest_path}; skipping checksum check")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("sha256")
    actual_hash = sha256_file(data_path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "Dataset SHA256 does not match data/MANIFEST.json"
            "Do not replace or modify the supplied CSV"
        )
    if expected_hash:
        print("Dataset SHA256: verified against MANIFEST.json")


def save_tables(
    exploration: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    metric_summary: pd.DataFrame,
    selections: pd.DataFrame,
    frequency_table: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Write all analysis tables as CSV for the report"""
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in exploration.items():
        table.to_csv(tables_dir / f"{name}.csv")
    metrics.to_csv(tables_dir / "all_fold_metrics.csv", index=False)
    metric_summary.to_csv(tables_dir / "metric_summary.csv", index=False)
    selections.to_csv(tables_dir / "selected_features_by_fold.csv", index=False)
    frequency_table.to_csv(tables_dir / "feature_selection_frequency.csv", index=False)
    fold_metrics.to_csv(tables_dir / "fold_metrics_with_selection_count.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Golub weighted-voting vs logistic regression pipeline"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to the supplied data/wdbc.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "MANIFEST.json",
                        help="Path to data/MANIFEST.json")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Directory in which tables and figures are written")
    parser.add_argument("--day", type=int, help="Birth day used only to compute the seed")
    parser.add_argument("--month", type=int, help="Birth month used only to compute the seed")
    parser.add_argument("--year-suffix", type=int,
                        help="Last two digits of birth year used only to compute the seed")
    parser.add_argument("--seed", type=int,
                        help="Alternatively provide the already-computed personal seed")
    args = parser.parse_args()

    if args.seed is None:
        seed = 42
    else:
        seed = args.seed
        if seed < 0:
            raise ValueError("Seed must be non-negative")

    print("\n[1/6] Verify supplied dataset and load data...")
    verify_manifest(args.data, args.manifest)
    X, y, diagnosis = load_dataset(args.data)
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Class counts: {diagnosis.value_counts().to_dict()}")

    results_dir = args.results.resolve()
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/6] Descriptive exploration (not used for feature selection)...")
    exploration = descriptive_exploration(X, y)
    save_exploration_figures(X, y, exploration, figures_dir)

    print("\n[3/6] Repeated stratified 5-fold cross-validation (3 repeats)...")
    metrics, selections, example = run_cross_validation(X, y, seed=seed, n_features=20)
    metric_summary = summarize_metrics(metrics)

    # The requested confusion matrix is for one example outer fold
    save_confusion_matrix(
        example["y_true"],
        example["vote_pred"],
        figures_dir / "weighted_voting_example_confusion_matrix.png",
    )

    print("\n[4/6] Feature-selection stability...")
    frequency_table, mean_jaccard, fold_metrics = feature_selection_stability(
        selections, list(X.columns)
    )
    save_selection_frequency(
        frequency_table, figures_dir / "feature_selection_frequency.png"
    )

    print("\n[5/6] Compare weighted voting and logistic regression...")
    save_metric_comparison(metrics, figures_dir / "model_metric_comparison.png")

    save_tables(
        exploration, metrics, metric_summary, selections, frequency_table,
        fold_metrics, tables_dir
    )

    # A compact machine-readable record of the central results
    report_summary = {
        "personal_seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in diagnosis.value_counts().items()},
        "cross_validation": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3)",
        "example_weighted_voting_fold": int(example["split"]),
        "example_weighted_voting_metrics": example["vote_metrics"],
        "mean_pairwise_jaccard": mean_jaccard,
        "metric_summary": metric_summary.to_dict(orient="records"),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(report_summary, indent=2), encoding="utf-8"
    )

    print("\n[6/6] Results saved")
    print("\nMean and standard deviation across the 15 validation folds:")
    print(metric_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMean pairwise Jaccard similarity: {mean_jaccard:.3f}")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()
