# Assignment 2 — Golub Weighted Voting on WDBC

This project applies the weighted-voting method described by Golub et al. (1999) to a different biological classification problem: distinguishing benign from malignant breast tumors using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The method is compared with logistic regression using the supplied WDBC CSV file. If `data/MANIFEST.json` is present, the program also verifies the CSV file's checksum to check data integrity.

## Project Structure

```text
.
├── analysis.py          # Data loading, descriptive statistics, models,
│                        # cross-validation, metrics, and stability analysis
├── visualization.py     # Figure generation
├── main.py              # End-to-end pipeline
├── requirements.txt     # Python dependencies
├── data/
│   ├── wdbc.csv
│   ├── MANIFEST.json
│   └── DATA_DICTIONARY.md
└── results/
    ├── tables/
    ├── figures/
    └── summary.json
```

## Installation

Make sure Python 3.10 or a newer version is installed.

Install the project dependencies from the root directory:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

To run the entire analysis:

```bash
python main.py
```

Alternatively, you can provide an already-computed seed directly (default seed is 42):

```bash
python main.py --seed YOUR_PERSONAL_SEED
```

To specify a different output directory:

```bash
python main.py --seed YOUR_PERSONAL_SEED --results results
```



## Pipeline Stages

1. **Data verification and validation:** Verify the CSV checksum if the manifest is present, then validate the class labels and input features.

2. **Descriptive and exploratory analysis:** Calculate class counts, feature summaries, class means, correlations, and exploratory figures. These analyses are not used to select model features.

3. **Cross-validation:** Perform stratified 5-fold cross-validation repeated 3 times, resulting in 15 evaluation splits per model.

4. **Golub weighted voting:** Calculate class means, standard deviations, feature weights, and the 10 features with the largest absolute weights using only the training data of each fold. A sample is classified as malignant when its weighted-voting score is greater than or equal to zero.

5. **Logistic regression:** Evaluate the model on exactly the same splits as the weighted-voting method. Fit `StandardScaler` using only the training data, then train logistic regression with the following parameters:

   - `solver="liblinear"`
   - `C=1.0`
   - `max_iter=1000`
   - Default classification probability threshold: 0.5

6. **Evaluation and result export:** Save fold-level metrics, mean and standard deviation of metrics, an illustrative confusion matrix, feature-selection frequencies, mean pairwise Jaccard similarity, and the corresponding figures.

## Output Files

All results are saved in the `results/` directory after the pipeline has completed.

### Tables

- `results/tables/feature_summary.csv`: Descriptive statistics for the features.
- `results/tables/class_means.csv`: Mean feature values for each class.
- `results/tables/feature_diagnosis_correlations.csv`: Correlations between features and diagnosis.
- `results/tables/class_counts.csv`: Number of samples in each class.
- `results/tables/all_fold_metrics.csv`: Evaluation metrics for each cross-validation fold.
- `results/tables/metric_summary.csv`: Summary of metrics, including means and standard deviations.
- `results/tables/selected_features_by_fold.csv`: Features selected in each fold.
- `results/tables/feature_selection_frequency.csv`: Frequency with which each feature is selected.

### Figures and Summary

- `results/figures/`: Figures generated during the analysis, saved in PNG format.
- `results/summary.json`: Summary of the experimental results in JSON format.

## Interpretation Cautions

- **Unit of observation:** Each row represents one patient or sample. Cross-validation splits patient-level rows, not individual cell nuclei.

- **Data leakage prevention:** All quantities learned by the models are estimated using only the training data in each fold. For logistic regression, feature scaling is also fitted exclusively on the training data.

- **Dependence between evaluations:** The 15 fold scores do not represent 15 independent experiments because training and test sets overlap across repetitions. Means and standard deviations should be interpreted descriptively, not as estimates based on 15 independent observations.

- **Dataset limitations:** The supplied CSV represents a curated development dataset and does not necessarily reflect a population undergoing clinical screening. Internal cross-validation does not replace independent external validation.

- **Scope of classification:** The models predict the supplied diagnosis from pre-extracted nuclear measurements. They do not evaluate raw-image segmentation, the original image acquisition pipeline, clinical history, or prospective clinical outcomes.

- **Illustrative results:** The first fold of the weighted-voting method is used as an example for the confusion matrix, AUC, and balanced accuracy. These illustrative results should not be confused with the average performance across all folds.

## Project Objective

The objective is to investigate the applicability of Golub's weighted-voting method to a classification problem different from the one addressed in the original paper, and to compare its performance with logistic regression under the same evaluation conditions.

The analysis examines both classification performance and the stability of selected features across cross-validation folds.

## Citations

Work based on "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring" paper from T R Golub 1, D K Slonim, P Tamayo, C Huard, M Gaasenbeek, J P Mesirov, H Coller, M L Loh, J R Downing, M A Caligiuri, C D Bloomfield, E S Lander

Wolberg, W., Mangasarian, O., Street, N., and Street, W. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. DOI: 10.24432/C5DW2B.

Street, W. N., Wolberg, W. H., and Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE Symposium on Electronic Imaging*, 1905, 861–870. DOI: 10.1117/12.148698.

Wolberg, W. H., Street, W. N., and Mangasarian, O. L. (1994). Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates. *Cancer Letters*, 77(2–3), 163–171. DOI: 10.1016/0304-3835(94)90099-X.