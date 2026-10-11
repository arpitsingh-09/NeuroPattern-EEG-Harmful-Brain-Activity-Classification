# NeuroPattern: EEG Harmful Brain Activity Classification

A classical machine-learning baseline for the Kaggle competition **[HMS – Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)**. The project classifies 50-second EEG segments into six brain-activity patterns using hand-crafted signal features with Random Forest and SVM models.

## Problem

Given a 50-second EEG segment, predict the pattern that a panel of expert neurologists voted on:

| Class | Meaning |
|---|---|
| Seizure | Seizure activity |
| LPD | Lateralized periodic discharges |
| GPD | Generalized periodic discharges |
| LRDA | Lateralized rhythmic delta activity |
| GRDA | Generalized rhythmic delta activity |
| Other | None of the above |

The competition metric is **KL divergence** between the predicted probabilities and the experts' vote distribution (soft labels).

## Pipeline

1. **Load data**: `train.csv` metadata plus raw EEG `.parquet` recordings.
2. **Segment extraction**: take the 50 s window (200 Hz, 10,000 samples) using `eeg_label_offset_seconds`.
3. **Preprocessing**
   - Linear interpolation of missing values; reject segments with more than 10% missing data on any electrode
   - 16-channel **bipolar montage** (double-banana layout, e.g. `Fp1-F7`)
   - **Butterworth band-pass filter**, 0.5–40 Hz, order 4, zero-phase (`sosfiltfilt`)
4. **Feature extraction**: 13 features x 16 channels = **208 features per segment**
   - Time domain: mean, std, RMS, min, max, median, variance, skewness, kurtosis
   - Frequency domain (Welch PSD): delta, theta, alpha and beta band power
5. **Labels**: hard labels (expert consensus) and soft targets (vote proportions).
6. **Validation split**: `StratifiedGroupKFold` (5 folds) grouped by `patient_id`, so no patient appears in both train and validation.
7. **Training**: 500 samples per class, validated on all of fold 0 to keep the real class distribution.

## Models

| Model | Details |
|---|---|
| Random Forest | `RandomForestClassifier`, 200 trees |
| SVM + StandardScaler | RBF kernel, `C=1.0`, `gamma="scale"` |
| SVM + RobustScaler | Same SVM, more outlier-resistant scaling |
| SVM + signed log | `sign(x) * log1p(abs(x))`, then StandardScaler, then SVM (final SVM version) |

## Evaluation

- Accuracy
- Macro F1
- KL divergence (competition metric)
- Per-class classification report and confusion matrix

## Results

Run the notebook to reproduce the numbers. They are saved to `baseline_comparison.csv`.

| Model | Accuracy | Macro F1 | KL Divergence |
|---|---|---|---|
| Random Forest | – | – | – |
| SVM (Standard) | – | – | – |
| SVM (Signed Log) | – | – | – |

## Outputs

Saved to `/kaggle/working/rf_features_v1/`:

- `X_train.parquet`, `X_validation.parquet`: feature tables
- `metadata_train.parquet`, `metadata_validation.parquet`
- `errors_train.parquet`, `errors_validation.parquet`: rejected segments
- `fold_assignments.parquet`: shared fold split for later models
- `config.json`: all preprocessing settings
- `random_forest.joblib`, `svm.joblib`, `svm_signed_log.joblib`: trained models
- `metrics.csv`, `baseline_comparison.csv`, `*_validation_predictions.parquet`

## Requirements

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
joblib
pyarrow
```

## How to run

1. Open the notebook on Kaggle and add the **HMS – Harmful Brain Activity Classification** competition data.
2. Run all cells in order.
3. Find the features, models and metrics in `/kaggle/working/rf_features_v1/`.

## Future work

- Train a deep learning model (CNN / ResNet / EfficientNet) on EEG spectrograms
- Add features from the provided spectrogram data
- Train directly on soft labels with a KL-divergence loss
- Use all 5 folds and ensemble the models
- Hyperparameter tuning for Random Forest and SVM

## Disclaimer

This is an educational and research project. It is not a medical device and must not be used for clinical diagnosis.
