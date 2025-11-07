# LightDarkBox-ML
Raw data and code for the paper “Assessment of Visual Function in Mice Using Light/Dark Box and Multi-Feature Machine Learning”

This repository contains the raw data and analysis code for the study
"Assessment of Visual Function in Mice Using Light/Dark Box and Multi-Feature Machine Learning"
(submitted to *PLOS Computational Biology*).

## Directory Structure
```
LightDarkBox-ML/
│
├── Framewise position data/ # Raw data recording the framewise position information (x, y) of each mouse during the recording period.
├── Mice information/ # Records the experimental information of each mouse, including group, sex, date of birth, and experiment date, etc.
├── Sliding window features/ # Extracted behavioral features in sliding windows
├── Training models/ # Stores the trained machine learning models.
│
├── main.py # Entry script for data processing and analysis
│
├── train_ml_models.py # Training and evaluation of ML classifiers (SVM, RF, etc.)
├── window_feature_extractor.py # Extracts behavioral features within time windows
├── draw_boxplot.py # Draws group comparison boxplots
├── draw_rf_feature.py # Plots Random Forest feature importance
├── draw_roc.py # Generates ROC curves for model evaluation
├── plot_confusion_matrix.py # Plots confusion matrices of classifiers
├── model_comparison.py # Compares performance among different models
├── opt_feature_group_analysis.py # Feature group optimization and analysis
├── dark_histogram.py # Generates histograms of dark-side occupancy
│
└── README.md # Project documentation (this file)
```

## Usage
All analysis procedures, including feature extraction, model training, and result plotting,
are organized in the `main.py` script. You can simply **uncomment** the corresponding code blocks
to run each step of the workflow sequentially.

### 1. Feature Extraction
In `main.py`, uncomment the block starting with:
```
# ===== Sliding window, extract features and write to file =====
```

This section will extract sliding-window features from framewise position data
and save them to the `Sliding window features/` directory.

### 2. Model Training
Uncomment the following block:
```
# ===== Train ML models =====
```

This section will train multiple machine learning models (e.g., SVM, RF, MLP, XGBoost)
based on the extracted features and save trained models to the `Training models/` folder.

### 3. Draw the figures from the paper
Uncomment the following blocks in sequence:
```
# ===== Figure 1 =====
# ===== Figure 2 =====
# ===== Figure 3 =====
# ===== Figure 4 =====
# ===== Discussion Figure =====
```

**Tip:**
If you want to quickly reproduce the figures from the paper, go directly to Step 3 — the extracted features and trained models are already included in the directory.

## Citation
(Coming soon)

## License
(Coming soon)

## Contact
**Tengxiao Wang** (Chongqing University)
Email: tengxiao.wang@stu.cqu.edu.cn

