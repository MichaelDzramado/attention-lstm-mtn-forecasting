# README --- **"Attention-Enhanced LSTM for MTN Ghana Stock Price Forecasting."**

## Project Overview


The project compares a conventional **Baseline LSTM** with a custom
**Attention-LSTM** across three forecasting formulations:

-   **Study I:** Original closing-price target
-   **Study II:** Scaled closing-price target
-   **Study III:** Next-day log-return target with reconstructed
    closing-price evaluation

------------------------------------------------------------------------

## Folder Structure and Linkage

``` text
Sourcce code/
│
├── models/
│   ├── lstm.py
│   ├── attention.py
│   ├── attention_lstm.py
│   ├── config.py
│   ├── feature_engineering.py
│   └── data/
│
├── utils/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── trainer.py
│   ├── metrics.py
│   ├── visualization.py
│   └── seed.py
│
├── notebooks/
│   ├── EDA_Feature_Engineering.ipynb
│   ├── Feature_Selection.ipynb
│   ├── Study _I_featurescaling_Training .ipynb
│   ├── Study_II_TargetScaling.ipynb
│   ├── Study III_Logreturn_Training.ipynb
│   └── final_cross_comparison.ipynb
│
├── model_checkpoints/
│   └── Study I–III trained .pth files
│
└── results/
    └── Training histories, predictions and evaluation results
```

### How the components work together

**1. Data → Feature preparation**

The files in `models/data/` provide the raw, cleaned, and final
datasets.\
`feature_engineering.py` contains the feature-preparation workflow used
to construct the modelling variables and next-day target.

**2. Preprocessing → Sequence preparation**

`utils/preprocessing.py` handles chronological ordering, feature/target
separation, dataset splitting, and scaling.\
`utils/dataset.py` converts the prepared sequences into PyTorch datasets
for model training.

**3. Model configuration → Architecture**

`models/config.py` stores the main experimental settings, including the
30-step sequence length, two LSTM layers, hidden size, dropout, training
ratios, and other model parameters.

`models/lstm.py` implements the **Baseline LSTM**.

`models/attention.py` implements the custom temporal attention
mechanism.

`models/attention_lstm.py` combines the stacked LSTM with the custom
attention mechanism to implement the **Attention-LSTM**.

**4. Training → Evaluation**

`utils/trainer.py` provides the common training and validation
procedure, including optimization, validation monitoring, learning-rate
scheduling, early stopping, and checkpoint handling.

`utils/metrics.py` calculates **MAE, MSE, RMSE, MAPE, and R²**.

`utils/seed.py` supports reproducible experiments.

**5. Notebooks → Experimental studies**

The notebooks are the main experiment workflows:

-   `EDA_Feature_Engineering.ipynb` --- exploratory analysis and feature
    engineering.
-   `Feature_Selection.ipynb` --- feature-selection analysis.
-   `Study _I_featurescaling_Training .ipynb` --- Study I training and
    evaluation.
-   `Study_II_TargetScaling.ipynb` --- Study II training and evaluation.
-   `Study III_Logreturn_Training.ipynb` --- Study III log-return
    modelling, reconstruction, and evaluation.
-   `final_cross_comparison.ipynb` --- combines the saved study results
    and checkpoints to produce the final cross-study comparisons and
    report visualisations.

The study notebooks import the reusable modules from `models/` and
`utils/` rather than duplicating the core model and training
implementations.

**6. Checkpoints → Results**

The `model_checkpoints/` folder contains the trained `.pth` files for
the Baseline LSTM and Attention-LSTM across the three studies.

The `results/` folder contains the exported training histories,
predictions, and evaluation tables used by the final comparison notebook
and the report.

------------------------------------------------------------------------

## Reproduction Flow

The overall project workflow can therefore be viewed as:

``` text
Data
  ↓
EDA & Feature Engineering
  ↓
Feature Selection
  ↓
Preprocessing & Chronological Split
  ↓
30-Step Sequence Construction
  ↓
Baseline LSTM / Attention-LSTM
  ↓
Training & Validation
  ↓
Saved Model Checkpoints
  ↓
Test Predictions & Metrics
  ↓
Study I + Study II + Study III
  ↓
Final Cross-Study Comparison
  ↓
Report Visualisations & Conclusions
```

