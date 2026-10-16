# Battery Prognostics ML

A Python research scaffold for battery state-of-health (SOH) estimation, remaining useful life (RUL) prediction, impedance feature analysis, and physics-informed machine learning.

The repository is being developed as a practical starting point for reproducible experiments. The current release provides data-validation utilities, feature-processing helpers, baseline model components, and evaluation functions. It does not claim validated performance on a particular battery dataset.

## Scope

- Inspect cycle-level battery data and check required columns.
- Derive simple degradation and impedance summaries from user-provided data.
- Establish baseline sequence-model components for RUL experiments.
- Evaluate predictions with standard regression metrics.
- Keep data, notebooks, source code, and reported results separate for reproducibility.

## Repository layout

```text
battery-prognostics-ml/
├── data/                 # Data instructions; datasets are not included
├── notebooks/            # Starter notebooks for analysis and modelling
├── src/                  # Reusable Python modules
├── results/              # Notes on storing experiment outputs
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Place data you are permitted to use in a local data directory and follow the schema described in [`data/README.md`](data/README.md). Raw datasets are intentionally not committed. Do not upload private, licensed, or restricted research data without permission.

The utilities expect tabular cycle-level data. Dataset-specific column names and end-of-life definitions should be stated in each experiment rather than assumed to be universal.

## Start here

1. Run `notebooks/01_data_exploration.ipynb` to inspect a local CSV file.
2. Use `notebooks/02_feature_engineering.ipynb` to review basic feature-processing examples.
3. Use `notebooks/03_rul_prediction_lstm.ipynb` to see the structure of a baseline sequence-model experiment.

The notebooks contain starter workflows, not published experimental results. Update paths and column names to match your data before running them.

## Methodological notes

- Split data by battery when evaluating cross-battery generalization; random row splits can leak information between train and test sets.
- Define RUL and end-of-life consistently with the dataset protocol and document any post-EOL observations.
- Fit preprocessing steps on training data only.
- Report the evaluation protocol, sample counts, metrics, and uncertainty where appropriate.
- A physics-informed loss must encode a stated physical relationship and be tested against a suitable baseline. A class name alone does not make a model physics-informed.
- Do not report benchmark claims until the experiment has been run and independently checked.

## Current status

This is an initial research toolkit. The data loaders and feature utilities provide basic reusable functionality; model components are starting points that require dataset-specific training, validation, and tuning. No performance claims are made in this repository at this stage.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
