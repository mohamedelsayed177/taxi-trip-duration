# NYC Taxi Trip Duration Prediction

This project builds a machine learning pipeline to predict the total ride duration of taxi trips in New York City using Ridge Regression.

**Status:** Active Model Optimization & EDA Finalization

![NYC Taxi Trip Duration Prediction](assets/nyc-taxi-banner.jpg)

---

## Project Structure

```text
taxi-trip-duration/
│
├── assets/
│   └── nyc-taxi-banner.png
│
├── dataset/
│   ├── split/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   │
│   └── split_sample/
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
│
├── notebook/
│   └── EDA.ipynb
│
├── results/
│   └── results.md
│
├── src/
│   └── taxi_trip_duration/
│       ├── __init__.py
│       │
│       ├── manipulation/
│       │   ├── __init__.py
│       │   ├── data_helper.py
│       │   └── feature_engineering.py
│       │
│       ├── model_building/
│       │   ├── __init__.py
│       │   ├── baseline.py
│       │   ├── explore.py
│       │   └── final_pipeline.py
│       │
│       ├── model_testing/
│       │   ├── __init__.py
│       │   └── test.py
│       │
│       └── eda/
│           ├── __init__.py
│           └── plotting.py
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Current Progress

1. **Project Structure:** Established a modular package structure separating data manipulation, feature engineering, model development, testing, and EDA utilities.

2. **Exploratory Data Analysis:** Developed the `EDA.ipynb` notebook, covering initial target-distribution analysis, geographic exploration, data-quality checks, and explanations of the engineered features. Deeper visual analysis is still being finalized.

3. **Feature Engineering:** Developed reusable feature-engineering and data-handling modules, including `data_helper.py` and `feature_engineering.py`, alongside plotting utilities.

4. **Model Development:** Implemented and benchmarked Ridge Regression experiments in `baseline.py` and `explore.py`, using a fixed regularization parameter of `alpha=1`.

   The best recorded validation performance is approximately **R² = 0.712**, achieved with engineered features and One-Hot Encoding. Further analysis and model optimization are ongoing.

---

## Model Configuration

The project uses Ridge Regression with a fixed regularization parameter:

| Parameter | Configuration |
|---|---|
| Model | Ridge Regression |
| Regularization parameter (`alpha`) | `1.0` |
| Target transformation | `np.log1p(trip_duration)` |
| Evaluation metric | R² score |
| Model selection | Validation-set performance |

The modeling experiments focus on understanding how data preparation and feature engineering affect predictive performance while keeping the model and its regularization parameter fixed.

---

## Feature Engineering

The project explores additional representations of the original trip information, including:

- **Geographic features:** Haversine distance, Manhattan-style distance, pickup bearing, and airport proximity.
- **Temporal features:** Pickup date and time components, cyclical time encodings, and rush-hour indicators.
- **Interaction features:** Distance features combined with the rush-hour indicator.
- **Spatial categories:** Pickup and drop-off grids, combined location categories, and day–hour groups.
- **Distance relationships:** Ratios and differences between geographic distance estimates.
- **Log transformations:** Log-scaled trip duration and geographic distances.

These transformations are evaluated through controlled experiments to determine their contribution to validation performance.

---

## Evaluation Strategy

The supplied training and validation datasets are kept separate for model development and feature selection.

- **Training set:** Used to fit the model.
- **Validation set:** Used to compare feature configurations and assess predictive performance during development.
- **Test set:** Reserved for the final evaluation.

The reported validation score is a development result and should not be interpreted as the final test performance.

---

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Ruff
- uv

---

## Project Status

The project is under active development. Current work focuses on finalizing the EDA, reviewing engineered features, optimizing the Ridge Regression pipeline, and preparing the final evaluation and report.

