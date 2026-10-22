# NYC Taxi Trip Duration Prediction

This project builds a machine learning pipeline to predict the total ride duration of taxi trips in New York City using Ridge Regression.

Status: Active Model Optimization & EDA Finalization

---

## Project Structure

```text
taxi-trip-duration/
│
├── dataset/
│   ├── split/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
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

## Current Progress
1. Established a modular package structure separating data manipulation, feature engineering, modeling, testing, and EDA plotting utilities.
2. Completed Exploratory Data Analysis (`EDA.ipynb`) covering target skewness, spatial outlier detection, and enriched deep-dive insights.
3. Developed dynamic feature engineering and plotting helper modules (`data_helper.py`, `feature_engineering.py`, `plotting.py`).
4. Implemented and benchmarked Ridge Regression experiments (`baseline.py`, `explore.py`), achieving peak validation performance (R² ~0.712) using raw data configurations and One-Hot Encoding.
