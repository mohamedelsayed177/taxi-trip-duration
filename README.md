# NYC Taxi Trip Duration Prediction

This project builds a machine learning pipeline to predict the total ride duration of taxi trips in New York City.

Status: Under Active Development

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
1. Created the comprehensive file structure and packages.
2. Built the dynamic data engineering modules.
3. Completed the pure baseline model code.
4. Next step is conducting the Exploratory Data Analysis (EDA).
