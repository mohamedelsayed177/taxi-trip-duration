"""
New York City Taxi Trip Duration - Baseline Modeling.

This script trains and evaluates a Ridge Regression baseline model.
It uses Scikit-Learn pipelines to apply preprocessing and train the model,
then evaluates its performance using the R² metric on the validation set.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from taxi_trip_duration.manipulation.feature_engineering import (
    target_log,
    preprocess_data,
)
from taxi_trip_duration.manipulation.data_helper import (
    load_dataset,
    drop_id_features,
    train_val_split,
    cat_num_features,
    preprocessing,
)


def ridge_model(preprocessor):
    """Combines the preprocessing steps with a Ridge Regression model.

    Args:
        preprocessor (sklearn.compose.ColumnTransformer): Configured
            preprocessing transformer.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline containing preprocessing
            and the Ridge Regression model.
    """
    model = Ridge(alpha=1.0)
    ridge_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return ridge_pipeline


if __name__ == "__main__":
    print("#" * 50)
    print("Initializing Baseline Ridge Regression...")
    print("#" * 50)

    # Load the training and validation datasets
    train, val, _ = load_dataset(data_type="split")

    # Remove identifier columns
    train = drop_id_features(train)
    val = drop_id_features(val)

    # Extract features from the pickup datetime
    train = preprocess_data(train)
    val = preprocess_data(val)

    # Apply the log1p transformation to the target variable
    train = target_log(train, drop_original=True)
    val = target_log(val, drop_original=True)

    # Separate features and target values
    X_train, y_train, X_val, y_val = train_val_split(train, val)

    # Select categorical and numerical features for the baseline model
    categorical, numerical = cat_num_features()

    # Compare StandardScaler and MinMaxScaler
    for scale in ["standard", "minmax"]:
        print(f"\n[Evaluation Phase] Testing {scale.upper()} scaling.")
        preprocessor = preprocessing(categorical, numerical, scaling=scale)

        # Build and train the pipeline
        pipeline = ridge_model(preprocessor)
        pipeline.fit(X_train, y_train)

        # Predict target values and evaluate the validation set
        preds = pipeline.predict(X_val)
        score = r2_score(y_val, preds)

        print(f">> Baseline Validation R² Score with {scale} scaling: {score:.5f}")
        print("-" * 50)


# ==============================================================================
#                           PIPELINE BENCHMARK RESULTS
# ==============================================================================
# The following results were obtained using the baseline model without
# engineered distance features.
#
# 1. Standard Scaling:
#    - Preprocessing: StandardScaler for numerical features and OneHotEncoder
#      for categorical features.
#    - Validation R² Score: 0.06645
#
# 2. MinMax Scaling:
#    - Preprocessing: MinMaxScaler for numerical features and OneHotEncoder
#      for categorical features.
#    - Validation R² Score: 0.05995
#
# DECISION AND NEXT STEPS:
# ------------------------
# StandardScaler achieved a slightly higher validation R² score than MinMaxScaler,
# so it is selected for the next modeling experiments.
#
# The baseline provides a reference point for evaluating the effect of
# additional features in explore.py and feature_engineering.py, such as
# Haversine distance and Manhattan-style distance.
# ==============================================================================
