"""
NYC Taxi Trip Duration - Feature Engineering Experiments.

Evaluates baseline and expanded feature configurations using Ridge
Regression with alpha=1.0. Compares training and validation R² scores
on the log-transformed trip duration.
"""

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from taxi_trip_duration.manipulation.data_helper import (
    cat_num_features,
    drop_id_features,
    load_dataset,
    preprocessing,
    train_val_split,
)
from taxi_trip_duration.manipulation.feature_engineering import (
    preprocess_data,
    target_log,
)


def ridge_model(preprocessor):
    """Builds a Ridge Regression pipeline.

    Args:
        preprocessor (sklearn.compose.ColumnTransformer): Configured
            transformer for categorical and numerical features.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline combining preprocessing
            with Ridge Regression using alpha=1.0.
    """
    model = Ridge(alpha=1.0)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def run_experiment(train, val, config):
    """Trains and evaluates a Ridge Regression feature configuration.

    Applies the selected feature transformations to the training and
    validation datasets, preprocesses the selected features, and
    evaluates model performance using R² on the log-transformed target.

    Args:
        train (pandas.DataFrame): Training dataset.
        val (pandas.DataFrame): Validation dataset.
        config (dict): Feature-engineering options passed to
            preprocess_data() and cat_num_features().

    Returns:
        tuple[float, float]: Training R² score and validation R² score,
            respectively, calculated on the log-transformed target.

    Raises:
        ValueError: If required engineered features are missing from
            the training dataset or feature dependencies are invalid.
    """
    train_processed = preprocess_data(train, **config)
    val_processed = preprocess_data(val, **config)

    train_processed = target_log(
        train_processed,
        drop_original=True,
    )
    val_processed = target_log(
        val_processed,
        drop_original=True,
    )

    X_train, y_train, X_val, y_val = train_val_split(
        train_processed,
        val_processed,
    )

    # Match feature selection to the current experiment.
    feature_options = {
        "haversine_distance": config.get("haversine", False),
        "manhattan_distance": config.get("manhattan", False),
        "pickup_bearing": config.get("bearing", False),
        "airport": config.get("airport", False),
        "interactions": config.get("interactions", False),
        "heavy_hitters": config.get("heavy_hitters", False),
        "log_dist": config.get("log_dist", False),
        "cyclical_time_features": config.get("cyclical_time_features", False),
        "distance_relationship_features": config.get(
            "distance_relationship_features", False
        ),
    }

    categorical, numerical = cat_num_features(**feature_options)

    selected_features = categorical + numerical
    missing_features = set(selected_features) - set(X_train.columns)

    if missing_features:
        raise ValueError(
            f"Selected features missing from training data: {sorted(missing_features)}"
        )

    preprocessor = preprocessing(
        categorical,
        numerical,
        scaling="standard",
    )

    pipeline = ridge_model(preprocessor)
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    val_predictions = pipeline.predict(X_val)

    train_score = r2_score(y_train, train_predictions)
    val_score = r2_score(y_val, val_predictions)

    return train_score, val_score  # Return as a tuple


if __name__ == "__main__":
    train, val, _ = load_dataset(data_type="split")

    if train is None or val is None:
        raise RuntimeError("Failed to load the training and validation datasets.")

    train = drop_id_features(train)
    val = drop_id_features(val)

    experiments = {
        "Baseline": {},
        "Distance Features": {
            "haversine": True,
            "manhattan": True,
        },
        "Distance + Bearing + Airport": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
        },
        "Add Time Interactions": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
        },
        "Add Heavy Hitters": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
        },
        "Add Log Distances": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
        },
        "Add Cyclical Time Features": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
            "cyclical_time_features": True,
        },
        "Add Distance Relationship Features": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
            "cyclical_time_features": True,
            "distance_relationship_features": True,
        },
        "Grid Precision 1": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
            "cyclical_time_features": True,
            "distance_relationship_features": True,
            "grid_precision": 1,
        },
        "Grid Precision 3": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
            "cyclical_time_features": True,
            "distance_relationship_features": True,
            "grid_precision": 3,
        },
        "Grid Precision 4": {
            "haversine": True,
            "manhattan": True,
            "bearing": True,
            "airport": True,
            "interactions": True,
            "heavy_hitters": True,
            "log_dist": True,
            "cyclical_time_features": True,
            "distance_relationship_features": True,
            "grid_precision": 4,
        },
    }

    results = {}

    print("=" * 60)
    print("NYC TAXI TRIP DURATION - FEATURE EXPERIMENTS")
    print("=" * 60)

    for name, config in experiments.items():
        train_score, val_score = run_experiment(train, val, config)
        # Store both train and validation scores as a tuple
        results[name] = (train_score, val_score)

        print(f"{name:<35} Train R²: {train_score:.5f} | Val R²: {val_score:.5f}")

    print("=" * 60)

    # Find the configuration with the highest validation R² score
    best_experiment = max(results, key=lambda name: results[name][1])
    best_train_r2, best_val_r2 = results[best_experiment]

    print(f"Best configuration: {best_experiment}")
    print(f"Best train R²:      {best_train_r2:.5f}")
    print(f"Best validation R²: {best_val_r2:.5f}")


# ==============================================================================
#                       FEATURE ENGINEERING EXPERIMENT RESULTS
# ==============================================================================
# The following results were obtained using Ridge Regression (alpha=1.0)
# and R² evaluated on the log-transformed trip duration.
#
# Binary time indicators and the rush-hour indicator are treated as numerical
# features and passed through StandardScaler in the current preprocessing
# configuration when their corresponding feature options are enabled.
#
# 1. Baseline:
#    - Training R² Score : 0.06436
#    - Validation R² Score: 0.06645
#
# 2. Distance Features:
#    - Training R² Score : 0.35409
#    - Validation R² Score: 0.38955
#
# 3. Distance + Bearing + Airport:
#    - Training R² Score : 0.40152
#    - Validation R² Score: 0.42380
#
# 4. Add Time Interactions:
#    - Training R² Score : 0.40604
#    - Validation R² Score: 0.42549
#
# 5. Add Heavy Hitters:
#    - Training R² Score : 0.68603
#    - Validation R² Score: 0.66113
#
# 6. Add Log Distances:
#    - Training R² Score : 0.72558
#    - Validation R² Score: 0.71050
#
# 7. Add Cyclical Time Features:
#    - Training R² Score : 0.72555
#    - Validation R² Score: 0.71049
#
# 8. Add Distance Relationship Features:
#    - Training R² Score : 0.72825
#    - Validation R² Score: 0.71223
#
# 9. Grid Precision Experiments:
#    - Precision 1: Training R² = 0.66529 | Validation R² = 0.65999
#    - Precision 2: Training R² = 0.72825 | Validation R² = 0.71223
#    - Precision 3: Training R² = 0.89245 | Validation R² = 0.69296
#    - Precision 4: Training R² = 0.94799 | Validation R² = 0.64835
#
#    Note: Precision 2 uses the default configuration in the
#    "Add Distance Relationship Features" experiment, so these results
#    represent the same configuration rather than an independent run.
#
# DECISION AND NEXT STEPS:
# ------------------------
# Distance relationship features improved validation R² from 0.71050 to
# 0.71223, providing a small improvement over the previous configuration.
#
# Grid precision 2 achieved the highest validation R² among the tested
# precision settings. Increasing precision to 3 and 4 substantially
# increased training performance but reduced validation performance,
# indicating overfitting.
#
# Changing the treatment of binary indicators to numerical features did
# not materially change the overall results. The best validation R² remains
# 0.71223, achieved by the configuration with distance relationship
# features and the default grid precision of 2.
#
# Therefore, grid_precision=2 is retained as the default setting.
# The next step is to continue EDA and investigate additional patterns
# that may guide further feature engineering.
# ==============================================================================
