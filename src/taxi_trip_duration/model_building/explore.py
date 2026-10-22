
"""
NYC Taxi Trip Duration - Feature Engineering Experiments.

Evaluates cumulative feature-engineering configurations using Ridge
Regression and validation R² on the log-transformed trip duration.
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
    """Builds a Ridge Regression pipeline.

    Args:
        preprocessor: Configured feature preprocessing transformer.

    Returns:
        A scikit-learn pipeline containing preprocessing and Ridge.
    """
    model = Ridge(alpha=1.0)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def run_experiment(train, val, config):
    """Trains and evaluates one feature configuration.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        config: Feature-engineering options for preprocess_data.

    Returns:
        Validation R² score on the log-transformed target.
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
        "cyclical_time_features": config.get(
            "cyclical_time_features", False
        ),
        "distance_relationship_features": config.get(
            "distance_relationship_features", False
        ),
    }

    categorical, numerical = cat_num_features(**feature_options)

    selected_features = categorical + numerical
    missing_features = set(selected_features) - set(X_train.columns)

    if missing_features:
        raise ValueError(
            f"Selected features missing from training data: "
            f"{sorted(missing_features)}"
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

    return train_score, val_score # Return as a tuple


if __name__ == "__main__":
    train, val, _ = load_dataset(data_type="split")

    if train is None or val is None:
        raise RuntimeError(
            "Failed to load the training and validation datasets."
        )

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
        "Full Configuration": {
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




# ============================================================
# NYC TAXI TRIP DURATION - FEATURE EXPERIMENTS
# ============================================================
# Baseline                            Train R²: 0.06436 | Val R²: 0.06645
# Distance Features                   Train R²: 0.35409 | Val R²: 0.38955
# Distance + Bearing + Airport        Train R²: 0.40152 | Val R²: 0.42380
# Add Time Interactions               Train R²: 0.40604 | Val R²: 0.42549
# Add Heavy Hitters                   Train R²: 0.68603 | Val R²: 0.66113
# Add Log Distances                   Train R²: 0.72561 | Val R²: 0.71050
# Add Cyclical Time Features          Train R²: 0.72554 | Val R²: 0.71050
# Full Configuration                  Train R²: 0.72809 | Val R²: 0.71223
# ============================================================
# Best configuration: Full Configuration
# Best train R²:      0.72809
# Best validation R²: 0.71223