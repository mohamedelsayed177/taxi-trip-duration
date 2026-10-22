
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

    predictions = pipeline.predict(X_val)

    return r2_score(y_val, predictions)


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
        score = run_experiment(train, val, config)
        results[name] = score

        print(f"{name:<35} Validation R²: {score:.5f}")

    print("=" * 60)

    best_experiment = max(results, key=results.get)

    print(f"Best configuration: {best_experiment}")
    print(f"Best validation R²: {results[best_experiment]:.5f}")




# ============================================================
# NYC TAXI TRIP DURATION - FEATURE EXPERIMENTS
# ============================================================
# Baseline                            Validation R²: 0.06645
# Distance Features                   Validation R²: 0.38955
# Distance + Bearing + Airport        Validation R²: 0.42380
# Add Time Interactions               Validation R²: 0.42549
# Add Heavy Hitters                   Validation R²: 0.66113
# Add Log Distances                   Validation R²: 0.71050
# Add Cyclical Time Features          Validation R²: 0.71050
# Full Configuration                  Validation R²: 0.71223
# ============================================================
# Best configuration: Full Configuration
# Best validation R²: 0.71223