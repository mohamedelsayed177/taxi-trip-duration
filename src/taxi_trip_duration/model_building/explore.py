"""
Note: I know that the best/fastest way to do these experiments is inside a notebook,
but I don't want to depend heavily on notebooks, so I'm doing it here in a .py file.
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
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])


if __name__ == "__main__":
    print("Loading Data...")
    train, val, _ = load_dataset(data_type="split")
    train = drop_id_features(train)
    val = drop_id_features(val)

    # Target transformation
    train = target_log(train, drop_original=True)
    val = target_log(val, drop_original=True)

    # Define the experiment configuration
    config = {
        "haversine": True,
        "manhattan": True,
        "bearing": True,
        "airport": True,
        "interactions": True,
        "heavy_hitters": True,
        "log_dist": True,
    }

    print(f"\nRunning Experiment with config: {config}")

    # Apply Feature Engineering
    train_processed = preprocess_data(train, **config)
    val_processed = preprocess_data(val, **config)

    # Split Features and Target
    X_train, y_train, X_val, y_val = train_val_split(train_processed, val_processed)

    #  Retrieve Categorical/Numerical column lists for the scaler
    categorical, numerical = cat_num_features(
        haversine_distance=config["haversine"],
        manhattan_distance=config["manhattan"],
        pickup_bearing=config["bearing"],
        airport=config["airport"],
        interactions=config["interactions"],
        heavy_hitters=config["heavy_hitters"],
        log_dist=config["log_dist"],
    )

    # Build and Train Model
    preprocessor = preprocessing(categorical, numerical, scaling="standard")
    pipeline = ridge_model(preprocessor)

    print("Training Ridge(alpha=1.0)...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_val)
    score = r2_score(y_val, preds)

    print("-" * 40)
    print(f"Validation R² Score: {score:.5f}")
    print("-" * 40)
