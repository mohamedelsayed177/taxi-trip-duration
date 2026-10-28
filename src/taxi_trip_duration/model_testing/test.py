"""
NYC Taxi Trip Duration - Test Set Evaluation.

Loads the finalized Ridge Regression model and evaluates its performance
on the unseen test dataset.
"""

import os
import joblib
from sklearn.metrics import r2_score

from taxi_trip_duration.manipulation.data_helper import (
    drop_id_features,
    load_dataset,
)
from taxi_trip_duration.manipulation.feature_engineering import (
    preprocess_data,
    target_log,
)

if __name__ == "__main__":
    print("Loading test dataset...")
    # load_dataset returns train, val, test
    _, _, test = load_dataset(data_type="split")

    if test is None:
        raise RuntimeError("Failed to load the test dataset.")

    print("Preprocessing test data...")
    # Remove identifiers
    test = drop_id_features(test)

    # Apply the exact feature engineering used in final_pipeline.py
    test = preprocess_data(
        test, 
        haversine=True, 
        manhattan=True, 
        bearing=True, 
        airport=True, 
        interactions=True, 
        heavy_hitters=True, 
        log_dist=True, 
        cyclical_time_features=True, 
        distance_relationship_features=True
    )

    # Apply log1p transformation to the target
    test = target_log(test, drop_original=True)

    # Separate predictors and target (manually, as train_val_split is for train/val only)
    X_test = test.drop(columns=["log1p_trip_duration"])
    y_test = test["log1p_trip_duration"]

    # Locate and load the saved model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "..", "..", "model", "ridge_model.pkl")
    
    print(f"Loading trained model from: {model_path}")
    pipeline = joblib.load(model_path)

    print("Generating predictions and evaluating...")
    test_preds = pipeline.predict(X_test)
    test_score = r2_score(y_test, test_preds)

    print("-" * 50)
    print(f">> Final Test R² Score: {test_score:.5f}")
    print("-" * 50)
    
# --------------------------------------------------
# >> Final Test R² Score: 0.71629
# --------------------------------------------------