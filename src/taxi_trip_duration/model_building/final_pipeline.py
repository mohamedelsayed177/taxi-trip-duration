"""
NYC Taxi Trip Duration - Final Training Pipeline.

Trains the final Ridge Regression model using the best feature 
configurations, evaluates it on the validation set, and saves the 
trained pipeline for future inference.
"""

import os
import joblib

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
    # Load and clean data
    train, val, _ = load_dataset(data_type="split")
    
    train = drop_id_features(train)
    val = drop_id_features(val)

    # Apply best feature engineering directly
    train = preprocess_data(
        train, haversine=True, manhattan=True, bearing=True, airport=True, 
        interactions=True, heavy_hitters=True, log_dist=True, 
        cyclical_time_features=True, distance_relationship_features=True
    )
    val = preprocess_data(
        val, haversine=True, manhattan=True, bearing=True, airport=True, 
        interactions=True, heavy_hitters=True, log_dist=True, 
        cyclical_time_features=True, distance_relationship_features=True
    )

    # Transform target
    train = target_log(train, drop_original=True)
    val = target_log(val, drop_original=True)

    X_train, y_train, X_val, y_val = train_val_split(train, val)

    # Select categorical and numerical features for the best config
    categorical, numerical = cat_num_features(
        haversine_distance=True, manhattan_distance=True, pickup_bearing=True, 
        airport=True, interactions=True, heavy_hitters=True, log_dist=True, 
        cyclical_time_features=True, distance_relationship_features=True
    )

    # Build, train, and evaluate pipeline
    preprocessor = preprocessing(categorical, numerical, scaling="standard")
    pipeline = ridge_model(preprocessor)
    pipeline.fit(X_train, y_train)

    train_score = r2_score(y_train, pipeline.predict(X_train))
    val_score = r2_score(y_val, pipeline.predict(X_val))

    print(f"Final Training R² Score:   {train_score:.5f}")
    print(f"Final Validation R² Score: {val_score:.5f}")

    # Save the trained model to the model folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the directory path first
    model_dir = os.path.join(current_dir, "..", "..", "..", "model")
    
    # Create the directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "ridge_model.pkl")
    joblib.dump(pipeline, model_path)
    
    print(f"Model successfully saved to '{model_path}'")
    
    
    
# Final Training R² Score:   0.72825
# Final Validation R² Score: 0.71223