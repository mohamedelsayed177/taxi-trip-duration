"""
Data Helper Utilities for NYC Taxi Trip Duration Pipeline.

This module provides enterprise-grade data management utility functions, including 
asynchronous/synchronous dataset loading, structural data validation, pipeline 
splitting, feature categorization, and automated Scikit-Learn transformers configuration.

Supported Formats: Google Python Style Guide Docstrings.
"""

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


def load_dataset(data_type="split"):
    """Loads train, validation, and test datasets from the localized repository.

    Args:
        data_type (str, optional): The target directory to fetch data from. 
            Options are "split" (full set) or "split_sample" (mini set). 
            Defaults to "split".

    Returns:
        tuple: A tuple containing three pandas.DataFrame objects:
            (train_dataframe, validation_dataframe, test_dataframe).
            Returns (None, None, None) if an execution exception occurs.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(current_dir, "..", "..", "..", "dataset", data_type)

        train_path = os.path.join(base_path, "train.csv")
        val_path = os.path.join(base_path, "val.csv")
        test_path = os.path.join(base_path, "test.csv")

        print(f"Executing Data Ingestion from: dataset/{data_type}/")
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        return train, val, test

    except Exception as e:
        print(f"Critical Error during dataset ingestion: {e}")
        return None, None, None


def checking_data(df):
    """Performs integrity checks including shape inspection, missing values, and duplicates.

    Args:
        df (pandas.DataFrame): The target DataFrame to profile.
    """
    shape = df.shape
    missing_values = df.isna().sum().sum()
    duplicates = df.duplicated().sum()

    print(f"Data Structural Diagnostics:")
    print(f" - Matrix Shape       : {shape}")
    print(f" - Null Observations  : {missing_values}")
    print(f" - Duplicate Records  : {duplicates}")


def drop_id_features(df):
    """Drops high-cardinality non-predictive operational identifiers.

    Args:
        df (pandas.DataFrame): The input dataset containing target identifiers.

    Returns:
        pandas.DataFrame: A deep copy of the dataset isolated from 'id' and 'vendor_id'.
    """
    df = df.copy()
    df.drop(columns=["id", "vendor_id"], inplace=True)
    return df


def train_val_split(train, val):
    """Splits the historical dataframes into predictor matrices (X) and target vectors (y).

    Args:
        train (pandas.DataFrame): The preprocessed training dataset.
        val (pandas.DataFrame): The preprocessed validation dataset.

    Returns:
        tuple: (X_train, y_train, X_val, y_val) partitioned datasets.
    """
    X_train = train.drop(columns=["trip_duration"])
    y_train = train["trip_duration"]
    X_val = val.drop(columns=["trip_duration"])
    y_val = val["trip_duration"]

    return X_train, y_train, X_val, y_val


def cat_num_features(haversine_distance=False):
    """Isolates categorical variables from numerical variables for pipeline compliance.

    Args:
        haversine_distance (bool, optional): If True, dynamically injects the 
            engineered 'haversine_distance' into the continuous numerical array. 
            Defaults to False.

    Returns:
        tuple: Two lists containing column names: (categorical_features, numeric_features).
    """
    categorical_features = [
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day_of_year",
        "passenger_count",
        "store_and_fwd_flag"
    ]

    numeric_features = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude"
    ]
    
    if haversine_distance:
        numeric_features.append("haversine_distance")

    return categorical_features, numeric_features


def preprocessing(categorical_features, numerical_features, scaling="standard"):
    """Constructs a composite transformation block utilizing scikit-learn ColumnTransformer.

    Args:
        categorical_features (list): String tokens of categorical columns to encoder.
        numerical_features (list): String tokens of numerical columns to scale.
        scaling (str, optional): Scaling algorithm topology. Supported tokens: 
            "standard" (StandardScaler) or "minmax" (MinMaxScaler). Defaults to "standard".

    Returns:
        sklearn.compose.ColumnTransformer: Pre-configured mathematical scaling 
            and encoding subgraph block.
            
    Raises:
        ValueError: If an unsupported scaling strategy argument is supplied.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid continuous scaling strategy. Select 'standard' or 'minmax'.")

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numerical_transformer = Pipeline(
        steps=[("scaler", scaler)]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"  # Strict schema enforcement to prevent un-encoded string leakages
    )

    return preprocessor
