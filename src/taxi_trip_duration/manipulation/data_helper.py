"""
Data Helper Utilities for NYC Taxi Trip Duration Pipeline.

This module provides dataset loading, data diagnostics, identifier removal,
target separation, feature categorization, and Scikit-Learn preprocessing.
"""

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def load_dataset(data_type="split"):
    """Loads the training, validation, and test datasets.

    Args:
        data_type (str, optional): Dataset directory to load.
            Options are "split" or "split_sample". Defaults to "split".

    Returns:
        tuple: (train_dataframe, validation_dataframe, test_dataframe).
            Returns (None, None, None) if loading fails.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(current_dir, "..", "..", "..", "dataset", data_type)

        train_path = os.path.join(base_path, "train.csv")
        val_path = os.path.join(base_path, "val.csv")
        test_path = os.path.join(base_path, "test.csv")

        print(f"Loading datasets from: dataset/{data_type}/")
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        return train, val, test

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None


def checking_data(df):
    """Prints the shape, missing-value count, and duplicate-row count.

    Args:
        df (pandas.DataFrame): DataFrame to inspect.
    """
    shape = df.shape
    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()

    print("Data Diagnostics:")
    print(f" - Shape              : {shape}")
    print(f" - Missing Values     : {missing_values}")
    print(f" - Duplicate Records  : {duplicate_rows}")


def drop_id_features(df):
    """Removes the identifier columns from a dataset.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id' and 'vendor_id'.

    Returns:
        pandas.DataFrame: Copy without the identifier columns.
    """
    df = df.copy()
    df.drop(columns=["id", "vendor_id"],errors="ignore",inplace=True)

    return df


def train_val_split(train, val):
    """Separates predictors and the log-transformed target.

    Args:
        train (pandas.DataFrame): Preprocessed training dataset.
        val (pandas.DataFrame): Preprocessed validation dataset.

    Returns:
        tuple: (X_train, y_train, X_val, y_val).
    """
    # Separate training predictors and target
    X_train = train.drop(columns=["log1p_trip_duration"])
    y_train = train["log1p_trip_duration"]

    # Separate validation predictors and target
    X_val = val.drop(columns=["log1p_trip_duration"])
    y_val = val["log1p_trip_duration"]

    return X_train, y_train, X_val, y_val


def cat_num_features(
    haversine_distance=False,
    manhattan_distance=False,
    pickup_bearing=False
):
    """Returns the categorical and numerical feature names for the model.

    Args:
        haversine_distance (bool, optional): Include 'haversine_distance'
            among numerical features. Defaults to False.
        manhattan_distance (bool, optional): Include 'manhattan_distance'
            among numerical features. Defaults to False.
        season (bool, optional): Include 'season' among categorical features.
            Defaults to False.
        pickup_bearing (bool, optional): Include 'pickup_bearing'
            among numerical features. Defaults to False.

    Returns:
        tuple: (categorical_features, numerical_features).
    """
    # Define the original categorical and numerical features
    categorical_features = [
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day_of_year",
        "passenger_count",
        "store_and_fwd_flag",
    ]

    numerical_features = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    # Include optional engineered features using their output column names
    if haversine_distance:
        numerical_features.append("haversine_distance")
    if manhattan_distance:
        numerical_features.append("manhattan_distance")
    if pickup_bearing:
        numerical_features.append("pickup_bearing")

    return categorical_features, numerical_features


def preprocessing(categorical_features, numerical_features, scaling="standard"):
    """Builds a preprocessing transformer for categorical and numerical features.

    Args:
        categorical_features (list): Categorical column names.
        numerical_features (list): Numerical column names.
        scaling (str, optional): Numerical scaling method: "standard" or
            "minmax". Defaults to "standard".

    Returns:
        sklearn.compose.ColumnTransformer: Configured preprocessing transformer.

    Raises:
        ValueError: If the scaling method is unsupported.
    """
    # Select the numerical scaling method
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling strategy. Select 'standard' or 'minmax'.")

    # Encode categorical features and scale numerical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    # Apply the appropriate transformer to each feature group
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
