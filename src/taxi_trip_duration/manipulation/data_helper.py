import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


def load_dataset(data_type="split"):
    """
    load_dataset loads the dataset from the specified data_type directory.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        base_path = os.path.join(current_dir, "..", "..", "..", "dataset", data_type)

        train_path = os.path.join(base_path, "train.csv")
        val_path = os.path.join(base_path, "val.csv")
        test_path = os.path.join(base_path, "test.csv")

        print(f"Loading files from: dataset/{data_type}/")
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        return train, val, test

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None


def checking_data(df):
    """
    Checks for missing values and duplicates in the DataFrame.
    """
    shape = df.shape
    missing_values = df.isna().sum().sum()
    duplicates = df.duplicated().sum()

    print("DataFrame Shape:", shape)
    print("Missing Values:", missing_values)
    print("Number of Duplicates:", duplicates)


def drop_id_features(df):
    """
    Drops specified features from the DataFrame.
    """
    df = df.copy()
    df.drop(columns=["id", "vendor_id"], inplace=True)
    return df


def train_val_split(train, val):
    """
    Splits the train and validation DataFrames into features and target variable.
    """
    X_train = train.drop(columns=["trip_duration"])
    y_train = train["trip_duration"]
    X_val = val.drop(columns=["trip_duration"])
    y_val = val["trip_duration"]

    return X_train, y_train, X_val, y_val


def cat_num_features(df):
    """
    Identifies categorical and numerical features in the DataFrame.
    """
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_features, numerical_features


def preprocessing(categorical_features, numerical_features, scaling="standard"):
    """
    Creates a preprocessing pipeline for categorical and numerical features.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling option. Choose 'standard' or 'minmax'.")

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numerical_transformer = Pipeline(steps=[("scaler", scaler)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
