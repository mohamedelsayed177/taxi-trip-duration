"""
Feature Engineering and Mathematical Transformations for NYC Taxi Analysis.

This module houses deterministic mathematical transformations executed on the raw 
data inputs. Operations include skewed target normalization (logarithmic stabilization), 
temporal feature deconstruction, and geospatial coordinate matrix calculations.
"""

import pandas as pd
import numpy as np


def target_log(df):
    """Stabilizes variance of the skewed target via a log1p transformation.

    Args:
        df (pandas.DataFrame): Input dataframe containing raw 'trip_duration'.

    Returns:
        pandas.DataFrame: Modified dataframe with log-normalized 'trip_duration' values.
    """
    df = df.copy()
    df["trip_duration"] = np.log1p(df["trip_duration"])
    return df


def date_time(df, eda=False):
    """Deconstructs the raw pickup datetime string into multi-interval structural features.

    Args:
        df (pandas.DataFrame): Input dataframe with standard 'pickup_datetime'.
        eda (bool, optional): If True, retains human-readable date and time stamps 
            for exploratory graph analysis. If False, drops the original datetime 
            object to meet model input specifications. Defaults to False.

    Returns:
        pandas.DataFrame: Augmented dataframe with temporal discrete predictors.
    """
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Extract cyclical and structural discrete sub-intervals
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day_of_year"] = df["pickup_datetime"].dt.dayofyear

    if eda:
        df["pickup_date"] = df["pickup_datetime"].dt.date
        df["pickup_time"] = df["pickup_datetime"].dt.time
    else:
        df.drop(columns=["pickup_datetime"], inplace=True)

    return df


def haversine_distance(df):
    """Computes the great-circle geodesic distance between two coordinate pairs using radians.

    Args:
        df (pandas.DataFrame): Input dataframe containing pickup/dropoff 
            latitude and longitude fields.

    Returns:
        pandas.DataFrame: Copied dataframe with the new calculated 'haversine_distance' 
            column scaled in kilometers.
    """
    df = df.copy()

    # Convert mapping projection system degrees to Earth radians
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    # Core mathematical haversine formulation
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Constant earth volumetric mean radius vector (Kilometers)
    r = 6371
    df["haversine_distance"] = r * c

    return df


def preprocess_data(df, eda=False, haversine=False):
    """Orchestrates sequentially chained temporal and spatial extraction operations.

    Args:
        df (pandas.DataFrame): The raw input telemetry data matrix.
        eda (bool, optional): Passed directly to control date time string deletion. 
            Defaults to False.
        haversine (bool, optional): Flag controlling the explicit calculation and 
            injection of the geodesic haversine distance metric. Defaults to False.

    Returns:
        pandas.DataFrame: Fully processed model-ready predictor matrix.
    """
    df = date_time(df, eda=eda)
    if haversine:
        df = haversine_distance(df)
        
    return df
