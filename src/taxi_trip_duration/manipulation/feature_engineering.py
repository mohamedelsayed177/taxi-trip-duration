import pandas as pd
import numpy as np


def target_log(df):
    """
    Applies log transformation to the 'trip_duration' target variable.
    """
    df = df.copy()
    df["trip_duration"] = np.log1p(df["trip_duration"])  # log1p to handle zero values
    return df


def date_time(df,eda=False):
    """
    Extracts date and time features from the 'pickup_datetime' column.
    If eda is True, it will also extract 'pickup_date' and 'pickup_time' for exploratory data analysis.
    """
    # Create a copy to avoid mutating the original DataFrame in place
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Extract structural numerical features
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month

    # Extract date and time features for EDA purposes
    if eda:
        df["pickup_date"] = df["pickup_datetime"].dt.date
        df["pickup_time"] = df["pickup_datetime"].dt.time
    else:
        df.drop(columns=["pickup_datetime"], inplace=True)  # drop the original datetime column

    return df


def haversine_distance(df):
    """
    Calculates the Haversine distance between pickup and dropoff coordinates.
    """
    df = df.copy()

    # Convert degrees to radians
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371
    df["haversine_distance"] = r * c

    return df


def preprocess_data(df):
    """
    Sequentially applies date_time and haversine_distance pipeline steps.
    """
    df = date_time(df)
    df = haversine_distance(df)
    return df
