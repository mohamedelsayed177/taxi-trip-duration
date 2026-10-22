"""
Feature Engineering and Mathematical Transformations for NYC Taxi Analysis.

This module provides target log transformation, datetime feature extraction,
and optional geographic feature calculations.
"""

import pandas as pd
import numpy as np


# Mean Earth radius in kilometers, used for geographic distance calculations
EARTH_RADIUS_KM = 6371.0


def target_log(df, drop_original=False):
    """Applies a log1p transformation to the trip duration target.

    Args:
        df (pandas.DataFrame): DataFrame containing 'trip_duration'.
        drop_original (bool, optional): If True, removes 'trip_duration'
            and retains 'log1p_trip_duration'. Defaults to False.

    Returns:
        pandas.DataFrame: Copy containing the log-transformed target.
    """
    df = df.copy()

    # Apply log1p to reduce the effect of right-skewed target values
    df["log1p_trip_duration"] = np.log1p(df["trip_duration"])

    if drop_original:
        df = df.drop(columns=["trip_duration"])

    return df


def date_time(df, eda=False):
    """Extracts datetime features from the pickup timestamp.

    Args:
        df (pandas.DataFrame): DataFrame containing 'pickup_datetime'.
        eda (bool, optional): If True, retains 'pickup_datetime' and adds
            'pickup_date' and 'pickup_time'. Otherwise, removes the original
            timestamp after extracting its components. Defaults to False.

    Returns:
        pandas.DataFrame: Copy augmented with datetime features.
    """
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Extract hour, weekday, month, and day of year
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day_of_year"] = df["pickup_datetime"].dt.dayofyear

    if eda:
        # Retain the timestamp and add date and time representations for EDA
        df["pickup_date"] = df["pickup_datetime"].dt.date
        df["pickup_time"] = df["pickup_datetime"].dt.time
    else:
        # Remove the original timestamp because its components are already extracted
        df.drop(columns=["pickup_datetime"], inplace=True)

    return df


def latitude_longitude_helper(df):
    """Converts pickup and dropoff coordinates from degrees to radians.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup and dropoff
            latitude and longitude columns.

    Returns:
        tuple: (pickup_latitude_rad, pickup_longitude_rad,
            dropoff_latitude_rad, dropoff_longitude_rad).
    """
    pickup_latitude_rad = np.radians(df["pickup_latitude"])
    pickup_longitude_rad = np.radians(df["pickup_longitude"])
    dropoff_latitude_rad = np.radians(df["dropoff_latitude"])
    dropoff_longitude_rad = np.radians(df["dropoff_longitude"])

    return (
        pickup_latitude_rad,
        pickup_longitude_rad,
        dropoff_latitude_rad,
        dropoff_longitude_rad,
    )


def haversine_distance(df):
    """Calculates the great-circle distance between pickup and dropoff points.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup and dropoff
            coordinates.

    Returns:
        pandas.DataFrame: Copy containing 'haversine_distance' in kilometers.
    """
    df = df.copy()

    # Convert the pickup and dropoff coordinates to radians
    (
        pickup_latitude_rad,
        pickup_longitude_rad,
        dropoff_latitude_rad,
        dropoff_longitude_rad,
    ) = latitude_longitude_helper(df)

    # Calculate coordinate differences in radians
    latitude_difference = dropoff_latitude_rad - pickup_latitude_rad
    longitude_difference = dropoff_longitude_rad - pickup_longitude_rad

    # Apply the Haversine formula to calculate the central angle
    haversine_a = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(pickup_latitude_rad)
        * np.cos(dropoff_latitude_rad)
        * np.sin(longitude_difference / 2) ** 2
    )

    # Bound the value to [0, 1] to avoid floating-point errors
    haversine_a = np.clip(haversine_a, 0, 1)
    central_angle = 2 * np.arcsin(np.sqrt(haversine_a))

    # Convert the central angle to a great-circle distance in kilometers
    df["haversine_distance"] = EARTH_RADIUS_KM * central_angle

    return df


def manhattan_distance(df):
    """Approximates Manhattan-style distance between pickup and dropoff points.

    The estimate sums the north-south distance and the east-west distance
    calculated at the mean latitude.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup and dropoff coordinates.

    Returns:
        pandas.DataFrame: Copy containing 'manhattan_distance' in kilometers.
    """
    df = df.copy()

    # Convert the pickup and dropoff coordinates to radians
    (
        pickup_latitude_rad,
        pickup_longitude_rad,
        dropoff_latitude_rad,
        dropoff_longitude_rad,
    ) = latitude_longitude_helper(df)

    # Calculate north-south distance along a meridian
    latitude_difference = dropoff_latitude_rad - pickup_latitude_rad
    latitude_distance = EARTH_RADIUS_KM * np.abs(latitude_difference)

    # Calculate the shortest longitude difference in radians
    longitude_difference = (dropoff_longitude_rad - pickup_longitude_rad + np.pi) % (
        2 * np.pi
    ) - np.pi

    # Approximate east-west distance at the mean latitude
    mean_latitude_rad = (pickup_latitude_rad + dropoff_latitude_rad) / 2

    longitude_distance = (
        EARTH_RADIUS_KM * np.cos(mean_latitude_rad) * np.abs(longitude_difference)
    )

    # Sum the two directional distances to estimate Manhattan-style distance
    df["manhattan_distance"] = latitude_distance + longitude_distance

    return df


def pickup_bearing(df):
    """Calculates the initial bearing from pickup to dropoff in degrees.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup and dropoff
            coordinates.

    Returns:
        pandas.DataFrame: Copy containing 'pickup_bearing' in degrees,
            ranging from -180 to 180.
    """
    df = df.copy()

    # Convert the pickup and dropoff coordinates to radians
    (
        pickup_latitude_rad,
        pickup_longitude_rad,
        dropoff_latitude_rad,
        dropoff_longitude_rad,
    ) = latitude_longitude_helper(df)

    # Calculate the longitude difference
    longitude_difference = dropoff_longitude_rad - pickup_longitude_rad

    # Calculate the components used to determine the initial bearing
    bearing_y = np.sin(longitude_difference) * np.cos(dropoff_latitude_rad)
    bearing_x = np.cos(pickup_latitude_rad) * np.sin(dropoff_latitude_rad) - np.sin(
        pickup_latitude_rad
    ) * np.cos(dropoff_latitude_rad) * np.cos(longitude_difference)

    # Convert the initial bearing from radians to degrees
    bearing_degrees = np.degrees(np.arctan2(bearing_y, bearing_x))
    df["pickup_bearing"] = bearing_degrees

    return df


def preprocess_data(
    df,
    eda=False,
    haversine=False,
    manhattan=False,
    bearing=False,
):
    """Extracts datetime features and optionally adds geographic features.

    Args:
        df (pandas.DataFrame): Input dataset containing pickup timestamps
            and geographic coordinates.
        eda (bool, optional): Controls retention of the original pickup
            timestamp and creation of date and time columns. Defaults to False.
        haversine (bool, optional): Adds 'haversine_distance'. Defaults to False.
        manhattan (bool, optional): Adds 'manhattan_distance'. Defaults to False.
        bearing (bool, optional): Adds 'pickup_bearing'. Defaults to False.

    Returns:
        pandas.DataFrame: Copy containing extracted datetime features and
            any requested geographic features.
    """
    # Extract datetime features first
    df = date_time(df, eda=eda)

    # Add optional geographic and seasonal features
    if haversine:
        df = haversine_distance(df)
    if manhattan:
        df = manhattan_distance(df)
    if bearing:
        df = pickup_bearing(df)

    return df
