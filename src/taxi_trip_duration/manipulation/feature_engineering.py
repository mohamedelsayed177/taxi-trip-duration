"""
Feature Engineering and Mathematical Transformations for NYC Taxi Analysis.

This module provides target log transformation, datetime feature extraction,
and optional geographic and advanced temporal feature calculations.
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


def airport_distances(df):
    """Calculates the proximity of pickups and dropoffs to major NYC airports.

    Computes proximity vectors using the great-circle Haversine formula relative to
    John F. Kennedy (JFK), LaGuardia (LGA), and Newark Liberty (EWR) airports.

    Args:
        df (pandas.DataFrame): Input DataFrame containing geographical coordinates.

    Returns:
        pandas.DataFrame: A copy of the DataFrame augmented with 6 columns
            containing distances to specific airports in kilometers.
    """
    df = df.copy()

    # Major NYC Airport Coordinates (Latitude, Longitude)
    jfk_coords = (40.6413, -73.7781)
    lga_coords = (40.7769, -73.8740)
    ewr_coords = (40.6895, -74.1745)

    def distance_to_fixed_point(lat_series, lon_series, fixed_lat, fixed_lon):
        """Helper to calculate haversine distance to a specific coordinate."""
        lat1, lon1 = np.radians(lat_series), np.radians(lon_series)
        lat2, lon2 = np.radians(fixed_lat), np.radians(fixed_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        a = np.clip(a, 0, 1)
        return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

    # Distances from Pickup
    df["pickup_jfk_distance"] = distance_to_fixed_point(
        df["pickup_latitude"], df["pickup_longitude"], *jfk_coords
    )
    df["pickup_lga_distance"] = distance_to_fixed_point(
        df["pickup_latitude"], df["pickup_longitude"], *lga_coords
    )
    df["pickup_ewr_distance"] = distance_to_fixed_point(
        df["pickup_latitude"], df["pickup_longitude"], *ewr_coords
    )

    # Distances from Dropoff
    df["dropoff_jfk_distance"] = distance_to_fixed_point(
        df["dropoff_latitude"], df["dropoff_longitude"], *jfk_coords
    )
    df["dropoff_lga_distance"] = distance_to_fixed_point(
        df["dropoff_latitude"], df["dropoff_longitude"], *lga_coords
    )
    df["dropoff_ewr_distance"] = distance_to_fixed_point(
        df["dropoff_latitude"], df["dropoff_longitude"], *ewr_coords
    )

    return df


def time_interactions(df):
    """Creates rush hour flags and interacts them with calculated distances.

    Args:
        df (pandas.DataFrame): DataFrame containing datetime and distance features.

    Returns:
        pandas.DataFrame: Copy augmented with traffic proxy interactions.
    """
    df = df.copy()

    # Rush Hour definition: Weekdays (Mon-Fri -> 0-4) AND peak hours (7-10 AM, 4-7 PM)
    is_weekday = df["pickup_day_of_week"] < 5
    is_peak_hour = df["pickup_hour"].isin([7, 8, 9, 10, 16, 17, 18, 19])

    df["is_rush_hour"] = (is_weekday & is_peak_hour).astype(int)

    # Interact rush hour flag with spatial distances
    if "manhattan_distance" in df.columns:
        df["rush_hour_manhattan"] = df["is_rush_hour"] * df["manhattan_distance"]

    if "haversine_distance" in df.columns:
        df["rush_hour_haversine"] = df["is_rush_hour"] * df["haversine_distance"]

    return df


def heavy_hitter_categories(df):
    """Creates categorical features for frequently used pickup and drop-off areas.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup coordinates,
            drop-off coordinates, and pickup time features.

    Returns:
        pandas.DataFrame: Copy with geographic grid and time-slot categories.
    """
    df = df.copy()

    df["pickup_grid"] = (
        df["pickup_latitude"].round(2).astype(str)
        + "_"
        + df["pickup_longitude"].round(2).astype(str)
    )

    df["dropoff_grid"] = (
        df["dropoff_latitude"].round(2).astype(str)
        + "_"
        + df["dropoff_longitude"].round(2).astype(str)
    )

    df["day_hour"] = (
        df["pickup_day_of_week"].astype(str) + "_" + df["pickup_hour"].astype(str)
    )

    # Represent a pickup and drop-off area combination
    df["pickup_dropoff_grid"] = df["pickup_grid"] + "_to_" + df["dropoff_grid"]

    return df


def cyclical_time(df):
    """Creates cyclical encodings and time-period indicators.

    Args:
        df (pandas.DataFrame): DataFrame containing pickup_hour and
            pickup_day_of_week.

    Returns:
        pandas.DataFrame: Copy with cyclical time features and time flags.
    """
    df = df.copy()

    # Encode hour as a circular variable
    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

    # Encode day of week as a circular variable (Monday = 0)
    df["pickup_day_sin"] = np.sin(2 * np.pi * df["pickup_day_of_week"] / 7)
    df["pickup_day_cos"] = np.cos(2 * np.pi * df["pickup_day_of_week"] / 7)

    # Create time-period indicators
    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["pickup_hour"] >= 22) | (df["pickup_hour"] < 6)).astype(int)

    is_weekday = df["pickup_day_of_week"] < 5

    df["is_morning_peak"] = (is_weekday & df["pickup_hour"].between(7, 10)).astype(int)

    df["is_evening_peak"] = (is_weekday & df["pickup_hour"].between(16, 19)).astype(int)

    return df


def distance_relationships(df):
    """Creates features describing the relationship between trip distances.

    Args:
        df (pandas.DataFrame): DataFrame containing haversine_distance
            and manhattan_distance.

    Returns:
        pandas.DataFrame: Copy with distance ratio and distance difference.
    """
    df = df.copy()

    df["distance_ratio"] = df["manhattan_distance"] / (df["haversine_distance"] + 1e-6)

    df["distance_difference"] = df["manhattan_distance"] - df["haversine_distance"]

    return df


def log_distances(df):
    """Applies a log1p transformation to geographic distance metrics.

    Reduces the right-skewness of raw distance features to better align their
    variance with log-transformed target variables (e.g., log-trip duration).

    Args:
        df (pandas.DataFrame): Input DataFrame optionally containing
            'haversine_distance' and 'manhattan_distance' columns.

    Returns:
        pandas.DataFrame: A shallow copy of the DataFrame with conditionally
            added 'log_haversine' and 'log_manhattan' columns.
    """
    df = df.copy()

    if "haversine_distance" in df.columns:
        df["log_haversine"] = np.log1p(df["haversine_distance"])

    if "manhattan_distance" in df.columns:
        df["log_manhattan"] = np.log1p(df["manhattan_distance"])

    return df


def preprocess_data(
    df,
    eda=False,
    haversine=False,
    manhattan=False,
    bearing=False,
    airport=False,
    interactions=False,
    heavy_hitters=False,
    log_dist=False,
    cyclical_time_features=False,
    distance_relationship_features=False,
):
    """Applies datetime and optional geographic feature transformations.

    Args:
        df (pandas.DataFrame): Input dataset.
        eda (bool, optional): Retains original datetime components for EDA.
            Defaults to False.
        haversine (bool, optional): Adds Haversine distance.
            Defaults to False.
        manhattan (bool, optional): Adds Manhattan distance.
            Defaults to False.
        bearing (bool, optional): Adds pickup-to-drop-off bearing.
            Defaults to False.
        airport (bool, optional): Adds distances to nearby airports.
            Defaults to False.
        interactions (bool, optional): Adds rush-hour interaction features.
            Defaults to False.
        heavy_hitters (bool, optional): Adds geographic grid and time-slot
            categories, including pickup-drop-off grid pairs.
            Defaults to False.
        log_dist (bool, optional): Adds log-scaled distance features.
            Defaults to False.
        cyclical_time_features (bool, optional): Adds cyclical time encodings
            and time-period indicators. Defaults to False.
        distance_relationship_features (bool, optional): Adds distance ratio
            and difference features. Defaults to False.

    Returns:
        pandas.DataFrame: Transformed dataset.
    """
    if distance_relationship_features and not (haversine and manhattan):
        raise ValueError(
            "Distance relationship features require both "
            "Haversine and Manhattan distances."
        )

    df = date_time(df, eda=eda)

    # Create optional geographic features
    if haversine:
        df = haversine_distance(df)

    if manhattan:
        df = manhattan_distance(df)

    if bearing:
        df = pickup_bearing(df)

    if airport:
        df = airport_distances(df)

    if interactions:
        df = time_interactions(df)

    if heavy_hitters:
        df = heavy_hitter_categories(df)

    if log_dist:
        df = log_distances(df)

    # Create additional time and distance features
    if cyclical_time_features:
        df = cyclical_time(df)

    if distance_relationship_features:
        # Both distance columns must already exist
        df = distance_relationships(df)

    return df
