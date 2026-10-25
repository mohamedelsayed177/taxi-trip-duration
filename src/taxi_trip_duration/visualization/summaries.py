"""Numerical summary utilities for NYC Taxi EDA."""

import numpy as np
import pandas as pd
from IPython.display import display


def summarize_distance_duration(df, n_bins=10):
    """Summarize trip duration across geographic distance bins.

    Args:
        df: DataFrame containing trip duration and distance features.
        n_bins: Number of approximately equal-frequency distance bins.

    Returns:
        Dictionary containing summaries for both distance measures.
    """
    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]
    required_columns = ["trip_duration", *distance_columns]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid observations are available.")

    if (data < 0).any().any():
        raise ValueError("Duration and distance values must be non-negative.")

    summaries = {}

    for column in distance_columns:
        if data[column].nunique() < 2:
            raise ValueError(f"At least two distinct values are required for {column}.")

        distance_bins = pd.qcut(
            data[column],
            q=n_bins,
            duplicates="drop",
        )

        grouped = data.groupby(distance_bins, observed=True)["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        summaries[column] = summary.reset_index(names="distance_bin")

    return summaries


def display_distance_duration_summary(df, n_bins=10):
    """Display trip-duration summaries for both distance measures."""
    summaries = summarize_distance_duration(df, n_bins=n_bins)

    for column, summary in summaries.items():
        print(f"\n{column.replace('_', ' ').title()}")
        display(summary.round(2))


def summarize_distance_distributions(df):
    """Summarize the distributions of geographic distance features.

    Args:
        df: DataFrame containing the two distance features.

    Returns:
        DataFrame containing counts, percentiles, and maximum distances.

    Raises:
        KeyError: If a required distance column is missing.
        ValueError: If no valid observations remain or distances are negative.
    """
    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]

    missing_columns = [
        column for column in distance_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    records = []

    for column in distance_columns:
        distances = df[column].replace([np.inf, -np.inf], np.nan).dropna()

        if distances.empty:
            raise ValueError(f"No valid observations for {column}.")

        if (distances < 0).any():
            raise ValueError(f"Negative distance values found in {column}.")

        p99 = distances.quantile(0.99)

        records.append(
            {
                "distance_feature": column,
                "valid_count": len(distances),
                "zero_count": int((distances == 0).sum()),
                "median_km": distances.median(),
                "p95_km": distances.quantile(0.95),
                "p99_km": p99,
                "max_km": distances.max(),
                "above_p99_count": int((distances > p99).sum()),
            }
        )

    return pd.DataFrame(records).set_index("distance_feature")


def summarize_log_distance_distributions(df):
    """Compare distance distribution statistics before and after log1p.

    Args:
        df: DataFrame containing both geographic distance features.

    Returns:
        DataFrame of distribution statistics on original and log1p scales.

    Raises:
        KeyError: If a required distance column is missing.
        ValueError: If no valid data remains or distances are negative.
    """
    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]

    missing_columns = [
        column for column in distance_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[distance_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid distance observations are available.")

    if (data < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    records = []

    for column in distance_columns:
        original = data[column]
        transformed = np.log1p(original)

        for scale, values in [
            ("Original", original),
            ("Log1p", transformed),
        ]:
            records.append(
                {
                    "distance_feature": column,
                    "scale": scale,
                    "valid_count": len(values),
                    "median": values.median(),
                    "p95": values.quantile(0.95),
                    "p99": values.quantile(0.99),
                    "maximum": values.max(),
                    "skewness": values.skew(),
                }
            )

    return pd.DataFrame(records).set_index(["distance_feature", "scale"])


def summarize_temporal_patterns(df):
    """Summarize trip duration by pickup hour, weekday, and month.

    Args:
        df: DataFrame containing temporal features and trip duration.

    Returns:
        Dictionary of summary DataFrames for hours, weekdays, and months.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no valid observations remain or values are invalid.
    """
    required_columns = [
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "trip_duration",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid temporal observations are available.")

    if (data["trip_duration"] < 0).any():
        raise ValueError("Trip durations must be non-negative.")

    if not data["pickup_hour"].between(0, 23).all():
        raise ValueError("Pickup hours must be between 0 and 23.")

    if not data["pickup_day_of_week"].between(0, 6).all():
        raise ValueError("Weekdays must be between 0 and 6.")

    if not data["pickup_month"].between(1, 12).all():
        raise ValueError("Pickup months must be between 1 and 12.")

    def summarize_by(group_column):
        grouped = data.groupby(group_column)["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        return summary

    by_hour = summarize_by("pickup_hour")
    by_day = summarize_by("pickup_day_of_week")
    by_month = summarize_by("pickup_month")

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    by_day.index = [day_names[int(day)] for day in by_day.index]
    by_month.index = [month_names[int(month) - 1] for month in by_month.index]

    return {
        "by_hour": by_hour,
        "by_day_of_week": by_day,
        "by_month": by_month,
    }


def display_temporal_summary(summaries):
    """Display temporal summary tables for hours, weekdays, and months."""
    sections = [
        ("Trip Duration by Pickup Hour", "by_hour"),
        ("Trip Duration by Day of the Week", "by_day_of_week"),
        ("Trip Duration by Month", "by_month"),
    ]

    for title, key in sections:
        print(f"\n{title}")
        display(summaries[key].round(2))


def summarize_weekday_weekend(df):
    """Summarize trip duration for weekdays and weekends.

    Monday through Friday are classified as weekdays, while Saturday
    and Sunday are classified as weekends.

    Args:
        df: DataFrame containing pickup_day_of_week and trip_duration.

    Returns:
        DataFrame containing trip counts, median durations, and
        90th-percentile durations for weekdays and weekends.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no valid observations remain or values are invalid.
    """
    required_columns = ["pickup_day_of_week", "trip_duration"]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if data.empty:
        raise ValueError("No valid observations are available.")

    if (data["trip_duration"] < 0).any():
        raise ValueError("Trip durations must be non-negative.")

    if not data["pickup_day_of_week"].between(0, 6).all():
        raise ValueError("Weekdays must be represented by values from 0 to 6.")

    data["day_type"] = np.where(
        data["pickup_day_of_week"].isin([5, 6]),
        "Weekend",
        "Weekday",
    )

    grouped = data.groupby("day_type")["trip_duration"]

    summary = grouped.agg(
        trip_count="size",
        median_duration_seconds="median",
    )

    summary["p90_duration_seconds"] = grouped.quantile(0.90)

    summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
    summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

    return summary.reindex(["Weekday", "Weekend"])


def summarize_rush_hour_patterns(df):
    """Summarize trip duration by the existing rush-hour indicator.

    Args:
        df: DataFrame containing is_rush_hour and trip_duration.

    Returns:
        DataFrame containing counts, dataset shares, and duration statistics.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no valid observations remain or values are invalid.
    """
    required_columns = ["is_rush_hour", "trip_duration"]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if data.empty:
        raise ValueError("No valid observations are available.")

    if not data["is_rush_hour"].isin([0, 1]).all():
        raise ValueError("is_rush_hour must contain only 0 and 1.")

    if (data["trip_duration"] < 0).any():
        raise ValueError("Trip durations must be non-negative.")

    data["rush_hour_group"] = data["is_rush_hour"].map(
        {
            0: "Non-Rush Hour",
            1: "Rush Hour",
        }
    )

    grouped = data.groupby("rush_hour_group")["trip_duration"]

    summary = grouped.agg(
        trip_count="size",
        mean_duration_seconds="mean",
        median_duration_seconds="median",
    )

    summary["p90_duration_seconds"] = grouped.quantile(0.90)

    summary["trip_share_pct"] = summary["trip_count"] / len(data) * 100

    summary["mean_duration_minutes"] = summary.pop("mean_duration_seconds") / 60
    summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
    summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

    return summary.reindex(["Non-Rush Hour", "Rush Hour"])


def summarize_cyclical_encodings(df):
    """Summarize distances between neighboring cyclical encodings.

    Args:
        df: DataFrame containing the time categories and their encodings.

    Returns:
        DataFrame comparing ordinary and wraparound encoded distances.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If categories are missing or encoded values are invalid.
    """
    configurations = [
        (
            "Pickup Hour",
            "pickup_hour",
            "pickup_hour_sin",
            "pickup_hour_cos",
            24,
        ),
        (
            "Day of Week",
            "pickup_day_of_week",
            "pickup_day_sin",
            "pickup_day_cos",
            7,
        ),
    ]

    required_columns = [
        column
        for _, category, sine, cosine, _ in configurations
        for column in [category, sine, cosine]
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    records = []

    for name, category, sine, cosine, period in configurations:
        data = df[[category, sine, cosine]].replace([np.inf, -np.inf], np.nan).dropna()

        if not data[category].isin(range(period)).all():
            raise ValueError(f"Invalid category values found for {name}.")

        encoded = data.groupby(category)[[sine, cosine]].mean().reindex(range(period))

        if encoded.isna().any().any():
            raise ValueError(f"Some categories are missing for {name}.")

        coordinates = encoded[[cosine, sine]].to_numpy()

        # Distance from each category to the next, including wraparound.
        next_coordinates = np.roll(coordinates, -1, axis=0)
        distances = np.linalg.norm(
            next_coordinates - coordinates,
            axis=1,
        )

        regular_distances = distances[:-1]
        regular_distance = distances[0]
        wraparound_distance = distances[-1]

        records.append(
            {
                "time_cycle": name,
                "category_count": period,
                "mean_neighbor_distance": regular_distances.mean(),
                "min_neighbor_distance": regular_distances.min(),
                "max_neighbor_distance": regular_distances.max(),
                "first_to_second_distance": regular_distance,
                "last_to_first_distance": wraparound_distance,
                "wraparound_to_regular_ratio": (
                    wraparound_distance / regular_distance
                    if regular_distance > 0
                    else np.nan
                ),
                "raw_regular_gap": 1,
                "raw_wraparound_gap": period - 1,
            }
        )

    return pd.DataFrame(records).set_index("time_cycle")


def summarize_location_coverage(df):
    """Summarize coordinate coverage within the geographic plotting window.

    Args:
        df: DataFrame containing pickup and drop-off coordinates.

    Returns:
        DataFrame containing valid coordinate counts and geographic
        window coverage for pickup and drop-off locations.

    Raises:
        KeyError: If required coordinate columns are missing.
    """
    coordinate_pairs = [
        ("Pickup", "pickup_longitude", "pickup_latitude"),
        ("Drop-off", "dropoff_longitude", "dropoff_latitude"),
    ]

    required_columns = [
        column
        for _, longitude, latitude in coordinate_pairs
        for column in [longitude, latitude]
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    records = []

    for location, longitude, latitude in coordinate_pairs:
        coordinates = (
            df[[longitude, latitude]].replace([np.inf, -np.inf], np.nan).dropna()
        )

        within_window = coordinates[longitude].between(-74.5, -73.3) & coordinates[
            latitude
        ].between(40.4, 41.2)

        valid_count = len(coordinates)
        within_count = int(within_window.sum())

        records.append(
            {
                "location": location,
                "valid_coordinate_pairs": valid_count,
                "within_window_count": within_count,
                "within_window_pct": (
                    within_count / valid_count * 100 if valid_count else np.nan
                ),
                "outside_window_count": valid_count - within_count,
            }
        )

    return pd.DataFrame(records).set_index("location")


def summarize_spatial_grid_patterns(df, top_n=15):
    """Summarize trip duration for the most frequent spatial grid cells.

    Args:
        df: DataFrame containing grid features and trip duration.
        top_n: Number of most frequent grid cells to return per endpoint.

    Returns:
        Dictionary containing pickup-grid and drop-off-grid summaries.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If top_n is invalid or no valid observations remain.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    grid_columns = {
        "pickup_grid": "Pickup Grid",
        "dropoff_grid": "Drop-off Grid",
    }

    required_columns = [*grid_columns, "trip_duration"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    summaries = {}

    for column, label in grid_columns.items():
        data = df[[column, "trip_duration"]].replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            raise ValueError(f"No valid observations available for {label}.")

        if (data["trip_duration"] < 0).any():
            raise ValueError("Trip durations must be non-negative.")

        grouped = data.groupby(column, observed=True)["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            mean_duration_seconds="mean",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["trip_share_pct"] = summary["trip_count"] / len(data) * 100
        summary["mean_duration_minutes"] = summary.pop("mean_duration_seconds") / 60
        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        summary = (
            summary.sort_values("trip_count", ascending=False)
            .head(top_n)
            .rename_axis("grid_cell")
            .reset_index()
        )

        summaries[column] = summary

    return summaries


def summarize_extreme_distance_coordinates(df, percentile=99, top_n=10):
    """Summarize extreme distances and their geographic coordinates.

    Args:
        df: DataFrame containing Haversine distance and endpoint coordinates.
        percentile: Percentile used to define the extreme-distance threshold.
        top_n: Number of largest-distance trips to return for inspection.

    Returns:
        Dictionary containing an overall comparison table and the largest-
        distance observations with their coordinates.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If arguments are invalid or no valid data remains.
    """
    required_columns = [
        "haversine_distance",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if data.empty:
        raise ValueError("No valid distance and coordinate records remain.")

    if (data["haversine_distance"] < 0).any():
        raise ValueError("Haversine distances must be non-negative.")

    if "id" in df.columns:
        data["id"] = df.loc[data.index, "id"]

    threshold = data["haversine_distance"].quantile(percentile / 100)

    pickup_inside = data["pickup_longitude"].between(-74.5, -73.3) & data[
        "pickup_latitude"
    ].between(40.4, 41.2)

    dropoff_inside = data["dropoff_longitude"].between(-74.5, -73.3) & data[
        "dropoff_latitude"
    ].between(40.4, 41.2)

    data["pickup_outside_window"] = ~pickup_inside
    data["dropoff_outside_window"] = ~dropoff_inside

    groups = {
        "All valid trips": data,
        f"Above {percentile}th percentile": data.loc[
            data["haversine_distance"] > threshold
        ],
    }

    records = []

    for group_name, group in groups.items():
        count = len(group)

        records.append(
            {
                "group": group_name,
                "trip_count": count,
                "share_of_valid_trips_pct": count / len(data) * 100,
                "distance_threshold_km": threshold,
                "median_haversine_km": group["haversine_distance"].median(),
                "maximum_haversine_km": group["haversine_distance"].max(),
                "pickup_outside_count": int(group["pickup_outside_window"].sum()),
                "pickup_outside_pct": (
                    group["pickup_outside_window"].mean() * 100 if count else np.nan
                ),
                "dropoff_outside_count": int(group["dropoff_outside_window"].sum()),
                "dropoff_outside_pct": (
                    group["dropoff_outside_window"].mean() * 100 if count else np.nan
                ),
                "either_endpoint_outside_count": int(
                    (
                        group["pickup_outside_window"] | group["dropoff_outside_window"]
                    ).sum()
                ),
                "either_endpoint_outside_pct": (
                    (
                        group["pickup_outside_window"] | group["dropoff_outside_window"]
                    ).mean()
                    * 100
                    if count
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(records).set_index("group")

    example_columns = [
        "haversine_distance",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_outside_window",
        "dropoff_outside_window",
    ]

    if "id" in data.columns:
        example_columns.insert(0, "id")

    top_examples = data.nlargest(top_n, "haversine_distance")[
        example_columns
    ].reset_index(drop=True)

    return {
        "summary": summary,
        "top_examples": top_examples,
    }


def summarize_airport_proximity(df):
    """Summarize trip duration across airport-distance bands.

    Args:
        df: DataFrame containing trip duration and six airport-distance
            features.

    Returns:
        Dictionary of summary DataFrames, one for each airport-distance
        feature. Each table contains trip counts and duration statistics.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no valid observations remain or values are invalid.
    """
    distance_columns = [
        "pickup_jfk_distance",
        "pickup_lga_distance",
        "pickup_ewr_distance",
        "dropoff_jfk_distance",
        "dropoff_lga_distance",
        "dropoff_ewr_distance",
    ]

    required_columns = ["trip_duration", *distance_columns]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    distance_bins = [0, 2, 5, 10, 20, np.inf]
    distance_labels = [
        "0–2 km",
        ">2–5 km",
        ">5–10 km",
        ">10–20 km",
        ">20 km",
    ]

    summaries = {}

    for column in distance_columns:
        data = df[[column, "trip_duration"]].replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            raise ValueError(f"No valid observations for {column}.")

        if (data[column] < 0).any():
            raise ValueError(f"Negative distances found in {column}.")

        if (data["trip_duration"] < 0).any():
            raise ValueError("Trip durations must be non-negative.")

        data = data.copy()

        data["distance_band"] = pd.cut(
            data[column],
            bins=distance_bins,
            labels=distance_labels,
            include_lowest=True,
        )

        grouped = data.groupby(
            "distance_band",
            observed=True,
        )["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            mean_duration_seconds="mean",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["trip_share_pct"] = summary["trip_count"] / len(data) * 100

        summary["mean_duration_minutes"] = summary.pop("mean_duration_seconds") / 60
        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        summaries[column] = summary

    return summaries


def summarize_bearing_distance_patterns(df):
    """Summarize trip directions, distances, and identical endpoints.

    Args:
        df: DataFrame containing pickup bearing, geographic distances,
            pickup and drop-off coordinates, and trip duration.

    Returns:
        Dictionary containing overall statistics and summaries by
        compass sector.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If no valid observations remain or values are invalid.
    """
    coordinate_columns = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]
    required_columns = [
        *coordinate_columns,
        *distance_columns,
        "pickup_bearing",
        "trip_duration",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if data.empty:
        raise ValueError("No valid observations are available.")

    if (data[distance_columns] < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    if (data["trip_duration"] < 0).any():
        raise ValueError("Trip durations must be non-negative.")

    if not data["pickup_bearing"].between(-180, 180).all():
        raise ValueError("Pickup bearing values must be between -180 and 180.")

    identical_endpoints = data["pickup_longitude"].eq(data["dropoff_longitude"]) & data[
        "pickup_latitude"
    ].eq(data["dropoff_latitude"])

    identical_data = data.loc[identical_endpoints]
    total_count = len(data)

    overview = pd.DataFrame(
        [
            {
                "valid_trip_count": total_count,
                "identical_endpoint_count": int(identical_endpoints.sum()),
                "identical_endpoint_pct": (identical_endpoints.mean() * 100),
                "zero_haversine_count": int(data["haversine_distance"].eq(0).sum()),
                "zero_manhattan_count": int(data["manhattan_distance"].eq(0).sum()),
                "haversine_median_km": data["haversine_distance"].median(),
                "haversine_p95_km": data["haversine_distance"].quantile(0.95),
                "haversine_p99_km": data["haversine_distance"].quantile(0.99),
                "haversine_max_km": data["haversine_distance"].max(),
                "manhattan_median_km": data["manhattan_distance"].median(),
                "manhattan_p95_km": data["manhattan_distance"].quantile(0.95),
                "manhattan_p99_km": data["manhattan_distance"].quantile(0.99),
                "manhattan_max_km": data["manhattan_distance"].max(),
                "identical_endpoint_median_duration_min": (
                    identical_data["trip_duration"].median() / 60
                    if not identical_data.empty
                    else np.nan
                ),
                "identical_endpoint_p90_duration_min": (
                    identical_data["trip_duration"].quantile(0.90) / 60
                    if not identical_data.empty
                    else np.nan
                ),
            }
        ]
    )

    # Identical endpoints have no meaningful geographic direction.
    bearing_data = data.loc[~identical_endpoints].copy()

    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    bearing_360 = bearing_data["pickup_bearing"] % 360
    sector_indices = np.floor((bearing_360 + 22.5) / 45).astype(int) % 8

    bearing_data["bearing_sector"] = pd.Categorical(
        [sectors[index] for index in sector_indices],
        categories=sectors,
        ordered=True,
    )

    grouped = bearing_data.groupby(
        "bearing_sector",
        observed=True,
    )

    by_sector = grouped.agg(
        trip_count=("trip_duration", "size"),
        median_haversine_km=("haversine_distance", "median"),
        median_manhattan_km=("manhattan_distance", "median"),
        median_duration_seconds=("trip_duration", "median"),
    )

    by_sector["p90_duration_seconds"] = grouped["trip_duration"].quantile(0.90)

    by_sector["trip_share_pct"] = (
        by_sector["trip_count"] / len(bearing_data) * 100
        if not bearing_data.empty
        else np.nan
    )

    by_sector["median_duration_minutes"] = by_sector.pop("median_duration_seconds") / 60
    by_sector["p90_duration_minutes"] = by_sector.pop("p90_duration_seconds") / 60

    by_sector = by_sector.reindex(sectors)
    by_sector["trip_count"] = by_sector["trip_count"].fillna(0).astype(int)

    return {
        "overview": overview,
        "by_bearing_sector": by_sector,
    }


def summarize_categorical_duration(df, top_n=15):
    """Summarize trip duration across selected categorical features.

    Args:
        df: DataFrame containing categorical features and trip duration.
        top_n: Number of most frequent day_hour categories to return.

    Returns:
        Dictionary containing summary DataFrames for vendor_id,
        store_and_fwd_flag, and the most frequent day_hour categories.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If top_n is invalid or no valid observations remain.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    categorical_columns = [
        "vendor_id",
        "store_and_fwd_flag",
        "day_hour",
    ]

    required_columns = [*categorical_columns, "trip_duration"]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    summaries = {}

    for column in categorical_columns:
        data = df[[column, "trip_duration"]].replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            raise ValueError(f"No valid observations available for {column}.")

        if (data["trip_duration"] < 0).any():
            raise ValueError("Trip durations must be non-negative.")

        grouped = data.groupby(column, observed=True)["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            mean_duration_seconds="mean",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["trip_share_pct"] = summary["trip_count"] / len(data) * 100
        summary["mean_duration_minutes"] = summary.pop("mean_duration_seconds") / 60
        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        if column == "day_hour":
            summary = summary.sort_values("trip_count", ascending=False).head(top_n)
        else:
            summary = summary.sort_index()

        summaries[column] = summary

    return summaries


def summarize_spatial_category_sparsity(df, top_n=15, rare_threshold=10):
    """Summarize category frequencies for spatial grid features.

    Args:
        df: DataFrame containing spatial grid category columns.
        top_n: Number of most frequent categories to return per feature.
        rare_threshold: Category counts below this value are considered
            infrequent for the purpose of the summary.

    Returns:
        Dictionary containing overall sparsity statistics, frequency
        bands, and the most frequent categories for each spatial feature.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If arguments are invalid or no valid categories remain.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if rare_threshold < 1:
        raise ValueError("rare_threshold must be at least 1.")

    grid_columns = [
        "pickup_grid",
        "dropoff_grid",
        "pickup_dropoff_grid",
    ]

    missing_columns = [column for column in grid_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    frequency_band_labels = [
        "1–9",
        "10–49",
        "50–99",
        "100–499",
        "500+",
    ]

    frequency_band_edges = [0, 9, 49, 99, 499, np.inf]

    overview_records = []
    frequency_records = []
    top_categories = {}

    for column in grid_columns:
        categories = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if categories.empty:
            raise ValueError(f"No valid categories are available for {column}.")

        counts = categories.value_counts()
        total_observations = len(categories)

        below_threshold = counts < rare_threshold
        below_50 = counts < 50

        observations_below_threshold = int(counts[below_threshold].sum())
        observations_below_50 = int(counts[below_50].sum())

        overview_records.append(
            {
                "category_feature": column,
                "valid_observations": total_observations,
                "unique_categories": len(counts),
                "single_observation_categories": int((counts == 1).sum()),
                "median_category_frequency": counts.median(),
                "p90_category_frequency": counts.quantile(0.90),
                "maximum_category_frequency": counts.max(),
                "categories_below_threshold": int(below_threshold.sum()),
                "observations_in_rare_categories": (observations_below_threshold),
                "rare_observation_share_pct": (
                    observations_below_threshold / total_observations * 100
                ),
                "categories_below_50": int(below_50.sum()),
                "observations_in_categories_below_50": (observations_below_50),
                "observations_below_50_share_pct": (
                    observations_below_50 / total_observations * 100
                ),
            }
        )

        category_bands = pd.cut(
            counts,
            bins=frequency_band_edges,
            labels=frequency_band_labels,
            include_lowest=True,
        )

        frequency_table = pd.DataFrame(
            {
                "frequency_band": category_bands,
                "trip_count": counts.to_numpy(),
            }
        )

        band_summary = (
            frequency_table.groupby(
                "frequency_band",
                observed=False,
            )
            .agg(
                category_count=("trip_count", "size"),
                observation_count=("trip_count", "sum"),
            )
            .reindex(frequency_band_labels, fill_value=0)
            .reset_index()
        )

        band_summary["observation_share_pct"] = (
            band_summary["observation_count"] / total_observations * 100
        )
        band_summary["category_feature"] = column

        frequency_records.append(band_summary)

        top_table = (
            counts.head(top_n)
            .rename_axis("category_value")
            .reset_index(name="trip_count")
        )

        top_table["trip_share_pct"] = top_table["trip_count"] / total_observations * 100

        top_categories[column] = top_table

    overview = pd.DataFrame(overview_records).set_index("category_feature")

    frequency_bands = pd.concat(
        frequency_records,
        ignore_index=True,
    )

    return {
        "overview": overview,
        "frequency_bands": frequency_bands,
        "top_categories": top_categories,
    }


def summarize_distance_metric_comparison(df):
    """Compare Haversine and Manhattan-style distances for the same trips.

    Args:
        df: DataFrame containing both geographic distance features.

    Returns:
        DataFrame containing paired distance and difference statistics.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no valid observations remain or distances are negative.
    """
    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]

    missing_columns = [
        column for column in distance_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[distance_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid paired distance observations remain.")

    if (data < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    haversine = data["haversine_distance"]
    manhattan = data["manhattan_distance"]
    difference = manhattan - haversine

    positive_haversine = haversine > 0
    distance_ratio = manhattan[positive_haversine] / haversine[positive_haversine]

    summary = {
        "valid_trip_count": len(data),
        "haversine_median_km": haversine.median(),
        "manhattan_median_km": manhattan.median(),
        "median_difference_km": difference.median(),
        "mean_difference_km": difference.mean(),
        "p90_difference_km": difference.quantile(0.90),
        "p99_difference_km": difference.quantile(0.99),
        "maximum_difference_km": difference.max(),
        "manhattan_greater_pct": (manhattan > haversine).mean() * 100,
        "equal_distances_pct": (manhattan == haversine).mean() * 100,
        "median_ratio_positive_haversine": distance_ratio.median(),
        "p90_ratio_positive_haversine": distance_ratio.quantile(0.90),
        "zero_haversine_count": int((haversine == 0).sum()),
    }

    return pd.DataFrame([summary])


def summarize_distance_relationship_patterns(df, n_bins=10):
    """Summarize trip duration across bins of distance-related features.

    Args:
        df: DataFrame containing trip duration, original distances, and
            distance relationship features.
        n_bins: Number of approximately equal-frequency bins.

    Returns:
        Dictionary containing feature distribution statistics and
        trip-duration summaries for each feature's bins.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If arguments are invalid or data cannot be binned.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    feature_columns = [
        "haversine_distance",
        "manhattan_distance",
        "distance_ratio",
        "distance_difference",
    ]

    required_columns = ["trip_duration", *feature_columns]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid observations remain for analysis.")

    if (data["trip_duration"] < 0).any():
        raise ValueError("Trip durations must be non-negative.")

    if (data[["haversine_distance", "manhattan_distance"]] < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    if (data["distance_ratio"] < 0).any():
        raise ValueError("Distance ratio values must be non-negative.")

    distribution_records = []
    duration_summaries = {}

    for column in feature_columns:
        values = data[column]

        distribution_records.append(
            {
                "feature": column,
                "valid_count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p90": values.quantile(0.90),
                "maximum": values.max(),
                "skewness": values.skew(),
            }
        )

        if values.nunique() < 2:
            raise ValueError(
                f"Cannot create multiple bins for {column}: "
                "fewer than two distinct values."
            )

        bins = pd.qcut(
            values,
            q=n_bins,
            duplicates="drop",
        )

        if len(bins.cat.categories) < 2:
            raise ValueError(f"Cannot create multiple quantile bins for {column}.")

        grouped = data.groupby(bins, observed=True)["trip_duration"]

        summary = grouped.agg(
            trip_count="size",
            mean_duration_seconds="mean",
            median_duration_seconds="median",
        )

        summary["p90_duration_seconds"] = grouped.quantile(0.90)

        summary["trip_share_pct"] = summary["trip_count"] / len(data) * 100
        summary["mean_duration_minutes"] = summary.pop("mean_duration_seconds") / 60
        summary["median_duration_minutes"] = summary.pop("median_duration_seconds") / 60
        summary["p90_duration_minutes"] = summary.pop("p90_duration_seconds") / 60

        duration_summaries[column] = summary.reset_index(names="feature_bin")

    distribution_summary = pd.DataFrame(distribution_records).set_index("feature")

    return {
        "feature_distributions": distribution_summary,
        "duration_by_bin": duration_summaries,
    }
