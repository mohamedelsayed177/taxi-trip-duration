"""
Plotting Utilities for NYC Taxi Trip Duration EDA.

This module provides visualization functions for target distributions,
geographic distributions, and enriched deep-dive insights.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set global seaborn styling
sns.set_style("whitegrid")


def plot_target_distribution(df):
    """Plot raw trip duration and its log1p transformation.

    Args:
        df: DataFrame containing trip duration feature.
    """
    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Focus on trips under two hours.
    sns.histplot(
        df[df["trip_duration"] < 7200]["trip_duration"],
        bins=100,
        ax=axes[0],
        color="#3498db",
    )
    axes[0].set_title("Raw Duration (Trips Under 2 Hours)")
    axes[0].set_xlabel("Trip Duration (seconds)")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Show the transformed target across the full dataset.
    sns.histplot(
        np.log1p(df["trip_duration"]),
        bins=100,
        ax=axes[1],
        color="#2ecc71",
    )
    axes[1].set_title("Log1p Transformed Duration (All Trips)")
    axes[1].set_xlabel("Log1p(Trip Duration)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_geographic_distribution(df, n_samples=100000):
    """Plot a sample of pickup locations within the NYC region.

    Args:
        df: DataFrame containing pickup geographic coordinates.
        n_samples: Maximum number of observations used for plotting.
    """
    plt.figure(figsize=(8, 6))

    sample_df = df.sample(
        n=min(n_samples, len(df)),
        random_state=42,
    )

    sns.scatterplot(
        x=sample_df["pickup_longitude"],
        y=sample_df["pickup_latitude"],
        alpha=0.4,
        s=15,
        color="#2c3e50",
        edgecolor=None,
    )

    plt.title(
        "Geographic Distribution of Pickup Locations (NYC Region)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)

    plt.xlim(-74.5, -73.3)
    plt.ylim(40.4, 41.2)

    plt.tight_layout()
    plt.show()


def plot_distance_vs_duration(df, n_samples=100_000):
    """Plot trip duration against Haversine and Manhattan-style distances.

    Args:
        df: DataFrame containing trip duration and distance features.
        n_samples: Maximum number of observations used for plotting.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If n_samples is invalid or no valid data remains.
    """
    required_columns = [
        "trip_duration",
        "haversine_distance",
        "manhattan_distance",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid observations are available for plotting.")

    if (data < 0).any().any():
        raise ValueError("Duration and distance values must be non-negative.")

    sample_df = data.sample(
        n=min(n_samples, len(data)),
        random_state=42,
    )

    log_duration = np.log1p(sample_df["trip_duration"])

    distance_columns = [
        "haversine_distance",
        "manhattan_distance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, column in zip(axes, distance_columns):
        log_distance = np.log1p(sample_df[column])

        hexbin = ax.hexbin(
            log_distance,
            log_duration,
            gridsize=60,
            mincnt=1,
            bins="log",
        )

        ax.set_title(f"{column.replace('_', ' ').title()} vs. Trip Duration")
        ax.set_xlabel("Log1p(Distance in km)")
        ax.set_ylabel("Log1p(Trip Duration in seconds)")

        fig.colorbar(
            hexbin,
            ax=ax,
            label="Observation Count (log scale)",
        )

    fig.suptitle(
        "Trip Duration vs. Geographic Distance",
        fontsize=15,
    )
    fig.tight_layout()
    plt.show()


def plot_distance_distributions(df, n_samples=100_000):
    """Plot the central distributions of both geographic distance features.

    Args:
        df: DataFrame containing the two distance features.
        n_samples: Maximum number of observations used for plotting.

    Raises:
        KeyError: If a required distance column is missing.
        ValueError: If n_samples is invalid or no valid data remains.
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

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    data = df[distance_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid distance observations are available.")

    if (data < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    sample_df = data.sample(
        n=min(n_samples, len(data)),
        random_state=42,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3498db", "#2ecc71"]

    for ax, column, color in zip(axes, distance_columns, colors):
        upper_limit = data[column].quantile(0.99)
        distances = sample_df.loc[
            sample_df[column] <= upper_limit,
            column,
        ]

        sns.histplot(
            distances,
            bins=100,
            ax=ax,
            color=color,
        )

        ax.set_title(f"{column.replace('_', ' ').title()} (Up to 99th Percentile)")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Frequency")

    fig.suptitle("Geographic Distance Distributions", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_distance_log_scaling(df, n_samples=100_000):
    """Compare original and log1p-transformed distance distributions.

    Args:
        df: DataFrame containing both geographic distance features.
        n_samples: Maximum number of observations used for plotting.

    Raises:
        KeyError: If a required distance column is missing.
        ValueError: If n_samples is invalid or no valid data remains.
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

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    data = df[distance_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid distance observations are available.")

    if (data < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    sample_df = data.sample(
        n=min(n_samples, len(data)),
        random_state=42,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["#3498db", "#2ecc71"]

    for column_index, (column, color) in enumerate(zip(distance_columns, colors)):
        upper_limit = data[column].quantile(0.99)

        distances = sample_df.loc[
            sample_df[column] <= upper_limit,
            column,
        ]

        sns.histplot(
            distances,
            bins=100,
            ax=axes[0, column_index],
            color=color,
        )
        axes[0, column_index].set_title(
            f"{column.replace('_', ' ').title()} — Original Scale"
        )
        axes[0, column_index].set_xlabel("Distance (km)")
        axes[0, column_index].set_ylabel("Frequency")

        sns.histplot(
            np.log1p(distances),
            bins=100,
            ax=axes[1, column_index],
            color=color,
        )
        axes[1, column_index].set_title(
            f"{column.replace('_', ' ').title()} — Log1p Scale"
        )
        axes[1, column_index].set_xlabel("Log1p(Distance in km)")
        axes[1, column_index].set_ylabel("Frequency")

    fig.suptitle(
        "Original vs. Log-Transformed Distance Distributions",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_temporal_patterns(summaries):
    """Plot median trip duration by pickup hour, weekday, and month.

    Args:
        summaries: Dictionary returned by summarize_temporal_patterns().
    """
    sections = [
        ("by_hour", "Pickup Hour", "Median Duration by Pickup Hour"),
        (
            "by_day_of_week",
            "Day of the Week",
            "Median Duration by Day of the Week",
        ),
        ("by_month", "Pickup Month", "Median Duration by Month"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, x_label, title) in zip(axes, sections):
        summary = summaries[key]
        x_values = summary.index.tolist()
        y_values = summary["median_duration_minutes"]

        if key == "by_hour":
            ax.plot(x_values, y_values, marker="o", color="#3498db")
            ax.set_xticks(range(24))
        elif key == "by_day_of_week":
            ax.bar(x_values, y_values, color="#3498db")
            ax.tick_params(axis="x", rotation=35)
        else:
            ax.plot(x_values, y_values, marker="o", color="#2ecc71")
            ax.tick_params(axis="x", rotation=35)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Median Trip Duration (minutes)")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_weekday_weekend_patterns(summary):
    """Plot median and 90th-percentile duration by day type.

    Args:
        summary: DataFrame returned by summarize_weekday_weekend().
    """
    day_types = summary.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plots = [
        (
            "median_duration_minutes",
            "Median Trip Duration",
        ),
        (
            "p90_duration_minutes",
            "90th-Percentile Trip Duration",
        ),
    ]

    for ax, (column, title) in zip(axes, plots):
        ax.bar(
            day_types,
            summary[column],
            color="#3498db",
        )
        ax.set_title(title)
        ax.set_xlabel("Day Type")
        ax.set_ylabel("Duration (minutes)")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_rush_hour_patterns(summary):
    """Compare mean, median, and p90 trip duration by rush-hour group.

    Args:
        summary: DataFrame returned by summarize_rush_hour_patterns().
    """
    groups = summary.index.tolist()
    x_values = np.arange(len(groups))
    bar_width = 0.24

    metrics = [
        ("mean_duration_minutes", "Mean", "#3498db"),
        ("median_duration_minutes", "Median", "#2ecc71"),
        ("p90_duration_minutes", "90th Percentile", "#e67e22"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for index, (column, label, color) in enumerate(metrics):
        positions = x_values + (index - 1) * bar_width

        ax.bar(
            positions,
            summary[column],
            width=bar_width,
            label=label,
            color=color,
        )

    ax.set_title("Trip Duration by Rush-Hour Group")
    ax.set_xlabel("Pickup Period")
    ax.set_ylabel("Trip Duration (minutes)")
    ax.set_xticks(x_values)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_cyclical_time_encodings(df):
    """Plot pickup-hour and weekday encodings on circular coordinates.

    Args:
        df: DataFrame containing cyclical time features.
    """
    configurations = [
        (
            "pickup_hour",
            "pickup_hour_sin",
            "pickup_hour_cos",
            24,
            "Hour of Day",
        ),
        (
            "pickup_day_of_week",
            "pickup_day_sin",
            "pickup_day_cos",
            7,
            "Day of Week",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (category, sine, cosine, period, title) in zip(axes, configurations):
        data = df[[category, sine, cosine]].dropna()

        encoded = data.groupby(category)[[sine, cosine]].mean().reindex(range(period))

        if encoded.isna().any().any():
            raise ValueError(f"Missing encoded categories for {title}.")

        x_values = encoded[cosine].to_numpy()
        y_values = encoded[sine].to_numpy()

        # Close the circle by connecting the last category to the first.
        x_closed = np.append(x_values, x_values[0])
        y_closed = np.append(y_values, y_values[0])

        ax.plot(x_closed, y_closed, marker="o", color="#3498db")

        for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            ax.annotate(
                str(index),
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_title(title)
        ax.set_xlabel("Cosine Encoding")
        ax.set_ylabel("Sine Encoding")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    fig.suptitle("Cyclical Time Encodings", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_pickup_dropoff_density(df):
    """Plot pickup and drop-off location densities using hexbin plots.

    Args:
        df: DataFrame containing pickup and drop-off coordinates.

    Raises:
        KeyError: If required coordinate columns are missing.
        ValueError: If no valid coordinates fall within the plotting window.
    """
    coordinate_pairs = [
        (
            "Pickup",
            "pickup_longitude",
            "pickup_latitude",
        ),
        (
            "Drop-off",
            "dropoff_longitude",
            "dropoff_latitude",
        ),
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

    longitude_min, longitude_max = -74.5, -73.3
    latitude_min, latitude_max = 40.4, 41.2

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (location, longitude, latitude) in zip(axes, coordinate_pairs):
        coordinates = (
            df[[longitude, latitude]].replace([np.inf, -np.inf], np.nan).dropna()
        )

        within_window = coordinates[longitude].between(
            longitude_min, longitude_max
        ) & coordinates[latitude].between(latitude_min, latitude_max)

        coordinates = coordinates.loc[within_window]

        if coordinates.empty:
            raise ValueError(
                f"No {location.lower()} coordinates fall within the plotting window."
            )

        hexbin = ax.hexbin(
            coordinates[longitude],
            coordinates[latitude],
            gridsize=100,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )

        ax.set_title(f"{location} Location Density")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(longitude_min, longitude_max)
        ax.set_ylim(latitude_min, latitude_max)

        fig.colorbar(
            hexbin,
            ax=ax,
            label="Observation Count (log scale)",
        )

    fig.suptitle(
        "Pickup and Drop-off Geographic Concentration",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_spatial_grid_patterns(summaries):
    """Plot trip counts and median duration for frequent spatial grid cells.

    Args:
        summaries: Dictionary returned by summarize_spatial_grid_patterns().
    """
    configurations = [
        ("pickup_grid", "Pickup"),
        ("dropoff_grid", "Drop-off"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for row, (key, location) in enumerate(configurations):
        summary = summaries[key].sort_values(
            "trip_count",
            ascending=True,
        )

        grid_cells = summary["grid_cell"].astype(str)

        axes[row, 0].barh(
            grid_cells,
            summary["trip_count"],
            color="#3498db",
        )
        axes[row, 0].set_title(f"Most Frequent {location} Grid Cells")
        axes[row, 0].set_xlabel("Trip Count")
        axes[row, 0].set_ylabel("Grid Cell")
        axes[row, 0].grid(axis="x", alpha=0.3)

        axes[row, 1].barh(
            grid_cells,
            summary["median_duration_minutes"],
            color="#2ecc71",
        )
        axes[row, 1].set_title(f"Median Trip Duration by {location} Grid Cell")
        axes[row, 1].set_xlabel("Median Duration (minutes)")
        axes[row, 1].set_ylabel("Grid Cell")
        axes[row, 1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_extreme_distance_coordinates(df, percentile=99):
    """Plot endpoints of trips above a Haversine distance percentile.

    Args:
        df: DataFrame containing distances and pickup/drop-off coordinates.
        percentile: Percentile used to define the extreme-distance threshold.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the percentile is invalid or no valid data remains.
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

    data = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid distance and coordinate records remain.")

    threshold = data["haversine_distance"].quantile(percentile / 100)
    extreme = data.loc[data["haversine_distance"] > threshold]

    if extreme.empty:
        raise ValueError("No trips exceed the selected distance threshold.")

    endpoint_specs = [
        ("Pickup", "pickup_longitude", "pickup_latitude", "#3498db"),
        ("Drop-off", "dropoff_longitude", "dropoff_latitude", "#e67e22"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, longitude, latitude, color in endpoint_specs:
        coordinates = extreme[[longitude, latitude]]

        inside = coordinates[longitude].between(-74.5, -73.3) & coordinates[
            latitude
        ].between(40.4, 41.2)

        axes[0].scatter(
            coordinates.loc[inside, longitude],
            coordinates.loc[inside, latitude],
            s=10,
            alpha=0.4,
            color=color,
            label=name,
        )

        axes[1].scatter(
            coordinates.loc[~inside, longitude],
            coordinates.loc[~inside, latitude],
            s=25,
            alpha=0.7,
            color=color,
            label=name,
        )

    axes[0].set_title(
        f"Extreme-Distance Trip Endpoints Within Window\n"
        f"Haversine Distance > {threshold:.2f} km"
    )
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_xlim(-74.5, -73.3)
    axes[0].set_ylim(40.4, 41.2)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Endpoints Outside the Geographic Window")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if (
        extreme["pickup_longitude"].between(-74.5, -73.3)
        & extreme["pickup_latitude"].between(40.4, 41.2)
    ).all() and (
        extreme["dropoff_longitude"].between(-74.5, -73.3)
        & extreme["dropoff_latitude"].between(40.4, 41.2)
    ).all():
        axes[1].text(
            0.5,
            0.5,
            "No endpoints outside the window",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    fig.suptitle(
        "Geographic Inspection of Extreme-Distance Trips",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_airport_proximity_patterns(summaries):
    """Plot median trip duration across airport-distance bands.

    Args:
        summaries: Dictionary returned by summarize_airport_proximity().
    """
    airport_labels = {
        "pickup_jfk_distance": "Pickup Distance to JFK",
        "pickup_lga_distance": "Pickup Distance to LGA",
        "pickup_ewr_distance": "Pickup Distance to EWR",
        "dropoff_jfk_distance": "Drop-off Distance to JFK",
        "dropoff_lga_distance": "Drop-off Distance to LGA",
        "dropoff_ewr_distance": "Drop-off Distance to EWR",
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    for ax, (column, title) in zip(
        axes.flat,
        airport_labels.items(),
    ):
        summary = summaries[column]

        ax.bar(
            summary.index.astype(str),
            summary["median_duration_minutes"],
            color="#3498db",
        )

        ax.set_title(title)
        ax.set_xlabel("Distance Band")
        ax.set_ylabel("Median Trip Duration (minutes)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Trip Duration by Proximity to Airport Reference Points",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_bearing_distance_patterns(summaries):
    """Plot trip frequency and median distance by compass sector.

    Args:
        summaries: Dictionary returned by
            summarize_bearing_distance_patterns().
    """
    summary = summaries["by_bearing_sector"]
    sectors = summary.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        sectors,
        summary["trip_count"],
        color="#3498db",
    )
    axes[0].set_title("Trip Count by Pickup Bearing Sector")
    axes[0].set_xlabel("Compass Sector")
    axes[0].set_ylabel("Trip Count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        sectors,
        summary["median_haversine_km"],
        color="#2ecc71",
    )
    axes[1].set_title("Median Haversine Distance by Bearing Sector")
    axes[1].set_xlabel("Compass Sector")
    axes[1].set_ylabel("Median Distance (km)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_categorical_duration_patterns(summaries):
    """Plot trip counts and median durations for categorical features.

    Args:
        summaries: Dictionary returned by summarize_categorical_duration().
    """
    configurations = [
        ("vendor_id", "Vendor ID"),
        ("store_and_fwd_flag", "Store-and-Forward Flag"),
        ("day_hour", "Day–Hour Category"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for row, (key, label) in enumerate(configurations):
        summary = summaries[key]

        count_data = summary.sort_values(
            "trip_count",
            ascending=True,
        )

        median_data = summary.sort_values(
            "median_duration_minutes",
            ascending=True,
        )

        axes[row, 0].barh(
            count_data.index.astype(str),
            count_data["trip_count"],
            color="#3498db",
        )
        axes[row, 0].set_title(f"Trip Count by {label}")
        axes[row, 0].set_xlabel("Trip Count")
        axes[row, 0].set_ylabel(label)
        axes[row, 0].grid(axis="x", alpha=0.3)

        axes[row, 1].barh(
            median_data.index.astype(str),
            median_data["median_duration_minutes"],
            color="#2ecc71",
        )
        axes[row, 1].set_title(f"Median Duration by {label}")
        axes[row, 1].set_xlabel("Median Trip Duration (minutes)")
        axes[row, 1].set_ylabel(label)
        axes[row, 1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_spatial_category_sparsity(summaries):
    """Plot spatial category frequency distributions and frequent cells.

    Args:
        summaries: Dictionary returned by
            summarize_spatial_category_sparsity().
    """
    configurations = [
        ("pickup_grid", "Pickup Grid"),
        ("dropoff_grid", "Drop-off Grid"),
        ("pickup_dropoff_grid", "Pickup–Drop-off Grid Pair"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for column_index, (key, label) in enumerate(configurations):
        frequency_data = summaries["frequency_bands"]
        frequency_data = frequency_data.loc[frequency_data["category_feature"] == key]

        axes[0, column_index].bar(
            frequency_data["frequency_band"].astype(str),
            frequency_data["category_count"],
            color="#3498db",
        )
        axes[0, column_index].set_title(f"Category Frequency Distribution: {label}")
        axes[0, column_index].set_xlabel("Observations per Category")
        axes[0, column_index].set_ylabel("Number of Categories")
        axes[0, column_index].tick_params(axis="x", rotation=30)
        axes[0, column_index].grid(axis="y", alpha=0.3)

        top_data = summaries["top_categories"][key].sort_values(
            "trip_count",
            ascending=True,
        )

        axes[1, column_index].barh(
            top_data["category_value"].astype(str),
            top_data["trip_count"],
            color="#2ecc71",
        )
        axes[1, column_index].set_title(f"Most Frequent Categories: {label}")
        axes[1, column_index].set_xlabel("Trip Count")
        axes[1, column_index].set_ylabel("Category")
        axes[1, column_index].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_distance_metric_comparison(df, n_samples=100_000):
    """Plot paired Haversine and Manhattan-style distance estimates.

    Args:
        df: DataFrame containing both geographic distance features.
        n_samples: Maximum number of observations used for plotting.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If n_samples is invalid or no valid data remains.
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

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    data = df[distance_columns].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid paired distance observations remain.")

    if (data < 0).any().any():
        raise ValueError("Distance values must be non-negative.")

    # Restrict the plot to a common upper limit for readable axes.
    plot_limit = data.quantile(0.99).max()

    plot_data = data.loc[data[distance_columns].le(plot_limit).all(axis=1)]

    if plot_data.empty:
        raise ValueError("No observations remain within the plotting range.")

    sample_df = plot_data.sample(
        n=min(n_samples, len(plot_data)),
        random_state=42,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    hexbin = ax.hexbin(
        sample_df["haversine_distance"],
        sample_df["manhattan_distance"],
        gridsize=60,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )

    ax.plot(
        [0, plot_limit],
        [0, plot_limit],
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Equal-distance reference",
    )

    ax.set_xlim(0, plot_limit)
    ax.set_ylim(0, plot_limit)
    ax.set_title("Haversine vs. Manhattan-Style Distance")
    ax.set_xlabel("Haversine Distance (km)")
    ax.set_ylabel("Manhattan-Style Distance (km)")
    ax.legend()

    fig.colorbar(
        hexbin,
        ax=ax,
        label="Observation Count (log scale)",
    )

    fig.tight_layout()
    plt.show()


def plot_distance_relationship_patterns(summaries):
    """Plot trip-duration trends across bins of distance-related features.

    Args:
        summaries: Dictionary returned by
            summarize_distance_relationship_patterns().
    """
    feature_labels = {
        "haversine_distance": "Haversine Distance",
        "manhattan_distance": "Manhattan-Style Distance",
        "distance_ratio": "Distance Ratio",
        "distance_difference": "Distance Difference",
    }

    duration_summaries = summaries["duration_by_bin"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (column, title) in zip(
        axes.flat,
        feature_labels.items(),
    ):
        summary = duration_summaries[column]
        bin_numbers = np.arange(1, len(summary) + 1)

        ax.plot(
            bin_numbers,
            summary["median_duration_minutes"],
            marker="o",
            label="Median",
            color="#3498db",
        )

        ax.plot(
            bin_numbers,
            summary["p90_duration_minutes"],
            marker="s",
            label="90th Percentile",
            color="#e67e22",
        )

        ax.set_title(title)
        ax.set_xlabel("Feature Quantile Bin (Low to High)")
        ax.set_ylabel("Trip Duration (minutes)")
        ax.set_xticks(bin_numbers)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Trip Duration Across Distance Feature Quantiles",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()
