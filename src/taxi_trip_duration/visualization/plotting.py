"""
Plotting Utilities for NYC Taxi Trip Duration EDA.

This module provides visualization functions for target distributions,
geographic distributions, and enriched deep-dive insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global seaborn styling
sns.set_style("whitegrid")


def plot_target_distribution(df):
    """Plots raw duration (zoomed) vs log1p transformed duration."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Zoom in on trips under 2 hours (7200 seconds)
    sns.histplot(
        df[df["trip_duration"] < 7200]["trip_duration"],
        bins=100,
        ax=axes[0],
        color="#3498db",
    )
    axes[0].set_title("Raw Duration (Zoomed < 2 hours)")
    axes[0].set_xlabel("Trip Duration (seconds)")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Log1p transformed distribution
    sns.histplot(np.log1p(df["trip_duration"]), bins=100, ax=axes[1], color="#2ecc71")
    axes[1].set_title("Log1p Transformed Duration")
    axes[1].set_xlabel("Log1p(Trip Duration)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_geographic_distribution(df, n_samples=100000):
    """Plots pickup coordinates to visualize spatial distribution and GPS errors."""
    plt.figure(figsize=(8, 6))

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    sns.scatterplot(
        x=sample_df["pickup_longitude"],
        y=sample_df["pickup_latitude"],
        alpha=0.4,
        s=15,
        color="#2c3e50",
        edgecolor=None,
    )

    plt.title(
        "Geographic Distribution of Pickups (Identifying GPS Outliers)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)

    plt.xlim(-74.5, -73.3)
    plt.ylim(40.4, 41.2)

    plt.tight_layout()
    plt.show()
