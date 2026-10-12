"""
Note: I know that the best/fastest way to do these experiments is inside a notebook,
but I don't want to depend heavily on notebooks, so I'm doing it here in a .py file.
"""

from taxi_trip_duration.manipulation.feature_engineering import (
    date_time,
    haversine_distance,
    target_log,
)
from taxi_trip_duration.manipulation.data_helper import (
    load_dataset,
    checking_data,
    drop_id_features,
    train_val_split,
)
