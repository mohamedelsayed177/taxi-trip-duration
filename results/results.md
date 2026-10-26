# NYC Taxi Trip Duration Prediction — Model Results

## 1. Baseline Model Evaluation

The initial experiments establish a baseline for Ridge Regression using the original features without the engineered geographic distance features.

The model uses `Ridge(alpha=1.0)`, with numerical features scaled using either `StandardScaler` or `MinMaxScaler` and categorical features encoded using `OneHotEncoder`.

### Scaler Comparison

| Scaler | Training R² | Validation R² |
|---|---:|---:|
| StandardScaler | 0.06436 | 0.06645 |
| MinMaxScaler | 0.06152 | 0.05995 |

### Decision

`StandardScaler` achieved a slightly higher validation R² score than `MinMaxScaler`. It was therefore selected for subsequent experiments.

The low baseline scores provided a reference point for investigating whether engineered geographic, temporal, and categorical features could improve model performance.

---

## 2. Feature Engineering Experiments

The following experiments use Ridge Regression with `alpha=1.0`. Performance is evaluated using R² on the log-transformed trip duration target, `log1p_trip_duration`.

Each experiment progressively enables additional feature groups to investigate their contribution to training and validation performance.

### Experiment Results

| Experiment | Training R² | Validation R² |
|---|---:|---:|
| Baseline | 0.06436 | 0.06645 |
| Distance Features | 0.35409 | 0.38955 |
| Distance + Bearing + Airport | 0.40152 | 0.42380 |
| Add Time Interactions | 0.40604 | 0.42549 |
| Add Heavy Hitters | 0.68603 | 0.66113 |
| Add Log Distances | 0.72558 | 0.71050 |
| Add Cyclical Time Features | 0.72555 | 0.71049 |
| Add Distance Relationship Features | 0.72825 | 0.71223 |

### Interpretation

**Geographic distance features produced the first substantial improvement.**

Adding Haversine and Manhattan-style distance increased validation R² from 0.06645 to 0.38955. Adding pickup bearing and airport proximity increased it further to 0.42380.

**Rush-hour interactions provided a smaller improvement.**

Adding time interactions increased validation R² from 0.42380 to 0.42549. This suggests that the additional features contributed relatively little to validation performance in this configuration.

**Spatial categories produced a substantial improvement.**

Adding the heavy-hitter categories increased validation R² from 0.42549 to 0.66113. These categorical features provide an alternative representation of recurring spatial and temporal patterns.

**Log-transformed distances produced another substantial improvement.**

Adding logarithmic distance features increased validation R² from 0.66113 to 0.71050. The result suggests that these alternative representations provide useful information for the Ridge Regression pipeline.

**Cyclical time features did not improve the result in this experiment.**

Validation R² changed slightly from 0.71050 to 0.71049 after adding cyclical time features. This difference is negligible and does not indicate an improvement.

**Distance relationship features achieved the best validation result.**

Adding `distance_ratio` and `distance_difference` increased validation R² from 0.71050 to 0.71223, a small but measurable improvement.

The final configuration achieved a training R² of 0.72825 and a validation R² of 0.71223.

---

## 3. Spatial Grid Precision Experiments

The spatial grid precision determines how pickup and drop-off coordinates are grouped into categorical features. Different precision settings produce different numbers of spatial categories and levels of detail.

The following experiments compare precision settings of 1, 2, 3, and 4 decimal places.

| Grid Precision | Training R² | Validation R² |
|---|---:|---:|
| 1 | 0.66529 | 0.65999 |
| 2 | 0.72825 | 0.71223 |
| 3 | 0.89245 | 0.69296 |
| 4 | 0.94799 | 0.64835 |

**Note:** Precision 2 corresponds to the default grid precision used in the *Add Distance Relationship Features* experiment. Its results are the same configuration, not an independent experiment.

### Interpretation

Precision 1 produced lower training and validation scores than precision 2, suggesting that the coarser spatial representation may not capture enough location detail.

Precision 2 achieved the highest validation R² of the tested settings, at 0.71223.

Increasing precision to 3 and 4 substantially increased training performance while reducing validation performance. At precision 4, training R² reached 0.94799, but validation R² fell to 0.64835.

This widening performance gap indicates that the more detailed spatial categories do not generalize as well to the validation data and are consistent with overfitting.

### Decision

`grid_precision=2` is retained as the default setting because it achieved the best validation performance among the tested precision values.

---

## 4. Final Model Performance

The final pipeline uses Ridge Regression with `alpha=1.0`, `StandardScaler` for numerical features, and `OneHotEncoder` for categorical features.

The selected feature configuration includes geographic distances, pickup bearing, airport proximity, rush-hour interactions, spatial and day–hour categories, logarithmic distances, cyclical time features, and distance relationship features.

### Final Scores

| Dataset | R² Score |
|---|---:|
| Training | 0.72825 |
| Validation | 0.71223 |
| Test | 0.71629 |

The final test score was obtained by loading the saved model and evaluating it on the prepared test split.

All three R² scores are calculated against `log1p_trip_duration`, rather than the original trip duration in seconds. Therefore, they measure performance on the log-transformed target scale and should not be interpreted as direct measures of prediction error in seconds.

### Generalization

The training R² of 0.72825 and validation R² of 0.71223 show a relatively small performance gap compared with the higher-precision spatial grid experiments.

The test R² of 0.71629 is slightly higher than the validation R². This is consistent with broadly similar performance across the validation and test splits, although it does not by itself establish performance on every possible set of taxi trips.

---

## 5. Conclusions

The experiments demonstrate that feature engineering substantially improved the Ridge Regression baseline.

The main findings are:

- Haversine and Manhattan-style distances provided substantial improvements over the original-feature baseline.
- Spatial and day–hour categories produced a further increase in validation performance.
- Log-transformed distances improved the model's validation score.
- Cyclical time features produced no measurable improvement in the tested configuration.
- Distance relationship features provided a small additional improvement.
- Spatial grid precision 2 achieved better validation performance than the tested alternatives at precision 1, 3, and 4.

The final selected configuration achieved a **validation R² of 0.71223** and a **test R² of 0.71629**, evaluated on the log-transformed target.

These experiments support retaining the current feature configuration as the final model for this project. The results describe performance on the prepared dataset splits; further work could investigate prediction errors, unusual geographic observations, and alternative approaches to high-cardinality spatial features.

