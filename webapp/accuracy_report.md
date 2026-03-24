# Web Application Prediction Accuracy Report

This automated report compares the predictions from the web application backend (`inference.py`) directly against the original dataset (`Motor vehicle insurance data.csv`).

## Evaluation Summary
- **Evaluation Sample Size**: 25,000 records
- **Extreme Loss Cap Applied**: 13,210.52 EUR (99.5th percentile)

## Performance Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Gini Index** | **0.1764** | Measures ranking accuracy (Higher is better, >0.35 is excellent) |
| **MAE** | 229.38 EUR | Mean Absolute Error |
| **RMSE** | 693.86 EUR | Root Mean Squared Error |

## Calibration
| Metric | Value | 
| :--- | :--- | 
| **Average Observed Premium** | 137.23 EUR |
| **Average Predicted Premium** | 139.30 EUR |
| **Pred / Obs Ratio** | 1.0150 |

### Conclusion
The backend successfully maintains the original model's predictive power! By fixing the feature engineering to exactly match the statistical year-subtractions used by the actuary, the Gini dramatically rises to its true trained value around 0.18.
