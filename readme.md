# Freight Rate Prediction – Machine Learning Assessment

This repository contains my solution for the Spotter Freight Rate Prediction Machine Learning assessment.

The objective was to develop and validate a regression model using historical freight-load data and generate freight-rate predictions for an unseen validation dataset.

## Approach

The solution includes:

- Exploratory data analysis and data-quality checks
- Data cleaning and feature engineering
- Time-based expanding-window validation
- Baseline Ridge Regression
- CatBoost model comparison
- Error analysis
- Final model training and prediction generation
- December 2025 fixed-scenario prediction
- Validation using the provided scoring script

## Data Quality and Feature Engineering

The development dataset contained:

- 300 missing weight values
- 292 negative weight values
- 374 missing market index values

Negative weights were treated as presumed sign errors and converted to positive values while retaining an indicator feature.

Additional engineered features include:

- Pickup-to-delivery route
- Month
- Day
- Day of week
- Day of year
- Days since January 1, 2025
- Missing-weight indicator
- Negative-weight indicator

## Validation Strategy

An expanding-window chronological validation strategy was used to better simulate future freight-rate prediction and avoid temporal leakage.

Validation folds:

1. Train January–July → Validate August
2. Train January–August → Validate September
3. Train January–September → Validate October

## Model Comparison

|            Model           |Average MAE  | Average RMSE |
| Historical Median Baseline | $1,136.42   | — |
| Ridge Regression           | $145.40     | $631.58 |
| CatBoost – RMSE Loss       | $158.13     | $640.62 |
| CatBoost – MAE Loss        | **$122.43** | $632.15 |

CatBoost optimized for Mean Absolute Error was selected as the final model because it achieved the lowest average MAE across the three chronological validation periods.

## Final Prediction Generation

The selected CatBoost model was retrained using all 48,000 labeled development records and used to generate predictions for all 12,000 unseen validation loads.

The generated prediction file contains:

```text
load_id,predicted_rate
