# 📺 Predicting User Viewing Behavior — STC TV

Forecasting daily watch hours for STC Jawwy TV using [Prophet](https://facebook.github.io/prophet/).

## Objective
Analyze viewing patterns and build a model to forecast future watch hours.

## Dataset
86 days of daily watch-hour data (Jan–Apr 2018).

## Approach
- EDA + seasonal decomposition (trend, weekly pattern, residuals)
- Prophet model tuned via grid search
- Evaluated with **time series cross-validation** (rolling-origin), not a static train/test split

**Best config:** `changepoint_prior_scale=0.001`, `seasonality_prior_scale=1.0`, `weekly_seasonality=False`

## Results
| Metric | Value |
|---|---|
| MAE | 75.11 hours |
| RMSE | 92.61 hours |

## Key Insights
- Overall downward trend in watch hours.
- Weekly seasonality looked promising in EDA, but cross-validation showed it *hurt* accuracy — likely too little data (~12 weeks) to estimate it reliably.
- The forecast line is smoother than actual data by design: it captures trend/seasonality, not daily noise.
- 60-day forecast is a long horizon relative to 86 days of history, so uncertainty grows over time.

## What I Learned
First version scored the model on the same data it trained on, giving a misleadingly good MAE. Fixed by evaluating out-of-sample with `cross_validation`, which reflects real forecasting performance.

## Tech Stack
Python, pandas, Prophet, scikit-learn, statsmodels, Plotly/Matplotlib

## Run It
```
pip install pandas prophet scikit-learn statsmodels plotly matplotlib scipy
```
Open the notebook, update the dataset path, run all cells.
