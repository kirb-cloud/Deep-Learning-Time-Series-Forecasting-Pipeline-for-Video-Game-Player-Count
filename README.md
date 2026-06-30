# Forecasting Player Engagement in Counter-Strike 2: A Time Series Analysis of Trends, Seasonality, and Player Behavior

A capstone project applying classical statistical methods and deep learning models to forecast daily concurrent player counts for Counter-Strike 2 (CS2), using historical data sourced from SteamDB.

## Research Question

What trends, seasonal patterns, and temporal dependencies characterize player engagement in Counter-Strike 2, and how accurately can statistical and deep learning models forecast future player participation?

## Dataset

- **Source:** SteamDB Counter-Strike 2 player count chart (App ID 730)
- **Period of study:** May 2023 – May 2026
- **Observations:** 1,471 daily concurrent player count records
- **Columns used:** `DateTime`, `Players`, `Average Players`

Note: SteamDB tracks history under App ID 730, which was previously used by CS:GO, so the raw export includes data back to 2011. The notebook filters this down to the CS2-relevant study period (May 2023 onward).

## Project Structure

The analysis is organized into 12 parts inside the notebook:

1. **Research Foundation** — research question and objective
2. **Data Set Description** — relationship of the dataset to the research objective
3. **Data Preparation** — loading, type correction, datetime indexing, filtering to the study period, missing value checks, descriptive statistics
4. **Exploratory Time Series Analysis** — time series plot, box plots by year, monthly heatmap, lag plots, ACF/PACF
5. **Time Series Diagnostics & Transformations** — downsampling/upsampling, time shifting, rolling/expanding statistics, white noise diagnostics, stationarity testing (ADF/KPSS), trend & seasonal decomposition (including custom LOESS/STL decomposition)
6. **Forecasting Methods & Model Evaluation** — mean/naive/drift baselines, calendar/population/inflation/mathematical adjustments, residual diagnostics, train-test split strategies, walk-forward validation (expanding & sliding window)
7. **Handling Missing Data** — univariate/multivariate imputation, interpolation, exponential smoothing comparisons, Holt-Winters seasonal smoothing, AR/MA modeling
8. **Statistical Forecasting Models** — Autoregression (AR), Moving Average (MA), ARIMA, SARIMA, Holt-Winters
9. **Convolutional Neural Network (CNN)** — univariate and multi-step CNN forecasting
10. **Deep Learning — MLP** — multi-layer perceptron forecasting
11. **Deep Learning — LSTM** — vanilla, stacked, bidirectional, CNN-LSTM, ConvLSTM, vector-output, and encoder-decoder architectures
12. **Hyperparameter Optimization** — grid search framework and differencing order tuning

A full write-up and supporting analysis is also provided as an accompanying PDF report.

## Key Findings

- A **positive long-term trend** in player engagement from launch through 2025.
- **Consistent weekly seasonality**, with higher player counts on weekends.
- **Strong short-term autocorrelation**, requiring differencing to stabilize the series for modeling.
- The **Autoregression (AR) model** delivered the best forecasting performance, with an RMSE of **45,687** and a MAPE of **3.98%** — outperforming ARIMA, Holt-Winters, and deep learning approaches (CNN, LSTM, MLP) on this dataset.
- For short-term CS2 player forecasts, traditional statistical models that leverage strong autocorrelation outperformed more complex deep learning architectures.

## Repository Contents

| File | Description |
|---|---|
| `Time_Series_Capstone_Project.ipynb` | Full analysis notebook (data prep, EDA, diagnostics, statistical models, deep learning models, hyperparameter tuning) |
| `Time_Series_Capstone_Project.pdf` | Exported report version of the analysis |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
statsmodels
pmdarima
tensorflow
keras
ipywidgets
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels pmdarima tensorflow keras ipywidgets
```

## Usage

1. Download the SteamDB CS2 player count CSV export (App ID 730).
2. Update the file path in the data-loading cell of the notebook to point to your local CSV.
3. Run the notebook top to bottom — sections are sequential, with later sections (statistical and deep learning models) depending on the cleaned/decomposed series built in earlier sections.
4. Model evaluation metrics (RMSE, MAE, MAPE) are reported per model in Part 8 onward for direct comparison.

## Notes

- Several Part 7 / Part 5 sections (e.g., upsampling to hourly frequency, missing-data imputation methods) are included as methodological exercises to demonstrate technique, even where the CS2 dataset itself (daily, complete after filtering) doesn't strictly require them.
- Walk-forward validation (both expanding and sliding window) is used in addition to single/multiple train-test splits to more rigorously evaluate model performance on sequential data.
