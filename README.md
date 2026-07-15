# EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows

An end-to-end, zero-maintenance MLOps pipeline that ingests live US electrical grid data across multiple regions, uses weather-enhanced Prophet forecasting to predict hourly renewable energy percentage and carbon intensity, and displays optimal "green compute windows" on a public dashboard.

**Demo** -> https://kavyasridhar1501.github.io/EcoNode/

## Architecture

```
EIA API (hourly grid data)    Open-Meteo (weather)
    │                              │
    ▼                              ▼
GitHub Actions (daily cron) ──▶ pipeline.py
    │                              │
    │  3 regions (US48,            │  Prophet + weather regressors
    │  PJM, ERCOT)                 │  48-hour outlook per region
    │                              │  model evaluation (MAE/RMSE)
    ▼                              ▼
Supabase PostgreSQL ◀──── history + forecasts + green windows + metrics
    │
    ▼
GitHub Pages (index.html) ──▶ Multi-region Chart.js dashboard
```

**Zero local execution.** All ML training runs inside GitHub Actions. The dashboard is pure static HTML/JS served from GitHub Pages.

## Data Science Methodology

### Feature Engineering

The pipeline transforms raw grid generation data into a feature set for forecasting:

- **Target variable:** `renewable_percentage = (wind_mwh + solar_mwh) / total_generation_mwh * 100`
- **Exogenous regressors:** Hourly temperature (C), cloud cover (%), and wind speed (m/s) from Open-Meteo, aligned to each region's representative geographic coordinate
- **Derived feature:** `carbon_intensity = emission_factor * (1 - renewable_pct / 100)` using EPA-informed region-specific emission factors (PJM: 500, ERCO: 500, US48: 550 gCO2/kWh)
- **Temporal features:** Daily and weekly seasonality components captured by Prophet's Fourier decomposition

### Time-Series Forecasting

Each region gets an independent **Facebook Prophet** model configured with:

- **Multiplicative seasonality** — captures the proportional (not additive) nature of renewable generation cycles
- **Daily seasonality** — models the solar generation curve (peak at solar noon, zero at night)
- **Weekly seasonality** — captures demand-driven patterns (lower weekend load increases renewable share)
- **Changepoint prior scale = 0.05** — regularized to prevent overfitting to short-term regime changes
- **Weather regressors** — temperature, cloud cover, and wind speed added via `add_regressor()`, providing the model with forward-looking meteorological context for the 48-hour prediction horizon

The model outputs point predictions with uncertainty intervals (Prophet's built-in Bayesian uncertainty estimation).

### Model Evaluation & Monitoring

Every pipeline run performs holdout backtesting — training on all-but-last-168-hours, evaluating on the held-out week:

| Metric | What It Measures |
|---|---|
| **MAE** (Mean Absolute Error) | Average magnitude of forecast errors in percentage points |
| **RMSE** (Root Mean Squared Error) | Error magnitude with higher penalty for large deviations |
| **MAPE** (Mean Absolute Percentage Error) | Scale-independent accuracy measure |
| **Prediction Interval Coverage** | % of actuals falling within Prophet's uncertainty bands (target: ~80%) |

Metrics are stored per region per day in the `model_metrics` table, enabling accuracy trend analysis and model drift detection over time.

### Data Quality Pipeline

Before training, every ingested batch passes through validation:

- **Negative value removal** — filters physically impossible negative generation readings
- **Zero-generation filtering** — removes rows where total generation is zero (prevents division-by-zero in percentage calculation)
- **Statistical outlier detection** — removes data points beyond 3 standard deviations from the mean renewable percentage
- **Temporal gap detection** — logs gaps > 1.5 hours in the time series for observability
- **Value clamping** — constrains renewable percentage to [0, 100] to prevent model training on impossible values

### Green Window Detection

A **sliding window algorithm** scans the 48-hour forecast to find optimal compute scheduling blocks:

1. Computes rolling 4-hour averages of predicted renewable percentage
2. Ranks all windows by average renewable % (descending)
3. Selects the top 3 non-overlapping windows
4. Computes average carbon intensity per window for cost-of-carbon comparison

### Exploratory Data Analysis

The `notebooks/eda.ipynb` Jupyter notebook provides:

- **Time-series visualization** — renewable percentage trends per region
- **Seasonal decomposition** — STL decomposition (via Statsmodels) separating trend, daily seasonality, and residual components
- **Daily/weekly seasonality profiling** — hour-of-day and day-of-week aggregated patterns with confidence bands
- **Distribution analysis** — renewable % and carbon intensity histograms per region
- **Weather correlation analysis** — scatter plots with OLS trend lines and Pearson correlation coefficients for each weather variable
- **Correlation heatmaps** — multi-variable correlation matrices (renewable %, temperature, cloud cover, wind speed)
- **Regional comparison** — box plots and overlaid hourly profiles across ISO regions
- **Model performance tracking** — MAE/RMSE/coverage trends over time

## Multi-Region Support

The pipeline processes three US Independent System Operator (ISO) regions, each with distinct generation mixes:

| Region | ISO | Characteristics | Emission Factor |
|---|---|---|---|
| **US48** | National | Balanced mix of gas, coal, nuclear, wind, solar | 550 gCO2/kWh |
| **PJM** | PJM Interconnection | Nuclear baseload, natural gas, moderate wind | 500 gCO2/kWh |
| **ERCO** | ERCOT (Texas) | Strong wind penetration, natural gas dominant | 500 gCO2/kWh |

Each region is modeled independently with its own weather regressors sourced from a representative geographic coordinate.

## Repository Structure

```
├── .github/workflows/
│   └── pipeline.yml              # Daily cron GitHub Actions workflow
├── sql/
│   ├── schema.sql                # Full Supabase schema (5 tables + RLS)
│   └── migrate_v2.sql            # Migration for existing v1 users
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis notebook
│   └── requirements.txt          # Notebook-specific dependencies
├── pipeline.py                   # Multi-region pipeline with weather + eval
├── index.html                    # Static dashboard for GitHub Pages
├── requirements.txt              # Python dependencies for GitHub Actions
└── README.md
```

## How It Works

1. **Data Ingestion:** Pulls 60 days of hourly generation data per region from the EIA API v2 (wind, solar, total) and merges hourly weather data from Open-Meteo.

2. **Data Quality:** Validates for negative values, zero generation, statistical outliers (>3 sigma), and time gaps. Removes or flags problematic records before training.

3. **Carbon Intensity:** Estimates grid carbon intensity using `carbon_factor * (1 - renewable_pct / 100)` with region-specific emission factors (PJM: 500, ERCO: 500, US48: 550 gCO2/kWh for non-renewable generation).

4. **Model Evaluation:** Performs holdout backtesting by training on all-but-last-168-hours and evaluating on the held-out week. Computes MAE, RMSE, MAPE, prediction interval coverage, and skill score vs. persistence baseline. Metrics are stored per region per day.

5. **Weather-Enhanced Forecasting:** Fits Prophet with daily/weekly seasonality plus temperature, cloud cover, and wind speed as exogenous regressors. Generates 48-hour forward outlook with confidence intervals.

6. **Green Windows:** Sliding window algorithm finds the top 3 contiguous 4-hour blocks with the highest average predicted renewable percentage and lowest carbon intensity.

7. **Dashboard:** Region-selectable static page with forecast vs actuals overlay, carbon intensity chart, green window cards with CO2 estimates, model accuracy metrics, and recent grid history.

## Model Performance

Current backtest results (168-hour holdout, evaluated daily):

| Region | MAE (pp) | Forecast Improvement | Interval Coverage | CO2 Savings |
|---|---|---|---|---|
| **US Lower 48** | 1.59 | 45% | 85% | 11% |
| **Mid-Atlantic (PJM)** | 2.08 | 16% | 86% | 6% |
| **Texas (ERCOT)** | 5.99 | 29% | 80% | 23% |

- **Forecast Improvement** = skill score vs. persistence (repeat-yesterday) baseline
- **Interval Coverage** = % of actuals within Prophet's Bayesian uncertainty bands (target: ~80%)
- **CO2 Savings** = carbon reduction from scheduling in green windows vs. grid average

## Dashboard

The interactive dashboard provides:

- **Region selector** — switch between US48, PJM, and ERCOT
- **Real-time stats** — current renewable %, forecast average, carbon intensity, peak %, model accuracy
- **48-hour forecast chart** — predicted renewable % with confidence bands, actuals overlay, and green window highlighting
- **Carbon intensity chart** — separate gCO2/kWh time series with forecast and actuals
- **Green compute window cards** — top 3 optimal 4-hour blocks ranked by renewable %, with carbon intensity estimates
- **Model accuracy panel** — MAE, RMSE, MAPE, interval coverage, sample size, last evaluation date
- **Grid history table** — last 24 hours of actual generation data with color-coded renewable % bars
