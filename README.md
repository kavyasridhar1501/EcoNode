# EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows

An end-to-end, zero-maintenance MLOps pipeline that ingests live US electrical grid data across multiple regions, uses weather-enhanced Prophet forecasting to predict hourly renewable energy percentage and carbon intensity, and displays optimal "green compute windows" on a public dashboard.

**Demo** -> https://kavyasridhar1501.github.io/EcoNode/

## Architecture

```
EIA API (hourly grid data)    Open-Meteo (weather)
    │                              │
    ▼                              ▼
GitHub Actions (daily cron) ──▶ pipeline.py ──▶ analysis.py
    │                              │                │
    │  3 regions (US48,            │  Prophet +     │  hourly/weekly profiles,
    │  PJM, ERCOT)                 │  weather        │  weather correlations,
    │                              │  regressors,    │  forecastability
    │                              │  2 baselines    │  diagnostics, CO2 impact
    ▼                              ▼                ▼
Supabase PostgreSQL ◀──── history + forecasts + green windows + metrics + insights
    │
    ▼
GitHub Pages (index.html) ──▶ Multi-region Chart.js dashboard
```

**Zero local execution.** All ML training and analysis runs inside GitHub Actions on every scheduled run — not in a notebook someone has to remember to open. The dashboard is pure static HTML/JS served from GitHub Pages.

A separate [`tests.yml`](.github/workflows/tests.yml) workflow runs the pytest suite on every push/PR, independent of the daily data pipeline.

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

**Two baselines, not one.** A skill score against a single naive baseline is easy to game by picking a weak baseline. Every backtest compares Prophet against:

- **Persistence** — repeat the value from 24 hours ago
- **Climatology** — mean renewable % for that (weekday, hour) bucket, computed from the training window

If Prophet can't beat climatology, the weather regressors and Fourier seasonality aren't adding value for that region, and the dashboard's "vs Climatology Baseline" card will show it.

**Skill score has a confidence interval, not just a point estimate.** `analysis.bootstrap_skill_ci()` bootstraps the paired (model, baseline) errors 1,000x to produce a 90% CI on the skill score — a "45% better than baseline" number on a 168-point holdout week means little without knowing whether that's distinguishable from sampling noise.

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

### Region Insights (`analysis.py`)

This used to be a Jupyter notebook. Notebooks don't run themselves, and this one used to sit with every cell unexecuted — no output, no evidence any of it had actually been run against real data. `analysis.py` is a plain Python module of pure, unit-tested functions that runs as a real pipeline step on every scheduled run, and writes its results to Supabase's `region_insights` table so they show up on the live dashboard, not just in a file nobody opens:

- **Hour-of-day / day-of-week profiling** — peak generation hour and best weekday per region
- **Weather correlation** — Pearson correlation of renewable % against temperature, cloud cover, wind speed
- **Forecastability diagnostic** — lag-1 autocorrelation of the renewable % series. A smooth, solar-driven diurnal curve is highly self-similar hour to hour; a wind-dominant mix ramps less predictably. This is what actually explains *why* one region's MAE is consistently worse than another's, instead of just reporting the gap.
- **Absolute CO2 impact** — translates the abstract carbon-intensity delta between a green window and the grid average into kilograms of CO2 saved for an example workload (a 300W job run for 4 hours), so the number means something outside a percentage.

Every function in `analysis.py` is covered by `tests/test_analysis.py` with synthetic data (a known diurnal curve, pure noise, a perfectly repeating pattern) so the statistics can be checked against a known-correct answer, not just "does it run."

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
│   ├── pipeline.yml               # Daily cron: ingest, train, forecast, analyze
│   └── tests.yml                  # pytest on every push/PR
├── sql/
│   ├── schema.sql                 # Full Supabase schema (6 tables + RLS)
│   ├── migrate_v2.sql             # Migration for v1 -> v2 users
│   ├── migrate_v3.sql             # Migration for v2 -> v3 users
│   └── migrate_v4.sql             # Migration for v3 -> v4 users (insights, CIs)
├── tests/
│   ├── conftest.py                # Stubs Prophet/Supabase for fast unit tests
│   ├── test_pipeline.py           # Data quality, carbon math, green windows
│   └── test_analysis.py           # Statistics in analysis.py, against synthetic data
├── pipeline.py                    # Multi-region pipeline with weather + eval
├── analysis.py                    # Region insights, computed every run (was a notebook)
├── index.html                     # Static dashboard for GitHub Pages
├── requirements.txt               # Python dependencies for the pipeline
├── requirements-dev.txt           # Test-only dependencies
├── pytest.ini
└── README.md
```

## How It Works

1. **Data Ingestion:** Pulls 60 days of hourly generation data per region from the EIA API v2 (wind, solar, total) and merges hourly weather data from Open-Meteo.

2. **Data Quality:** Validates for negative values, zero generation, statistical outliers (>3 sigma), and time gaps. Removes or flags problematic records before training.

3. **Carbon Intensity:** Estimates grid carbon intensity using `carbon_factor * (1 - renewable_pct / 100)` with region-specific emission factors (PJM: 500, ERCO: 500, US48: 550 gCO2/kWh for non-renewable generation).

4. **Model Evaluation:** Performs holdout backtesting by training on all-but-last-168-hours and evaluating on the held-out week. Computes MAE, RMSE, MAPE, prediction interval coverage, and skill score vs. persistence baseline. Metrics are stored per region per day.

5. **Weather-Enhanced Forecasting:** Fits Prophet with daily/weekly seasonality plus temperature, cloud cover, and wind speed as exogenous regressors. Generates 48-hour forward outlook with confidence intervals.

6. **Green Windows:** Sliding window algorithm finds the top 3 contiguous 4-hour blocks with the highest average predicted renewable percentage and lowest carbon intensity.

7. **Region Insights:** `analysis.py` computes peak generation hour, best weekday, weather correlations, a forecastability diagnostic (autocorrelation), and an absolute CO2-savings example from the same run's data, and writes them to `region_insights`.

8. **Dashboard:** Region-selectable static page with forecast vs actuals overlay, carbon intensity chart, green window cards with CO2 estimates, model accuracy metrics (including both baselines and the skill score CI), region insights, and recent grid history.

## Real-World Impact

Percentages are easy to report and easy to ignore. To make the carbon savings concrete: for an example 300W GPU job run for 4 hours, `analysis.co2_savings_example()` converts the carbon-intensity delta between a region's best green window and its grid average into kilograms of CO2 actually saved by scheduling in that window — shown per-region on the dashboard's Insights panel. Swap in your own workload's wattage and duration to size the impact for a real deployment.

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

The dashboard now additionally reports each region's skill score against a **climatology baseline** and a **90% bootstrap CI** on the persistence skill score — see [Model Evaluation & Monitoring](#model-evaluation--monitoring). ERCOT's substantially higher MAE isn't just reported: `region_insights.forecastability_note` explains it via lag-1 autocorrelation — ERCOT's wind-heavy mix ramps less predictably hour to hour than the more solar-driven regions.

## Dashboard

The interactive dashboard provides:

- **Region selector** — switch between US48, PJM, and ERCOT
- **Real-time stats** — current renewable %, forecast average, carbon intensity, peak %, model accuracy
- **48-hour forecast chart** — predicted renewable % with confidence bands, actuals overlay, and green window highlighting
- **Carbon intensity chart** — separate gCO2/kWh time series with forecast and actuals
- **Green compute window cards** — top 3 optimal 4-hour blocks ranked by renewable %, with carbon intensity estimates
- **Model accuracy panel** — MAE, RMSE, MAPE, interval coverage, sample size, skill vs. two baselines with a confidence interval, last evaluation date
- **Region insights panel** — peak hour, best weekday, weather correlations, forecastability note, absolute CO2 savings for an example workload
- **Grid history table** — last 24 hours of actual generation data with color-coded renewable % bars

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_pipeline.py` and `tests/test_analysis.py` unit test the pure functions — data validation, carbon intensity math, green window selection, and every statistic in `analysis.py` — against synthetic data with known-correct answers (a perfect diurnal curve, pure noise, a perfectly repeating weekly pattern). `tests/conftest.py` stubs out Prophet and Supabase so the suite runs in under a second with no network access and no compiled Stan backend; it's testing the pipeline's logic, not re-testing Prophet or Postgres. CI runs this on every push and PR via `.github/workflows/tests.yml`, separate from the daily data pipeline.
