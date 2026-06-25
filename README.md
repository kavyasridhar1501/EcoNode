# EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows

A production-grade MLOps pipeline that ingests live US electrical grid data across three ISO regions, engineers weather-derived features, trains region-specific Prophet time-series models with exogenous regressors, and serves a 48-hour renewable energy forecast on a public dashboard — enabling carbon-optimal scheduling of AWS batch workloads.

Built entirely on free-tier infrastructure with zero local execution.

## Architecture

```
  EIA API v2                   Open-Meteo API
  (hourly grid generation)     (hourly weather regressors)
       │                              │
       ▼                              ▼
  ┌──────────────────────────────────────────────┐
  │        GitHub Actions  (daily cron)          │
  │                                              │
  │  ┌─────────────┐  ┌───────────────────────┐  │
  │  │  Ingestion   │  │  Feature Engineering  │  │
  │  │  3 regions   │  │  temp, cloud, wind    │  │
  │  └──────┬───────┘  └──────────┬────────────┘  │
  │         │    ┌────────────────┘               │
  │         ▼    ▼                                │
  │  ┌─────────────────┐  ┌──────────────────┐   │
  │  │  Data Quality    │  │  Model Eval      │   │
  │  │  Validation      │  │  MAE/RMSE/MAPE   │   │
  │  └──────┬───────────┘  └──────────────────┘   │
  │         ▼                                     │
  │  ┌─────────────────┐  ┌──────────────────┐   │
  │  │  Prophet Model   │  │  Green Window    │   │
  │  │  + Regressors    │──│  Detection       │   │
  │  └─────────────────┘  └──────────────────┘   │
  └──────────────────┬───────────────────────────┘
                     ▼
            Supabase PostgreSQL
            (5 tables, RLS policies)
                     │
                     ▼
            GitHub Pages Dashboard
            (Chart.js, vanilla JS)
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Sources** | EIA API v2, Open-Meteo API | Hourly grid generation (wind, solar, total) + weather regressors (temperature, cloud cover, wind speed) |
| **ML Framework** | Facebook Prophet | Additive/multiplicative time-series decomposition with exogenous regressors |
| **Data Processing** | Pandas, NumPy | Feature engineering, data quality validation, sliding window analysis |
| **Database** | Supabase (PostgreSQL) | Time-series storage with upsert conflict resolution, Row Level Security |
| **Orchestration** | GitHub Actions (cron) | Scheduled daily pipeline execution, dependency caching, secret management |
| **Frontend** | Chart.js, Vanilla JS | Interactive time-series visualization, REST API consumption |
| **Deployment** | GitHub Pages | Zero-config static hosting |
| **EDA** | Jupyter, Matplotlib, Seaborn, Statsmodels | Seasonal decomposition, correlation analysis, distribution profiling |

## Data Science Methodology

### Feature Engineering

The pipeline transforms raw grid generation data into a rich feature set for forecasting:

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

Every pipeline run performs backtesting against the previous day's predictions:

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
│   └── migrate_v2.sql            # Additive migration for existing deployments
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis (10 sections)
│   └── requirements.txt          # Notebook dependencies (matplotlib, seaborn, statsmodels)
├── pipeline.py                   # Multi-region pipeline (ingestion, weather, quality, forecasting, eval)
├── index.html                    # Interactive dashboard (Chart.js, Supabase REST)
├── requirements.txt              # Pipeline dependencies (prophet, pandas, supabase)
└── README.md
```

## Setup Instructions

### 1. Get API Keys

**EIA API Key:**
1. Go to https://www.eia.gov/opendata/register.php
2. Register for a free account
3. Copy your API key from the confirmation email

**Supabase Project:**
1. Go to https://supabase.com and create a free account
2. Create a new project (choose any region, set a database password)
3. Go to **Settings → API Keys** and note:
   - **Project URL** (visible in the browser URL: `https://<project-id>.supabase.co`)
   - **Publishable key** (for the frontend — safe to expose)
   - **Secret key** (for the pipeline — keep this secret)

No API key is needed for Open-Meteo (weather data) — it is free and open.

### 2. Set Up Supabase Tables

**New project:** Run `sql/schema.sql` in the Supabase SQL Editor.

**Upgrading from v1:** Run `sql/migrate_v2.sql` instead — it adds new columns and the `model_metrics` table without dropping existing data.

### 3. Configure the Frontend

Edit `index.html` and replace the two placeholder values:

```javascript
const SUPABASE_URL = 'https://YOUR-PROJECT-ID.supabase.co';
const SUPABASE_ANON_KEY = 'your-publishable-key-here';
```

### 4. Configure GitHub Repository Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|---|---|
| `EIA_API_KEY` | Your EIA API key |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Your Supabase **Secret key** |

### 5. Enable GitHub Pages

1. Go to **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/ (root)`
4. Your dashboard will be at `https://<username>.github.io/EcoNode/`

### 6. Test the Pipeline

1. Go to **Actions → EcoNode Daily Pipeline → Run workflow**
2. The first run processes 3 regions (~8-12 minutes)
3. Refresh your dashboard to see the data
4. Model evaluation metrics appear after the second run (needs previous forecasts to compare against)

## Dashboard

The interactive dashboard provides:

- **Region selector** — switch between US48, PJM, and ERCOT
- **Real-time stats** — current renewable %, forecast average, carbon intensity, peak %, model accuracy
- **48-hour forecast chart** — predicted renewable % with confidence bands, actuals overlay, and green window highlighting
- **Carbon intensity chart** — separate gCO2/kWh time series with forecast and actuals
- **Green compute window cards** — top 3 optimal 4-hour blocks ranked by renewable %, with carbon intensity estimates
- **Model accuracy panel** — MAE, RMSE, MAPE, interval coverage, sample size, last evaluation date
- **Grid history table** — last 24 hours of actual generation data with color-coded renewable % bars
