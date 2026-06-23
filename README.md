# EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows

An end-to-end, zero-maintenance MLOps pipeline that ingests live US electrical grid data across multiple regions, uses weather-enhanced Prophet forecasting to predict hourly renewable energy percentage and carbon intensity, and displays optimal "green compute windows" on a public dashboard.

## Architecture

```
EIA API (hourly grid data)    Open-Meteo (weather)
    │                              │
    ▼                              ▼
GitHub Actions (daily cron) ──▶ pipeline.py
    │                              │
    │  3 regions (US48,            │  Prophet + weather regressors
    │  CAISO, ERCOT)               │  48-hour outlook per region
    │                              │  model evaluation (MAE/RMSE)
    ▼                              ▼
Supabase PostgreSQL ◀──── history + forecasts + green windows + metrics
    │
    ▼
GitHub Pages (index.html) ──▶ Multi-region Chart.js dashboard
```

**Zero local execution.** All ML training runs inside GitHub Actions. The dashboard is pure static HTML/JS served from GitHub Pages.

## Features

- **Multi-region support** — US Lower 48, California (CAISO), Texas (ERCOT)
- **Weather-enhanced forecasting** — Temperature, cloud cover, and wind speed as Prophet regressors via Open-Meteo (free, no API key)
- **Carbon intensity estimation** — gCO2/kWh computed from renewable % using region-specific emission factors
- **Model evaluation** — MAE, RMSE, MAPE, and prediction interval coverage tracked per run
- **Forecast vs actuals overlay** — Dashboard shows recent actuals leading into the forecast
- **Data quality checks** — Outlier detection, gap detection, negative value removal
- **EDA notebook** — Jupyter notebook with seasonality decomposition, weather correlations, and regional comparison

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

### 2. Set Up Supabase Tables

**New project:** Run `sql/schema.sql` in the Supabase SQL Editor.

**Upgrading from v1:** Run `sql/migrate_v2.sql` instead — it adds the new columns and `model_metrics` table without dropping existing data.

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

## How It Works

1. **Data Ingestion:** Pulls 30 days of hourly generation data per region from the EIA API v2 (wind, solar, total) and merges hourly weather data from Open-Meteo.

2. **Data Quality:** Validates for negative values, zero generation, statistical outliers (>3 sigma), and time gaps. Removes or flags problematic records before training.

3. **Carbon Intensity:** Estimates grid carbon intensity using `carbon_factor * (1 - renewable_pct / 100)` with region-specific emission factors (CISO: 350, ERCO: 500, US48: 550 gCO2/kWh for non-renewable generation).

4. **Model Evaluation:** Compares yesterday's forecasts against actual grid data. Computes MAE, RMSE, MAPE, and prediction interval coverage. Metrics are stored per region per day.

5. **Weather-Enhanced Forecasting:** Fits Prophet with daily/weekly seasonality plus temperature, cloud cover, and wind speed as exogenous regressors. Generates 48-hour forward outlook with confidence intervals.

6. **Green Windows:** Sliding window algorithm finds the top 3 contiguous 4-hour blocks with the highest average predicted renewable percentage and lowest carbon intensity.

7. **Dashboard:** Region-selectable static page with forecast vs actuals overlay, carbon intensity chart, green window cards with CO2 estimates, model accuracy metrics, and recent grid history.
