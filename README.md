# EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows

An end-to-end, zero-maintenance MLOps pipeline that ingests live US electrical grid data, forecasts hourly renewable energy percentage, and displays optimal "green compute windows" on a public dashboard for scheduling AWS batch workloads.

## Architecture

```
EIA API (hourly grid data)
    │
    ▼
GitHub Actions (daily cron) ──▶ pipeline.py
    │                              │
    │  fetch 30 days of            │  Prophet forecast
    │  wind + solar + total        │  48-hour outlook
    │                              │
    ▼                              ▼
Supabase PostgreSQL ◀──────── upsert history + forecasts + green windows
    │
    ▼
GitHub Pages (index.html) ──▶ Chart.js dashboard
```

**Zero local execution.** All ML training runs inside GitHub Actions. The dashboard is pure static HTML/JS served from GitHub Pages.

## Repository Structure

```
├── .github/workflows/
│   └── pipeline.yml          # GitHub Actions cron workflow
├── sql/
│   └── schema.sql            # Supabase table definitions + RLS policies
├── pipeline.py               # Data ingestion, Prophet forecasting, green windows
├── index.html                # Static dashboard for GitHub Pages
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

1. **Data Ingestion:** Pulls 30 days of hourly US grid generation data from the EIA API v2, including wind, solar, and total generation across the lower 48 states.

2. **Renewable Percentage:** Computes `(wind + solar) / total_generation × 100` for each hour.

3. **Forecasting:** Fits a Facebook Prophet model with daily and weekly seasonality on the renewable percentage time series, then generates a 48-hour forward outlook with confidence intervals.

4. **Green Windows:** Applies a sliding window algorithm to find the top 3 contiguous 4-hour blocks with the highest average predicted renewable percentage.

5. **Dashboard:** A static page fetches forecast data, green windows, and recent history directly from Supabase's REST API using the public anon key, then renders everything with Chart.js.
