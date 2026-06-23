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

## Setup Instructions

### 1. Get API Keys

**EIA API Key:**
1. Go to https://www.eia.gov/opendata/register.php
2. Register for a free account
3. Copy your API key from the confirmation email

**Supabase Project:**
1. Go to https://supabase.com and create a free account
2. Create a new project (choose any region, set a database password)
3. Once created, go to **Settings → API** and note:
   - **Project URL** (e.g., `https://abcdefgh.supabase.co`)
   - **anon / public** key (for the frontend)
   - **service_role** key (for the pipeline — keep this secret)

### 2. Set Up Supabase Tables

1. In your Supabase project, go to **SQL Editor**
2. Copy the entire contents of `sql/schema.sql`
3. Paste and run it — this creates all four tables with proper indexes and RLS policies

### 3. Configure the Frontend

Edit `index.html` and replace the two placeholder values near the top of the `<script>` section:

```javascript
const SUPABASE_URL = 'https://YOUR-PROJECT-ID.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGci...your-anon-key-here';
```

**Important:** Use the **anon** key here (not the service role key). The anon key is safe to expose in client-side code because RLS policies restrict it to read-only access.

### 4. Configure GitHub Repository Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|---|---|
| `EIA_API_KEY` | Your EIA API key |
| `SUPABASE_URL` | Your Supabase project URL (e.g., `https://xyz.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | Your Supabase **service_role** key |

### 5. Enable GitHub Pages

1. Go to **Settings → Pages**
2. Under **Source**, select **Deploy from a branch**
3. Choose the `main` branch and `/ (root)` folder
4. Click **Save**
5. Your dashboard will be live at `https://<username>.github.io/EcoNode/`

### 6. Test the Pipeline

1. Go to **Actions → EcoNode Daily Pipeline**
2. Click **Run workflow** to trigger it manually
3. Watch the logs to confirm data ingestion, forecasting, and green window detection
4. Refresh your GitHub Pages dashboard to see the data

After the first successful run, the pipeline will execute automatically every day at 06:00 UTC.

## How It Works

1. **Data Ingestion:** Pulls 30 days of hourly US grid generation data from the EIA API v2, including wind, solar, and total generation across the lower 48 states.

2. **Renewable Percentage:** Computes `(wind + solar) / total_generation × 100` for each hour.

3. **Forecasting:** Fits a Facebook Prophet model with daily and weekly seasonality on the renewable percentage time series, then generates a 48-hour forward outlook with confidence intervals.

4. **Green Windows:** Applies a sliding window algorithm to find the top 3 contiguous 4-hour blocks with the highest average predicted renewable percentage.

5. **Dashboard:** A static page fetches forecast data, green windows, and recent history directly from Supabase's REST API using the public anon key, then renders everything with Chart.js.
