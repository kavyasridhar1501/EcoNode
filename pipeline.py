"""
EcoNode Pipeline: Ingest EIA grid data, forecast renewable %, identify green windows.

Runs entirely in GitHub Actions. Reads from EIA API v2, writes to Supabase.
"""

import os
import sys
import uuid
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from prophet import Prophet
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("econode")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EIA_API_KEY = os.environ["EIA_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

EIA_BASE = "https://api.eia.gov/v2"
REGION = "US48"
LOOKBACK_DAYS = 30
FORECAST_HOURS = 48
GREEN_WINDOW_HOURS = 4
GREEN_WINDOW_COUNT = 3

# EIA fuel-type IDs for the electricity API
# SUN = solar, WND = wind
FUEL_SOLAR = "SUN"
FUEL_WIND = "WND"


def supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# 1. Data Ingestion
# ---------------------------------------------------------------------------

def fetch_eia_series(fuel_type: str | None, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly generation from EIA API v2 electricity/rto/fuel-type-data.

    If fuel_type is None, fetches total generation across all fuel types.
    """
    url = f"{EIA_BASE}/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        "facets[respondent][]": REGION,
    }
    if fuel_type:
        params["facets[fueltype][]"] = fuel_type

    all_rows = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("response", {}).get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        total = int(data["response"].get("total", len(all_rows)))
        offset += len(rows)

        if offset >= total:
            break

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def ingest_grid_data() -> pd.DataFrame:
    """Pull last N days of hourly generation, compute renewable_percentage."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_str = start_dt.strftime("%Y-%m-%dT%H")
    end_str = end_dt.strftime("%Y-%m-%dT%H")

    log.info("Fetching EIA data from %s to %s", start_str, end_str)

    df_solar = fetch_eia_series(FUEL_SOLAR, start_str, end_str)
    df_wind = fetch_eia_series(FUEL_WIND, start_str, end_str)

    # For total generation, fetch ALL fuel types and sum per period
    df_all = fetch_eia_series(None, start_str, end_str)

    if df_all.empty:
        log.warning("No generation data returned from EIA")
        return pd.DataFrame()

    total = df_all.groupby("period")["value"].sum().reset_index()
    total.columns = ["timestamp_utc", "total_generation_mwh"]

    def agg_fuel(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["timestamp_utc", col_name])
        grouped = df.groupby("period")["value"].sum().reset_index()
        grouped.columns = ["timestamp_utc", col_name]
        return grouped

    solar = agg_fuel(df_solar, "solar_generation_mwh")
    wind = agg_fuel(df_wind, "wind_generation_mwh")

    merged = total.merge(solar, on="timestamp_utc", how="left")
    merged = merged.merge(wind, on="timestamp_utc", how="left")
    merged = merged.fillna(0)

    merged["renewable_percentage"] = np.where(
        merged["total_generation_mwh"] > 0,
        (merged["wind_generation_mwh"] + merged["solar_generation_mwh"])
        / merged["total_generation_mwh"]
        * 100,
        0,
    )

    merged["region"] = REGION
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)

    log.info("Ingested %d hourly records", len(merged))
    return merged


def upsert_history(sb: Client, df: pd.DataFrame) -> int:
    """Upsert historical grid data into Supabase."""
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "timestamp_utc": row["timestamp_utc"].isoformat(),
            "total_generation_mwh": float(row["total_generation_mwh"]),
            "wind_generation_mwh": float(row["wind_generation_mwh"]),
            "solar_generation_mwh": float(row["solar_generation_mwh"]),
            "renewable_percentage": round(float(row["renewable_percentage"]), 4),
            "region": row["region"],
        })

    # Upsert in batches of 500
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        sb.table("grid_history").upsert(
            batch, on_conflict="timestamp_utc,region"
        ).execute()

    log.info("Upserted %d history records", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# 2. Forecasting
# ---------------------------------------------------------------------------

def train_and_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Train Prophet on renewable_percentage, produce 48-hour forecast."""
    if len(df) < 48:
        log.warning("Insufficient data for forecasting (%d rows), need >= 48", len(df))
        return pd.DataFrame()

    prophet_df = df[["timestamp_utc", "renewable_percentage"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq="h")
    forecast = model.predict(future)

    # Only keep the future portion
    cutoff = prophet_df["ds"].max()
    future_forecast = forecast[forecast["ds"] > cutoff].copy()

    result = pd.DataFrame({
        "forecast_time": pd.to_datetime(future_forecast["ds"], utc=True),
        "renewable_percentage_predicted": future_forecast["yhat"].clip(lower=0, upper=100),
        "lower_bound": future_forecast["yhat_lower"].clip(lower=0, upper=100),
        "upper_bound": future_forecast["yhat_upper"].clip(lower=0, upper=100),
    })

    log.info("Generated %d-hour forecast", len(result))
    return result.reset_index(drop=True)


def upsert_forecasts(sb: Client, df: pd.DataFrame) -> int:
    """Upsert forecast rows into Supabase."""
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "forecast_time": row["forecast_time"].isoformat(),
            "renewable_percentage_predicted": round(float(row["renewable_percentage_predicted"]), 4),
            "lower_bound": round(float(row["lower_bound"]), 4),
            "upper_bound": round(float(row["upper_bound"]), 4),
            "model_version": "prophet-v1",
            "region": REGION,
        })

    sb.table("forecasts").upsert(
        records, on_conflict="forecast_time,region"
    ).execute()

    log.info("Upserted %d forecast records", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# 3. Green Window Detection
# ---------------------------------------------------------------------------

def find_green_windows(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Find the top N contiguous windows of GREEN_WINDOW_HOURS length."""
    if len(forecast_df) < GREEN_WINDOW_HOURS:
        return pd.DataFrame()

    df = forecast_df.sort_values("forecast_time").reset_index(drop=True)
    vals = df["renewable_percentage_predicted"].values
    times = df["forecast_time"].values

    # Sliding window average
    windows = []
    for i in range(len(vals) - GREEN_WINDOW_HOURS + 1):
        avg = float(np.mean(vals[i : i + GREEN_WINDOW_HOURS]))
        windows.append({
            "window_start": pd.Timestamp(times[i]),
            "window_end": pd.Timestamp(times[i + GREEN_WINDOW_HOURS - 1]) + pd.Timedelta(hours=1),
            "avg_renewable_percentage": round(avg, 4),
        })

    windows_df = pd.DataFrame(windows)
    windows_df = windows_df.sort_values(
        "avg_renewable_percentage", ascending=False
    ).head(GREEN_WINDOW_COUNT)
    windows_df["rank"] = range(1, len(windows_df) + 1)
    windows_df["region"] = REGION

    log.info("Found %d green windows", len(windows_df))
    return windows_df.reset_index(drop=True)


def upsert_green_windows(sb: Client, df: pd.DataFrame) -> int:
    """Write green windows to Supabase (replace current set)."""
    if df.empty:
        return 0

    # Delete stale windows for this region, then insert fresh
    sb.table("green_windows").delete().eq("region", REGION).execute()

    records = []
    for _, row in df.iterrows():
        records.append({
            "window_start": row["window_start"].isoformat(),
            "window_end": row["window_end"].isoformat(),
            "avg_renewable_percentage": float(row["avg_renewable_percentage"]),
            "rank": int(row["rank"]),
            "region": REGION,
        })

    sb.table("green_windows").insert(records).execute()

    log.info("Inserted %d green windows", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# 4. Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline():
    run_id = str(uuid.uuid4())[:8]
    sb = supabase_client()

    # Log pipeline start
    sb.table("pipeline_runs").insert({
        "run_id": run_id,
        "status": "running",
    }).execute()

    try:
        # Step 1: Ingest
        log.info("=== Step 1: Data Ingestion ===")
        history_df = ingest_grid_data()
        records_ingested = upsert_history(sb, history_df)

        # Step 2: Forecast
        log.info("=== Step 2: Forecasting ===")
        forecast_df = train_and_forecast(history_df)
        forecast_hours = upsert_forecasts(sb, forecast_df)

        # Step 3: Green Windows
        log.info("=== Step 3: Green Window Detection ===")
        windows_df = find_green_windows(forecast_df)
        windows_found = upsert_green_windows(sb, windows_df)

        # Update pipeline run
        sb.table("pipeline_runs").update({
            "status": "success",
            "records_ingested": records_ingested,
            "forecast_hours": forecast_hours,
            "green_windows_found": windows_found,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("run_id", run_id).execute()

        log.info("=== Pipeline complete: %d ingested, %d forecast hours, %d green windows ===",
                 records_ingested, forecast_hours, windows_found)

    except Exception as exc:
        log.exception("Pipeline failed")
        sb.table("pipeline_runs").update({
            "status": "failed",
            "error_message": str(exc)[:500],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("run_id", run_id).execute()
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
