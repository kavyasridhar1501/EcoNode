"""
EcoNode Pipeline v2: Multi-region ingestion, weather-enhanced Prophet forecasting,
carbon intensity estimation, model evaluation, and green window detection.

Runs entirely in GitHub Actions. Reads from EIA API v2 + Open-Meteo, writes to Supabase.
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

import analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("econode")

# Configuration
# Read lazily (not with os.environ[...] at import time) so this module — and its
# pure functions — can be imported and unit tested without real credentials set.
EIA_API_KEY = os.environ.get("EIA_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

EIA_BASE = "https://api.eia.gov/v2"
LOOKBACK_DAYS = 60
FORECAST_HOURS = 48
GREEN_WINDOW_HOURS = 4
GREEN_WINDOW_COUNT = 3

FUEL_SOLAR = "SUN"
FUEL_WIND = "WND"

REGIONS = {
    "US48": {
        "name": "Lower 48 (US)",
        "lat": 39.10,
        "lon": -94.58,
        "carbon_factor": 550,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.70,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "PJM": {
        "name": "Mid-Atlantic (PJM)",
        "lat": 39.95,
        "lon": -75.16,
        "carbon_factor": 500,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.90,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "ERCO": {
        "name": "Texas (ERCOT)",
        "lat": 29.76,
        "lon": -95.37,
        "carbon_factor": 500,
        "changepoint_prior_scale": 0.08,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.70,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "CISO": {
        "name": "California (CAISO)",
        "lat": 38.58,
        "lon": -121.49,
        "carbon_factor": 400,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "MISO": {
        "name": "Midwest (MISO)",
        "lat": 39.77,
        "lon": -86.16,
        "carbon_factor": 520,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "ISNE": {
        "name": "New England (ISO-NE)",
        "lat": 42.36,
        "lon": -71.06,
        "carbon_factor": 350,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "NYIS": {
        "name": "New York (NYISO)",
        "lat": 42.65,
        "lon": -73.75,
        "carbon_factor": 380,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "SWPP": {
        "name": "Southwest Power Pool",
        "lat": 34.75,
        "lon": -92.29,
        "carbon_factor": 480,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "SOCO": {
        "name": "Southern Company (Southeast)",
        "lat": 33.75,
        "lon": -84.39,
        "carbon_factor": 550,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
    "BPAT": {
        "name": "Pacific Northwest (BPA)",
        "lat": 45.52,
        "lon": -122.68,
        # BPA's non-wind/solar generation is dominated by hydro, not fossil —
        # this pipeline only counts wind+solar as "renewable," so a flat
        # fossil-typical factor here would overstate BPA's real carbon
        # intensity. Set low to reflect the hydro-heavy residual mix.
        "carbon_factor": 90,
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "multiplicative",
        "interval_width": 0.80,
        "daily_fourier": 10,
        "weekly_fourier": 3,
    },
}

OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# 1. EIA Data Ingestion

def fetch_eia_series(fuel_type: str | None, start: str, end: str, region: str) -> pd.DataFrame:
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
        "facets[respondent][]": region,
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


# 2. Weather Data (Open-Meteo, free, no API key)

def fetch_weather_historical(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_HISTORICAL, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        if not hourly or not hourly.get("time"):
            return pd.DataFrame()
        return pd.DataFrame({
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "temperature_c": hourly.get("temperature_2m"),
            "cloud_cover_pct": hourly.get("cloud_cover"),
            "wind_speed_ms": hourly.get("wind_speed_10m"),
        })
    except Exception as e:
        log.warning("Weather historical fetch failed: %s", e)
        return pd.DataFrame()


def fetch_weather_forecast(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m",
        "forecast_days": 3,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        if not hourly or not hourly.get("time"):
            return pd.DataFrame()
        return pd.DataFrame({
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "temperature_c": hourly.get("temperature_2m"),
            "cloud_cover_pct": hourly.get("cloud_cover"),
            "wind_speed_ms": hourly.get("wind_speed_10m"),
        })
    except Exception as e:
        log.warning("Weather forecast fetch failed: %s", e)
        return pd.DataFrame()


# 3. Data Quality Checks

def validate_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df.empty:
        return df

    initial_count = len(df)

    neg_mask = (
        (df["total_generation_mwh"] < 0)
        | (df["solar_generation_mwh"] < 0)
        | (df["wind_generation_mwh"] < 0)
    )
    if neg_mask.any():
        log.warning("[%s] Removing %d rows with negative generation", region, neg_mask.sum())
        df = df[~neg_mask]

    zero_mask = df["total_generation_mwh"] == 0
    if zero_mask.any():
        log.warning("[%s] Removing %d rows with zero total generation", region, zero_mask.sum())
        df = df[~zero_mask]

    if len(df) > 10:
        mean_pct = df["renewable_percentage"].mean()
        std_pct = df["renewable_percentage"].std()
        if std_pct > 0:
            outlier_mask = (
                (df["renewable_percentage"] < mean_pct - 3 * std_pct)
                | (df["renewable_percentage"] > mean_pct + 3 * std_pct)
            )
            if outlier_mask.any():
                log.warning("[%s] Removing %d outlier rows (>3 sigma)", region, outlier_mask.sum())
                df = df[~outlier_mask]

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if len(df) > 1:
        time_diffs = df["timestamp_utc"].diff().dt.total_seconds() / 3600
        gaps = time_diffs[time_diffs > 1.5]
        if len(gaps) > 0:
            log.warning(
                "[%s] Found %d time gaps (max gap: %.1fh)", region, len(gaps), gaps.max()
            )

    df["renewable_percentage"] = df["renewable_percentage"].clip(0, 100)

    removed = initial_count - len(df)
    if removed > 0:
        log.info("[%s] Data quality: %d -> %d rows (%d removed)", region, initial_count, len(df), removed)
    else:
        log.info("[%s] Data quality check passed: %d rows clean", region, len(df))

    return df.reset_index(drop=True)


# 4. Carbon Intensity

def compute_carbon_intensity(renewable_pct: float, carbon_factor: float) -> float:
    pct = max(0.0, min(100.0, renewable_pct))
    return round(carbon_factor * (1.0 - pct / 100.0), 2)


# 5. Full Ingestion (EIA + Weather + Quality)

def ingest_grid_data(region: str, config: dict) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_str = start_dt.strftime("%Y-%m-%dT%H")
    end_str = end_dt.strftime("%Y-%m-%dT%H")

    log.info("[%s] Fetching EIA data %s to %s", region, start_str, end_str)

    df_solar = fetch_eia_series(FUEL_SOLAR, start_str, end_str, region)
    df_wind = fetch_eia_series(FUEL_WIND, start_str, end_str, region)
    df_all = fetch_eia_series(None, start_str, end_str, region)

    if df_all.empty:
        log.warning("[%s] No generation data returned from EIA", region)
        return pd.DataFrame()

    total = df_all.groupby("period")["value"].sum().reset_index()
    total.columns = ["timestamp_utc", "total_generation_mwh"]

    def agg_fuel(df, col_name):
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

    merged["region"] = region
    merged["carbon_intensity_gco2kwh"] = merged["renewable_percentage"].apply(
        lambda pct: compute_carbon_intensity(pct, config["carbon_factor"])
    )

    weather = fetch_weather_historical(
        config["lat"],
        config["lon"],
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )
    if not weather.empty:
        merged = merged.merge(weather, on="timestamp_utc", how="left")
        log.info("[%s] Merged %d weather records", region, len(weather))
    else:
        merged["temperature_c"] = np.nan
        merged["cloud_cover_pct"] = np.nan
        merged["wind_speed_ms"] = np.nan

    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged = validate_data(merged, region)

    log.info("[%s] Ingested %d hourly records", region, len(merged))
    return merged


def upsert_history(sb: Client, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        rec = {
            "timestamp_utc": row["timestamp_utc"].isoformat(),
            "total_generation_mwh": float(row["total_generation_mwh"]),
            "wind_generation_mwh": float(row["wind_generation_mwh"]),
            "solar_generation_mwh": float(row["solar_generation_mwh"]),
            "renewable_percentage": round(float(row["renewable_percentage"]), 4),
            "carbon_intensity_gco2kwh": float(row.get("carbon_intensity_gco2kwh", 0)),
            "region": row["region"],
        }
        for col in ("temperature_c", "cloud_cover_pct", "wind_speed_ms"):
            if pd.notna(row.get(col)):
                rec[col] = round(float(row[col]), 2)
        records.append(rec)

    batch_size = 500
    for i in range(0, len(records), batch_size):
        sb.table("grid_history").upsert(
            records[i : i + batch_size], on_conflict="timestamp_utc,region"
        ).execute()

    log.info("[%s] Upserted %d history records", df["region"].iloc[0], len(records))
    return len(records)


# 6. Weather-Enhanced Forecasting

def _logit(y: np.ndarray) -> np.ndarray:
    p = np.clip(y / 100.0, 0.005, 0.995)
    return np.log(p / (1 - p))


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-x))


def train_and_forecast(df: pd.DataFrame, region: str, config: dict) -> pd.DataFrame:
    if len(df) < 48:
        log.warning("[%s] Insufficient data (%d rows, need >= 48)", region, len(df))
        return pd.DataFrame()

    prophet_df = df[["timestamp_utc", "renewable_percentage"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    use_logit = config.get("use_logit", False)
    if use_logit:
        prophet_df["y"] = _logit(prophet_df["y"].values)

    has_weather = (
        "temperature_c" in df.columns
        and df["temperature_c"].notna().sum() > len(df) * 0.5
    )

    weather_cols = ["temperature_c", "cloud_cover_pct", "wind_speed_ms"]

    if has_weather:
        for col in weather_cols:
            prophet_df[col] = df[col].ffill().bfill().values

    prophet_df["hour"] = prophet_df["ds"].dt.hour

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
        seasonality_mode=config.get("seasonality_mode", "multiplicative"),
        interval_width=config.get("interval_width", 0.80),
        n_changepoints=config.get("n_changepoints", 25),
    )

    model.add_seasonality(
        name="daily", period=1, fourier_order=config.get("daily_fourier", 10)
    )
    model.add_seasonality(
        name="weekly", period=7, fourier_order=config.get("weekly_fourier", 3)
    )
    model.add_regressor("hour")

    if has_weather:
        for col in weather_cols:
            model.add_regressor(col)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq="h")
    future["hour"] = future["ds"].dt.hour

    if has_weather:
        weather_fc = fetch_weather_forecast(config["lat"], config["lon"])
        if not weather_fc.empty:
            weather_fc["ds"] = weather_fc["timestamp_utc"].dt.tz_localize(None)
            future = future.merge(
                weather_fc[["ds"] + weather_cols], on="ds", how="left"
            )
            train_map = prophet_df.set_index("ds")[weather_cols]
            for col in weather_cols:
                mask = future[col].isna()
                future.loc[mask, col] = future.loc[mask, "ds"].map(train_map[col])
        else:
            for col in weather_cols:
                future[col] = prophet_df[col].iloc[-1]

        for col in weather_cols:
            future[col] = future[col].ffill().bfill()

    forecast = model.predict(future)

    cutoff = prophet_df["ds"].max()
    fut = forecast[forecast["ds"] > cutoff].copy()

    if use_logit:
        yhat = _inv_logit(fut["yhat"].values)
        yhat_lower = _inv_logit(fut["yhat_lower"].values)
        yhat_upper = _inv_logit(fut["yhat_upper"].values)
    else:
        yhat = fut["yhat"].clip(0, 100).values
        yhat_lower = fut["yhat_lower"].clip(0, 100).values
        yhat_upper = fut["yhat_upper"].clip(0, 100).values

    carbon_factor = config["carbon_factor"]
    result = pd.DataFrame({
        "forecast_time": pd.to_datetime(fut["ds"], utc=True),
        "renewable_percentage_predicted": yhat,
        "lower_bound": yhat_lower,
        "upper_bound": yhat_upper,
        "carbon_intensity_gco2kwh": pd.Series(yhat).apply(
            lambda p: compute_carbon_intensity(p, carbon_factor)
        ).values,
    })

    log.info("[%s] Generated %d-hour forecast (weather=%s)", region, len(result), has_weather)
    return result.reset_index(drop=True)


def upsert_forecasts(sb: Client, df: pd.DataFrame, region: str) -> int:
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "forecast_time": row["forecast_time"].isoformat(),
            "renewable_percentage_predicted": round(float(row["renewable_percentage_predicted"]), 4),
            "lower_bound": round(float(row["lower_bound"]), 4),
            "upper_bound": round(float(row["upper_bound"]), 4),
            "carbon_intensity_gco2kwh": round(float(row["carbon_intensity_gco2kwh"]), 2),
            "model_version": "prophet-v2-weather",
            "region": region,
        })

    sb.table("forecasts").upsert(records, on_conflict="forecast_time,region").execute()

    log.info("[%s] Upserted %d forecast records", region, len(records))
    return len(records)


# 7. Model Evaluation (Forecast vs Actuals)

def evaluate_model(sb: Client, region: str) -> dict | None:
    try:
        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(hours=24)).isoformat()
        two_days_ago = (now - timedelta(hours=48)).isoformat()
        now_iso = now.isoformat()

        forecasts_resp = (
            sb.table("forecasts")
            .select("forecast_time,renewable_percentage_predicted,lower_bound,upper_bound")
            .eq("region", region)
            .gte("forecast_time", yesterday)
            .lte("forecast_time", now_iso)
            .execute()
        )

        if not forecasts_resp.data:
            log.info("[%s] No past forecasts for evaluation", region)
            return None

        actuals_resp = (
            sb.table("grid_history")
            .select("timestamp_utc,renewable_percentage")
            .eq("region", region)
            .gte("timestamp_utc", two_days_ago)
            .lte("timestamp_utc", now_iso)
            .execute()
        )

        if not actuals_resp.data:
            log.info("[%s] No actuals for evaluation", region)
            return None

        fc = pd.DataFrame(forecasts_resp.data)
        ac = pd.DataFrame(actuals_resp.data)

        log.info("[%s] Evaluation: %d forecasts, %d actuals fetched", region, len(fc), len(ac))

        fc["hour"] = pd.to_datetime(fc["forecast_time"], utc=True).dt.floor("h")
        ac["hour"] = pd.to_datetime(ac["timestamp_utc"], utc=True).dt.floor("h")

        merged = fc.merge(ac, on="hour", suffixes=("_pred", "_actual"))
        log.info("[%s] Evaluation: %d matched points after merge", region, len(merged))

        if len(merged) < 3:
            log.info("[%s] Too few matching points (%d) — need at least 3", region, len(merged))
            return None

        predicted = merged["renewable_percentage_predicted"].values.astype(float)
        actual = merged["renewable_percentage"].values.astype(float)
        errors = predicted - actual

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        nonzero = actual != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
            if nonzero.sum() > 0
            else None
        )

        lower = merged["lower_bound"].values.astype(float)
        upper = merged["upper_bound"].values.astype(float)
        in_bounds = (actual >= lower) & (actual <= upper)
        coverage = float(in_bounds.mean() * 100)

        # Persistence baseline: predict each hour = actual from 24h ago
        ac_lookup = ac.set_index("hour")["renewable_percentage"]
        merged["baseline_pred"] = (merged["hour"] - pd.Timedelta(hours=24)).map(ac_lookup)

        has_baseline = merged["baseline_pred"].notna().sum() >= 3
        if has_baseline:
            bm = merged.dropna(subset=["baseline_pred"])
            baseline_errors = bm["baseline_pred"].values - bm["renewable_percentage"].values.astype(float)
            baseline_mae = float(np.mean(np.abs(baseline_errors)))
            skill_score = round(1.0 - (mae / baseline_mae), 4) if baseline_mae > 0 else None
        else:
            skill_score = None

        log.info(
            "[%s] Skill score: %s (model MAE=%.2f vs baseline MAE=%.2f)",
            region,
            f"{skill_score:.2f}" if skill_score is not None else "N/A",
            mae,
            baseline_mae if has_baseline else 0,
        )

        metrics = {
            "region": region,
            "run_date": now.strftime("%Y-%m-%d"),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "coverage_80": round(coverage, 2),
            "sample_size": len(merged),
            "skill_score": skill_score,
            "model_version": "prophet-v2-weather",
        }

        sb.table("model_metrics").upsert(metrics, on_conflict="region,run_date").execute()

        log.info(
            "[%s] Evaluation: MAE=%.2f RMSE=%.2f Coverage=%.1f%% Skill=%.2f (n=%d)",
            region, mae, rmse, coverage,
            skill_score if skill_score is not None else 0,
            len(merged),
        )
        return metrics

    except Exception as e:
        log.warning("[%s] Evaluation failed: %s", region, e)
        return None


# 7b. Holdout Backtest (instant metrics, no 1-day lag)

def backtest_model(history_df: pd.DataFrame, region: str, config: dict, sb: Client) -> dict | None:
    """Hold out last 168h (1 week) of historical data, train on the rest, evaluate immediately."""
    try:
        df = history_df.sort_values("timestamp_utc").reset_index(drop=True)
        test_hours = 168

        if len(df) < test_hours + 72:
            log.warning("[%s] Not enough data for backtest (%d rows)", region, len(df))
            return None

        train = df.iloc[:-test_hours].copy()
        test = df.iloc[-test_hours:].copy()

        prophet_df = train[["timestamp_utc", "renewable_percentage"]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

        has_weather = (
            "temperature_c" in train.columns
            and train["temperature_c"].notna().sum() > len(train) * 0.5
        )
        weather_cols = ["temperature_c", "cloud_cover_pct", "wind_speed_ms"]

        if has_weather:
            for col in weather_cols:
                prophet_df[col] = train[col].ffill().bfill().values

        prophet_df["hour"] = prophet_df["ds"].dt.hour

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
            seasonality_mode=config.get("seasonality_mode", "multiplicative"),
            interval_width=config.get("interval_width", 0.80),
            n_changepoints=config.get("n_changepoints", 25),
        )
        model.add_seasonality(
            name="daily", period=1, fourier_order=config.get("daily_fourier", 10)
        )
        model.add_seasonality(
            name="weekly", period=7, fourier_order=config.get("weekly_fourier", 3)
        )
        model.add_regressor("hour")
        if has_weather:
            for col in weather_cols:
                model.add_regressor(col)

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=test_hours, freq="h")
        future["hour"] = future["ds"].dt.hour

        if has_weather:
            test_w = test[["timestamp_utc"] + weather_cols].copy()
            test_w["ds"] = test_w["timestamp_utc"].dt.tz_localize(None)
            future = future.merge(test_w[["ds"] + weather_cols], on="ds", how="left")
            train_map = prophet_df.set_index("ds")[weather_cols]
            for col in weather_cols:
                mask = future[col].isna()
                future.loc[mask, col] = future.loc[mask, "ds"].map(train_map[col])
            for col in weather_cols:
                future[col] = future[col].ffill().bfill()

        forecast = model.predict(future)

        cutoff = prophet_df["ds"].max()
        fut = forecast[forecast["ds"] > cutoff].head(test_hours)

        predicted = fut["yhat"].clip(0, 100).values
        lower = fut["yhat_lower"].clip(0, 100).values
        upper = fut["yhat_upper"].clip(0, 100).values
        actual = test["renewable_percentage"].values

        n = min(len(predicted), len(actual))
        if n < 3:
            log.warning("[%s] Backtest: too few points (%d)", region, n)
            return None

        predicted, lower, upper, actual = predicted[:n], lower[:n], upper[:n], actual[:n]

        errors = predicted - actual
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        nonzero = actual != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
            if nonzero.sum() > 0
            else None
        )
        in_bounds = (actual >= lower) & (actual <= upper)
        coverage = float(in_bounds.mean() * 100)

        baseline_start = len(df) - test_hours - 24
        baseline_end = baseline_start + n
        skill_ci = None
        if baseline_start >= 0 and baseline_end <= len(df):
            baseline_pred = df.iloc[baseline_start:baseline_end]["renewable_percentage"].values
            baseline_errors = baseline_pred - actual
            baseline_mae = float(np.mean(np.abs(baseline_errors)))
            skill_score = round(1.0 - (mae / baseline_mae), 4) if baseline_mae > 0 else None
            skill_ci = analysis.bootstrap_skill_ci(errors, baseline_errors)
        else:
            skill_score = None
            baseline_mae = 0

        # Second baseline: climatology (mean by weekday/hour bucket). If Prophet
        # can't beat this cheap lookup table, the weather regressors and Fourier
        # seasonality aren't earning their keep for this region.
        climatology = analysis.climatology_baseline(train, test.iloc[:n])
        climatology_mae = climatology["mae"] if climatology else None
        skill_vs_climatology = (
            round(1.0 - (mae / climatology_mae), 4)
            if climatology_mae and climatology_mae > 0
            else None
        )

        now = datetime.now(timezone.utc)
        metrics = {
            "region": region,
            "run_date": now.strftime("%Y-%m-%d"),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "coverage_80": round(coverage, 2),
            "sample_size": n,
            "skill_score": skill_score,
            "skill_score_ci_low": round(skill_ci[0], 4) if skill_ci else None,
            "skill_score_ci_high": round(skill_ci[1], 4) if skill_ci else None,
            "climatology_mae": round(climatology_mae, 4) if climatology_mae is not None else None,
            "skill_vs_climatology": skill_vs_climatology,
            "model_version": "prophet-v2-weather",
        }

        sb.table("model_metrics").upsert(metrics, on_conflict="region,run_date").execute()

        log.info(
            "[%s] Backtest: MAE=%.2f RMSE=%.2f Coverage=%.1f%% Skill=%.4f (CI %s) vs climatology=%.4f (n=%d)",
            region, mae, rmse, coverage, skill_score or 0,
            f"[{skill_ci[0]:.2f}, {skill_ci[1]:.2f}]" if skill_ci else "n/a",
            skill_vs_climatology or 0, n,
        )
        return metrics

    except Exception as e:
        log.warning("[%s] Backtest failed: %s", region, e)
        return None


# 8. Green Window Detection

def find_green_windows(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if len(forecast_df) < GREEN_WINDOW_HOURS:
        return pd.DataFrame()

    df = forecast_df.sort_values("forecast_time").reset_index(drop=True)
    vals = df["renewable_percentage_predicted"].values
    carbon = df["carbon_intensity_gco2kwh"].values
    times = df["forecast_time"].values

    windows = []
    for i in range(len(vals) - GREEN_WINDOW_HOURS + 1):
        s = slice(i, i + GREEN_WINDOW_HOURS)
        windows.append({
            "window_start": pd.Timestamp(times[i]),
            "window_end": pd.Timestamp(times[i + GREEN_WINDOW_HOURS - 1]) + pd.Timedelta(hours=1),
            "avg_renewable_percentage": round(float(np.mean(vals[s])), 4),
            "avg_carbon_intensity": round(float(np.mean(carbon[s])), 2),
        })

    wdf = pd.DataFrame(windows)
    wdf = wdf.sort_values("avg_renewable_percentage", ascending=False).head(GREEN_WINDOW_COUNT)
    wdf["rank"] = range(1, len(wdf) + 1)
    return wdf.reset_index(drop=True)


def upsert_green_windows(sb: Client, df: pd.DataFrame, region: str) -> int:
    if df.empty:
        return 0

    sb.table("green_windows").delete().eq("region", region).execute()

    records = []
    for _, row in df.iterrows():
        records.append({
            "window_start": row["window_start"].isoformat(),
            "window_end": row["window_end"].isoformat(),
            "avg_renewable_percentage": float(row["avg_renewable_percentage"]),
            "avg_carbon_intensity": float(row["avg_carbon_intensity"]),
            "rank": int(row["rank"]),
            "region": region,
        })

    sb.table("green_windows").insert(records).execute()
    log.info("[%s] Inserted %d green windows", region, len(records))
    return len(records)


# 8b. Region Insights (replaces the old, never-executed EDA notebook)

def upsert_region_insights(sb: Client, insights: dict) -> None:
    sb.table("region_insights").upsert(insights, on_conflict="region,run_date").execute()
    log.info(
        "[%s] Insights: peak_hour=%s best_weekday=%s autocorr=%s co2_saved_kg=%s",
        insights["region"], insights.get("peak_hour_utc"), insights.get("best_weekday"),
        insights.get("autocorr_lag1"), insights.get("co2_saved_kg_example"),
    )


# 9. Pipeline Orchestrator

def run_pipeline():
    if not EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY must be set")

    run_id = str(uuid.uuid4())[:8]
    sb = supabase_client()

    sb.table("pipeline_runs").insert({"run_id": run_id, "status": "running"}).execute()

    try:
        total_ingested = 0
        total_forecast = 0
        total_windows = 0
        carbon_savings_all = []

        for region, config in REGIONS.items():
            log.info("=" * 60)
            log.info("Region: %s (%s)", region, config["name"])
            log.info("=" * 60)

            log.info("[%s] Step 1: Ingest EIA + weather data", region)
            history_df = ingest_grid_data(region, config)
            total_ingested += upsert_history(sb, history_df)

            log.info("[%s] Step 2: Backtest evaluation (holdout last 48h)", region)
            backtest_metrics = backtest_model(history_df, region, config, sb)

            log.info("[%s] Step 3: Weather-enhanced forecasting", region)
            forecast_df = train_and_forecast(history_df, region, config)
            total_forecast += upsert_forecasts(sb, forecast_df, region)

            log.info("[%s] Step 4: Green window detection", region)
            windows_df = find_green_windows(forecast_df)
            total_windows += upsert_green_windows(sb, windows_df, region)

            log.info("[%s] Step 5: Compute region insights", region)
            insights = analysis.build_region_insights(
                history_df, region, config["name"], backtest_metrics, forecast_df, windows_df
            )
            upsert_region_insights(sb, insights)

            if not forecast_df.empty and not windows_df.empty:
                avg_carbon_all = float(forecast_df["carbon_intensity_gco2kwh"].mean())
                avg_carbon_green = float(windows_df["avg_carbon_intensity"].mean())
                if avg_carbon_all > 0:
                    savings_pct = round((avg_carbon_all - avg_carbon_green) / avg_carbon_all * 100, 1)
                    carbon_savings_all.append(savings_pct)
                    log.info(
                        "[%s] Carbon savings: %.1f%% (green=%.0f vs avg=%.0f gCO2/kWh)",
                        region, savings_pct, avg_carbon_green, avg_carbon_all,
                    )

        avg_savings = round(sum(carbon_savings_all) / len(carbon_savings_all), 1) if carbon_savings_all else None

        sb.table("pipeline_runs").update({
            "status": "success",
            "records_ingested": total_ingested,
            "forecast_hours": total_forecast,
            "green_windows_found": total_windows,
            "carbon_savings_pct": avg_savings,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("run_id", run_id).execute()

        log.info(
            "Pipeline complete: %d ingested, %d forecast hours, %d green windows across %d regions",
            total_ingested, total_forecast, total_windows, len(REGIONS),
        )

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
