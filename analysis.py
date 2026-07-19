"""
EcoNode Analysis: statistical diagnostics computed fresh on every pipeline run.

Replaces the old EDA notebook. Notebooks don't run themselves and don't ship
their output anywhere a user (or recruiter) can see it — this module runs as
part of the daily pipeline, and every number it produces is written to
Supabase's `region_insights` table so the dashboard can render it.

Functions here are pure (dataframe/array in, plain values out) so they're
cheap to unit test without touching the network or a database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def hourly_profile(df: pd.DataFrame) -> pd.Series:
    """Mean renewable % by UTC hour of day."""
    if df.empty:
        return pd.Series(dtype=float)
    d = df.copy()
    d["hour"] = pd.to_datetime(d["timestamp_utc"]).dt.hour
    return d.groupby("hour")["renewable_percentage"].mean()


def weekday_profile(df: pd.DataFrame) -> pd.Series:
    """Mean renewable % by day of week, Monday first."""
    if df.empty:
        return pd.Series(dtype=float)
    d = df.copy()
    d["weekday"] = pd.to_datetime(d["timestamp_utc"]).dt.day_name()
    return d.groupby("weekday")["renewable_percentage"].mean().reindex(WEEKDAY_ORDER)


def weather_correlations(df: pd.DataFrame) -> dict:
    """Pearson correlation of renewable % against each available weather variable."""
    cols = ["temperature_c", "cloud_cover_pct", "wind_speed_ms"]
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        sub = df[["renewable_percentage", c]].dropna()
        if len(sub) >= 30:
            out[c] = float(sub["renewable_percentage"].corr(sub[c]))
    return out


def lag1_autocorrelation(df: pd.DataFrame) -> float | None:
    """Hour-to-hour autocorrelation of renewable % — a proxy for forecastability.

    A smooth, solar-driven diurnal curve is highly self-similar hour to hour
    (high autocorrelation). A wind-dominant mix ramps faster and less
    predictably, so this tends to run lower. Useful for explaining *why* one
    region's forecast MAE is worse than another's, rather than just reporting
    the gap.
    """
    if df.empty:
        return None
    s = df.sort_values("timestamp_utc")["renewable_percentage"].dropna()
    if len(s) < 50:
        return None
    value = s.autocorr(lag=1)
    return float(value) if value is not None and not np.isnan(value) else None


def climatology_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict | None:
    """Predict each test hour as the historical mean for its (weekday, hour) bucket.

    A second, near-free baseline alongside the persistence ("repeat yesterday")
    baseline already used for the skill score. If Prophet can't beat a lookup
    table this simple, the weather regressors and Fourier seasonality aren't
    earning their keep for that region.
    """
    if train_df.empty or test_df.empty:
        return None

    train = train_df.copy()
    test = test_df.copy()
    train["hour"] = pd.to_datetime(train["timestamp_utc"]).dt.hour
    train["weekday"] = pd.to_datetime(train["timestamp_utc"]).dt.dayofweek
    test["hour"] = pd.to_datetime(test["timestamp_utc"]).dt.hour
    test["weekday"] = pd.to_datetime(test["timestamp_utc"]).dt.dayofweek

    lookup = train.groupby(["weekday", "hour"])["renewable_percentage"].mean()
    overall_mean = float(train["renewable_percentage"].mean())

    preds = test.apply(
        lambda r: lookup.get((r["weekday"], r["hour"]), overall_mean), axis=1
    ).values
    actual = test["renewable_percentage"].values.astype(float)

    if len(actual) < 3:
        return None

    errors = preds - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def bootstrap_skill_ci(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap a confidence interval on the skill score.

    skill_score = 1 - MAE_model / MAE_baseline

    A single point estimate ("23% better than baseline") says nothing about
    whether that's distinguishable from sampling noise on a ~168-point
    holdout week. This resamples paired (model, baseline) errors with
    replacement to get a distribution of skill scores, then reports the
    `ci` interval (default 90%) around it.
    """
    model_errors = np.asarray(model_errors, dtype=float)
    baseline_errors = np.asarray(baseline_errors, dtype=float)
    n = len(model_errors)
    if n < 10 or len(baseline_errors) != n:
        return None

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        model_mae = np.mean(np.abs(model_errors[idx]))
        baseline_mae = np.mean(np.abs(baseline_errors[idx]))
        if baseline_mae > 0:
            scores.append(1.0 - model_mae / baseline_mae)

    if len(scores) < n_boot // 2:
        return None

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return float(np.percentile(scores, lo_pct)), float(np.percentile(scores, hi_pct))


def co2_savings_example(
    avg_carbon_all: float | None,
    avg_carbon_green: float | None,
    power_watts: float = 300,
    hours: float = 4,
) -> float | None:
    """Translate an abstract carbon-intensity delta into an absolute number.

    Answers: how much CO2 does scheduling one example workload (default: a
    300W GPU job for 4 hours — a single training/inference batch job) into
    the green window actually save, versus running it at the grid average?
    Returned in kilograms.
    """
    if avg_carbon_all is None or avg_carbon_green is None:
        return None
    delta_gco2_per_kwh = avg_carbon_all - avg_carbon_green
    kwh = (power_watts / 1000.0) * hours
    return round(delta_gco2_per_kwh * kwh / 1000.0, 4)


def forecastability_note(region_name: str, autocorr: float | None, mae: float | None) -> str | None:
    """Human-readable diagnostic tying forecast error to grid generation mix."""
    if autocorr is None or mae is None:
        return None
    if autocorr >= 0.85:
        tier = "high hour-to-hour persistence, consistent with a smooth, solar-driven diurnal cycle"
    elif autocorr >= 0.70:
        tier = "moderate hour-to-hour persistence"
    else:
        tier = (
            "low hour-to-hour persistence, consistent with a wind-dominant mix that ramps "
            "less predictably than a solar curve"
        )
    return f"{region_name}: {tier} (lag-1 autocorr={autocorr:.2f}); backtest MAE={mae:.2f}pp."


def build_region_insights(
    history_df: pd.DataFrame,
    region: str,
    region_name: str,
    backtest_metrics: dict | None,
    forecast_df: pd.DataFrame | None,
    green_windows_df: pd.DataFrame | None,
) -> dict:
    """Assemble one row of computed, per-region insights for storage + dashboard display."""
    hourly = hourly_profile(history_df)
    weekday = weekday_profile(history_df)
    corr = weather_correlations(history_df)
    autocorr = lag1_autocorrelation(history_df)

    peak_hour = int(hourly.idxmax()) if not hourly.empty else None
    peak_hour_pct = float(hourly.max()) if not hourly.empty else None
    best_weekday = weekday.idxmax() if not weekday.empty and weekday.notna().any() else None

    co2_example = None
    if forecast_df is not None and not forecast_df.empty and green_windows_df is not None and not green_windows_df.empty:
        avg_carbon_all = float(forecast_df["carbon_intensity_gco2kwh"].mean())
        avg_carbon_green = float(green_windows_df["avg_carbon_intensity"].mean())
        co2_example = co2_savings_example(avg_carbon_all, avg_carbon_green)

    mae = backtest_metrics.get("mae") if backtest_metrics else None
    note = forecastability_note(region_name, autocorr, mae)

    return {
        "region": region,
        "run_date": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "peak_hour_utc": peak_hour,
        "peak_hour_avg_pct": round(peak_hour_pct, 2) if peak_hour_pct is not None else None,
        "best_weekday": best_weekday,
        "temp_correlation": round(corr["temperature_c"], 3) if "temperature_c" in corr else None,
        "cloud_correlation": round(corr["cloud_cover_pct"], 3) if "cloud_cover_pct" in corr else None,
        "wind_correlation": round(corr["wind_speed_ms"], 3) if "wind_speed_ms" in corr else None,
        "autocorr_lag1": round(autocorr, 3) if autocorr is not None else None,
        "forecastability_note": note,
        "co2_saved_kg_example": co2_example,
        "co2_example_desc": "300W GPU job, 4h, green window vs grid average" if co2_example is not None else None,
    }
