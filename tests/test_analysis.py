import numpy as np
import pandas as pd
import pytest

import analysis


def _hours(n, start="2026-01-01T00:00:00Z"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def _solar_like_df(n=24 * 14):
    """A smooth diurnal curve (peaks midday, zero at night) repeated over weeks —
    should be highly autocorrelated at lag 1, like a solar-dominant region."""
    ts = _hours(n)
    hour = ts.hour.values
    pct = np.clip(40 * np.sin((hour - 6) / 24 * 2 * np.pi), 0, None)
    return pd.DataFrame({"timestamp_utc": ts, "renewable_percentage": pct})


def _noisy_df(n=24 * 14, seed=0):
    """Pure random noise — low/near-zero autocorrelation, like an unforecastable series."""
    rng = np.random.default_rng(seed)
    ts = _hours(n)
    pct = rng.uniform(0, 40, size=n)
    return pd.DataFrame({"timestamp_utc": ts, "renewable_percentage": pct})


class TestHourlyProfile:
    def test_peak_hour_matches_synthetic_curve(self):
        df = _solar_like_df()
        profile = analysis.hourly_profile(df)
        # sin((h-6)/24*2pi) peaks at h=12
        assert int(profile.idxmax()) == 12

    def test_empty_input(self):
        assert analysis.hourly_profile(pd.DataFrame()).empty


class TestWeekdayProfile:
    def test_returns_all_seven_days_in_order(self):
        df = _solar_like_df()
        profile = analysis.weekday_profile(df)
        assert list(profile.index) == analysis.WEEKDAY_ORDER


class TestWeatherCorrelations:
    def test_detects_strong_positive_correlation(self):
        n = 200
        temp = np.linspace(0, 30, n)
        df = pd.DataFrame({
            "renewable_percentage": temp * 2,  # perfectly correlated
            "temperature_c": temp,
        })
        corr = analysis.weather_correlations(df)
        assert corr["temperature_c"] == pytest.approx(1.0, abs=1e-6)

    def test_skips_columns_with_too_little_data(self):
        df = pd.DataFrame({
            "renewable_percentage": [10, 20, 30],
            "temperature_c": [1, 2, 3],
        })
        assert "temperature_c" not in analysis.weather_correlations(df)


class TestLag1Autocorrelation:
    def test_smooth_diurnal_series_is_highly_autocorrelated(self):
        df = _solar_like_df()
        r = analysis.lag1_autocorrelation(df)
        assert r > 0.9

    def test_pure_noise_has_low_autocorrelation(self):
        df = _noisy_df()
        r = analysis.lag1_autocorrelation(df)
        assert abs(r) < 0.3

    def test_insufficient_data_returns_none(self):
        df = _solar_like_df(n=10)
        assert analysis.lag1_autocorrelation(df) is None


class TestClimatologyBaseline:
    def test_perfect_repeating_pattern_gives_near_zero_error(self):
        # Same (weekday, hour) -> same value every week, train/test are
        # different weeks of the identical pattern.
        df = _solar_like_df(n=24 * 21)
        train = df.iloc[: 24 * 14]
        test = df.iloc[24 * 14 :]
        result = analysis.climatology_baseline(train, test)
        assert result["mae"] < 1.0

    def test_none_on_empty_input(self):
        assert analysis.climatology_baseline(pd.DataFrame(), pd.DataFrame()) is None


class TestBootstrapSkillCi:
    def test_model_much_better_than_baseline_gives_positive_ci(self):
        rng = np.random.default_rng(1)
        n = 200
        model_errors = rng.normal(0, 1, n)      # small errors
        baseline_errors = rng.normal(0, 10, n)  # large errors
        ci = analysis.bootstrap_skill_ci(model_errors, baseline_errors, n_boot=500)
        lo, hi = ci
        assert lo > 0  # confidently better than baseline
        assert lo <= hi

    def test_too_few_points_returns_none(self):
        assert analysis.bootstrap_skill_ci([1, 2], [1, 2]) is None

    def test_mismatched_lengths_returns_none(self):
        assert analysis.bootstrap_skill_ci([1] * 20, [1] * 10) is None


class TestCo2SavingsExample:
    def test_bigger_delta_means_more_savings(self):
        small = analysis.co2_savings_example(500, 480)
        large = analysis.co2_savings_example(500, 100)
        assert large > small > 0

    def test_none_when_inputs_missing(self):
        assert analysis.co2_savings_example(None, 100) is None

    def test_known_value(self):
        # delta = 200 gCO2/kWh, 0.3kW * 4h = 1.2kWh -> 240 gCO2 = 0.24 kg
        result = analysis.co2_savings_example(500, 300, power_watts=300, hours=4)
        assert result == pytest.approx(0.24)


class TestForecastabilityNote:
    def test_none_when_missing_inputs(self):
        assert analysis.forecastability_note("Region", None, 1.5) is None
        assert analysis.forecastability_note("Region", 0.9, None) is None

    def test_low_autocorrelation_flags_wind_dominant_language(self):
        note = analysis.forecastability_note("ERCOT", 0.5, 6.0)
        assert "wind" in note.lower()


class TestBuildRegionInsights:
    def test_assembles_expected_keys(self):
        history = _solar_like_df()
        forecast_df = pd.DataFrame({
            "carbon_intensity_gco2kwh": [400, 420, 410],
        })
        windows_df = pd.DataFrame({
            "avg_carbon_intensity": [200, 210],
        })
        insights = analysis.build_region_insights(
            history, "US48", "Lower 48 (US)",
            backtest_metrics={"mae": 1.5},
            forecast_df=forecast_df,
            green_windows_df=windows_df,
        )
        assert insights["region"] == "US48"
        assert insights["peak_hour_utc"] == 12
        assert insights["co2_saved_kg_example"] is not None
        assert insights["forecastability_note"] is not None
