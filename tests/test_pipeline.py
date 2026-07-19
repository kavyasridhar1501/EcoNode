import numpy as np
import pandas as pd
import pytest

import pipeline


def _hours(n, start="2026-01-01T00:00:00Z"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


class TestComputeCarbonIntensity:
    def test_zero_renewable_returns_full_factor(self):
        assert pipeline.compute_carbon_intensity(0, 500) == 500.0

    def test_full_renewable_returns_zero(self):
        assert pipeline.compute_carbon_intensity(100, 500) == 0.0

    def test_midpoint(self):
        assert pipeline.compute_carbon_intensity(50, 500) == 250.0

    def test_clamps_out_of_range_inputs(self):
        # Physically impossible inputs (e.g. a bad upstream forecast) shouldn't
        # produce a negative carbon intensity or one above the region's factor.
        assert pipeline.compute_carbon_intensity(-10, 500) == 500.0
        assert pipeline.compute_carbon_intensity(150, 500) == 0.0


class TestValidateData:
    def _base_df(self, n=20):
        return pd.DataFrame({
            "timestamp_utc": _hours(n),
            "total_generation_mwh": np.full(n, 1000.0),
            "wind_generation_mwh": np.full(n, 200.0),
            "solar_generation_mwh": np.full(n, 100.0),
            "renewable_percentage": np.full(n, 30.0),
        })

    def test_removes_negative_generation_rows(self):
        df = self._base_df()
        df.loc[0, "wind_generation_mwh"] = -5
        out = pipeline.validate_data(df, "TEST")
        assert len(out) == len(df) - 1

    def test_removes_zero_total_generation_rows(self):
        df = self._base_df()
        df.loc[0, "total_generation_mwh"] = 0
        out = pipeline.validate_data(df, "TEST")
        assert len(out) == len(df) - 1

    def test_removes_statistical_outliers(self):
        df = self._base_df(n=30)
        df.loc[0, "renewable_percentage"] = 99999.0
        out = pipeline.validate_data(df, "TEST")
        assert 99999.0 not in out["renewable_percentage"].values

    def test_clips_renewable_percentage_to_0_100(self):
        df = self._base_df(n=5)
        df.loc[0, "renewable_percentage"] = 150.0
        df.loc[1, "renewable_percentage"] = -10.0
        out = pipeline.validate_data(df, "TEST")
        assert out["renewable_percentage"].between(0, 100).all()

    def test_empty_input_returns_empty(self):
        out = pipeline.validate_data(pd.DataFrame(), "TEST")
        assert out.empty


class TestFindGreenWindows:
    def _forecast_df(self, values):
        n = len(values)
        return pd.DataFrame({
            "forecast_time": _hours(n),
            "renewable_percentage_predicted": values,
            "carbon_intensity_gco2kwh": [500 - v * 5 for v in values],
        })

    def test_picks_the_highest_average_window(self):
        # A clear 4h peak from hour 10-13 should win rank 1.
        values = [10] * 10 + [80, 85, 90, 88] + [10] * 34
        df = self._forecast_df(values)
        windows = pipeline.find_green_windows(df)
        assert len(windows) == 3
        assert windows.iloc[0]["rank"] == 1
        best = windows.iloc[0]
        assert best["avg_renewable_percentage"] == pytest.approx(np.mean([80, 85, 90, 88]))

    def test_returns_empty_when_insufficient_data(self):
        df = self._forecast_df([50, 60])  # fewer than GREEN_WINDOW_HOURS
        assert pipeline.find_green_windows(df).empty

    def test_windows_are_ranked_descending(self):
        values = list(range(48))  # monotonically increasing -> later windows win
        df = self._forecast_df(values)
        windows = pipeline.find_green_windows(df)
        pcts = windows["avg_renewable_percentage"].tolist()
        assert pcts == sorted(pcts, reverse=True)


class TestRegionsConfig:
    REQUIRED_KEYS = {
        "name", "lat", "lon", "carbon_factor", "changepoint_prior_scale",
        "seasonality_mode", "interval_width", "daily_fourier", "weekly_fourier",
    }

    def test_every_region_has_required_keys(self):
        for code, config in pipeline.REGIONS.items():
            missing = self.REQUIRED_KEYS - config.keys()
            assert not missing, f"{code} missing {missing}"

    def test_region_codes_are_unique_and_uppercase(self):
        codes = list(pipeline.REGIONS.keys())
        assert len(codes) == len(set(codes))
        assert all(c == c.upper() for c in codes)

    def test_coordinates_are_within_continental_us(self):
        for code, config in pipeline.REGIONS.items():
            assert 24 <= config["lat"] <= 50, code
            assert -125 <= config["lon"] <= -66, code
