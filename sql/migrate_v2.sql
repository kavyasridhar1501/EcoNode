-- ============================================================
-- EcoNode v2 Migration: Run this if you already have v1 tables
-- ============================================================

-- Add new columns to grid_history
ALTER TABLE grid_history ADD COLUMN IF NOT EXISTS carbon_intensity_gco2kwh DOUBLE PRECISION;
ALTER TABLE grid_history ADD COLUMN IF NOT EXISTS temperature_c DOUBLE PRECISION;
ALTER TABLE grid_history ADD COLUMN IF NOT EXISTS cloud_cover_pct DOUBLE PRECISION;
ALTER TABLE grid_history ADD COLUMN IF NOT EXISTS wind_speed_ms DOUBLE PRECISION;

-- Add carbon intensity to forecasts
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS carbon_intensity_gco2kwh DOUBLE PRECISION;

-- Add carbon intensity to green_windows
ALTER TABLE green_windows ADD COLUMN IF NOT EXISTS avg_carbon_intensity DOUBLE PRECISION;

-- New indexes for multi-region queries
CREATE INDEX IF NOT EXISTS idx_grid_history_region
    ON grid_history (region, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_region
    ON forecasts (region, forecast_time DESC);

-- Model evaluation metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id              BIGSERIAL PRIMARY KEY,
    region          TEXT NOT NULL,
    run_date        DATE NOT NULL,
    mae             DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    mape            DOUBLE PRECISION,
    coverage_80     DOUBLE PRECISION,
    sample_size     INTEGER,
    model_version   TEXT NOT NULL DEFAULT 'prophet-v2-weather',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_metrics_region_date UNIQUE (region, run_date)
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_region
    ON model_metrics (region, run_date DESC);

-- RLS for model_metrics
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access" ON model_metrics FOR SELECT USING (true);
CREATE POLICY "Service write access" ON model_metrics FOR ALL USING (true) WITH CHECK (true);
