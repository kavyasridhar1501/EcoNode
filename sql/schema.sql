-- ============================================================
-- EcoNode: Carbon-Aware Forecasting for AWS Jupyter Workflows
-- Supabase PostgreSQL Schema
-- ============================================================

-- Historical grid generation data from EIA
CREATE TABLE IF NOT EXISTS grid_history (
    id              BIGSERIAL PRIMARY KEY,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    total_generation_mwh   DOUBLE PRECISION,
    wind_generation_mwh    DOUBLE PRECISION,
    solar_generation_mwh   DOUBLE PRECISION,
    renewable_percentage   DOUBLE PRECISION,
    region          TEXT NOT NULL DEFAULT 'US48',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_grid_history_ts_region UNIQUE (timestamp_utc, region)
);

CREATE INDEX IF NOT EXISTS idx_grid_history_ts
    ON grid_history (timestamp_utc DESC);

-- 48-hour renewable energy forecasts
CREATE TABLE IF NOT EXISTS forecasts (
    id              BIGSERIAL PRIMARY KEY,
    forecast_time   TIMESTAMPTZ NOT NULL,
    renewable_percentage_predicted DOUBLE PRECISION NOT NULL,
    lower_bound     DOUBLE PRECISION,
    upper_bound     DOUBLE PRECISION,
    model_version   TEXT NOT NULL DEFAULT 'v1',
    region          TEXT NOT NULL DEFAULT 'US48',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_forecast_ts_region UNIQUE (forecast_time, region)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_time
    ON forecasts (forecast_time DESC);

-- Green compute windows (optimal scheduling blocks)
CREATE TABLE IF NOT EXISTS green_windows (
    id              BIGSERIAL PRIMARY KEY,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    avg_renewable_percentage DOUBLE PRECISION NOT NULL,
    rank            INTEGER NOT NULL,
    region          TEXT NOT NULL DEFAULT 'US48',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_green_window_rank_region UNIQUE (rank, region, created_at)
);

CREATE INDEX IF NOT EXISTS idx_green_windows_start
    ON green_windows (window_start DESC);

-- Pipeline run metadata for observability
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    records_ingested INTEGER DEFAULT 0,
    forecast_hours   INTEGER DEFAULT 0,
    green_windows_found INTEGER DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- ============================================================
-- Row Level Security (RLS) Policies for Supabase
-- Enable public read access via the anon key for the dashboard
-- ============================================================

ALTER TABLE grid_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE green_windows ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access" ON grid_history
    FOR SELECT USING (true);

CREATE POLICY "Public read access" ON forecasts
    FOR SELECT USING (true);

CREATE POLICY "Public read access" ON green_windows
    FOR SELECT USING (true);

CREATE POLICY "Public read access" ON pipeline_runs
    FOR SELECT USING (true);

-- Service role (used by pipeline) can insert/update
CREATE POLICY "Service write access" ON grid_history
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service write access" ON forecasts
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service write access" ON green_windows
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service write access" ON pipeline_runs
    FOR ALL USING (true) WITH CHECK (true);
