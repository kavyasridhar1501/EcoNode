-- ============================================================
-- EcoNode v4 Migration: Statistical rigor + real EDA insights
-- Run this after v3 migration is already applied
-- ============================================================

-- Bootstrap CI on skill score + a second (climatology) baseline for model_metrics
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS skill_score_ci_low DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS skill_score_ci_high DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS climatology_mae DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS skill_vs_climatology DOUBLE PRECISION;

-- Computed insights per region per day (replaces the old EDA notebook —
-- these are real numbers produced by analysis.py on every pipeline run)
CREATE TABLE IF NOT EXISTS region_insights (
    id                    BIGSERIAL PRIMARY KEY,
    region                TEXT NOT NULL,
    run_date              DATE NOT NULL,
    peak_hour_utc         INTEGER,
    peak_hour_avg_pct     DOUBLE PRECISION,
    best_weekday          TEXT,
    temp_correlation      DOUBLE PRECISION,
    cloud_correlation     DOUBLE PRECISION,
    wind_correlation      DOUBLE PRECISION,
    autocorr_lag1         DOUBLE PRECISION,
    forecastability_note  TEXT,
    co2_saved_kg_example  DOUBLE PRECISION,
    co2_example_desc      TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_region_insights_region_date UNIQUE (region, run_date)
);

CREATE INDEX IF NOT EXISTS idx_region_insights_region
    ON region_insights (region, run_date DESC);

ALTER TABLE region_insights ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access" ON region_insights FOR SELECT USING (true);
CREATE POLICY "Service write access" ON region_insights FOR ALL USING (true) WITH CHECK (true);
