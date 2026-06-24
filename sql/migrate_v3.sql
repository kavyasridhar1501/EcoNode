-- ============================================================
-- EcoNode v3 Migration: Skill score + carbon savings
-- Run this after v2 migration is already applied
-- ============================================================

-- Add skill score to model_metrics
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS skill_score DOUBLE PRECISION;

-- Add carbon savings to pipeline_runs
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS carbon_savings_pct DOUBLE PRECISION;
