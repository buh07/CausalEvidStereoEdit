# PREREG Addendum: May ARR Freeze (v1)

Date: 2026-05-25 (UTC)

This addendum prospectively freezes the corrected protocol used for the May ARR fixpack rerun. It does not retroactively alter the original preregistration; it records corrected confirmatory targets after identifying completion-format constraints.

## 1) Scope
- Model set (core confirmatory):
  - `google/gemma-2-2b`
  - `google/gemma-2-2b-it`
  - `meta-llama/Llama-3.2-3B`
- Frozen orchestration config: `configs/may_arr_freeze_v1.yaml`
- Frozen run family tag: single `RUN_TAG` produced by the tmux launcher.

## 2) Corrected protocol assumptions
- Completion-format causal-local interventions are evaluated at prediction position.
- Trait-position directions are allowed as cross-position causal objects.
- Prediction-position direction extraction in completion format is expected to collapse to zero by construction.

## 3) Confirmatory analyses
- Primary asymmetry endpoint: Exp16 `primary_inject_anti_minus_remove_stereo` (score endpoint, paired sign test).
- Secondary asymmetry endpoint: same contrast on margin endpoint (paired Wilcoxon).
- Inference policy:
  - Raw paired p-values are taken directly from Exp16 outputs.
  - Primary reported family: BH-FDR across models for the same named contrast.
  - Robustness families to report in parallel:
    - within-table BH (as implemented in Exp16 tables)
    - prereg-style Bonferroni across models.

## 4) Seed aggregation plan
- Fixed seeds: `11, 29, 47`.
- Fixed heldout size for core asymmetry reruns: `n=120`.
- Aggregate report fields:
  - pooled mean contrast across seeds
  - between-seed variance and SD
  - per-seed rows retained verbatim.

## 5) Transfer/equivalence framing
- SESOI fixed at `0.10`.
- Alpha fixed at `0.05`.
- Target power fixed at `0.80`.
- Exp21 outputs must include MDE approximation and power-vs-SESOI status.

## 6) Split hygiene and selection policy
- Exp02/Exp03 strict runs use `--split-scope train`.
- Exp09 adjudication uses Exp01 test split only.
- Exp09 truncation policy defaults to deterministic shuffle-before-truncate using fixed seed.

## 7) Exclusion rules
- Pair-level exclusions follow tokenizer-alignment and position-validity rules in each experiment script.
- Rows with undefined paired tests (e.g., zero-variance deltas) are retained with missing p/q, and excluded from multiplicity adjustments requiring finite p-values.

## 8) Reproducibility artifacts
- Final reported outputs must be tied to explicit run directories (no latest-run resolution).
- Required frozen artifacts:
  - run map JSON
  - artifact manifest JSON (with run dirs + commit + lockfile fingerprint)
  - checksum verification report.

## 9) Out-of-scope statements
- This addendum does not claim head-to-head mitigation efficacy against global baselines.
- Prompt baseline (Exp29) is calibration-only, not SOTA mitigation benchmarking.
