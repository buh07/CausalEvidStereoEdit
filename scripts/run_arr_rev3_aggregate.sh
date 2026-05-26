#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
CONFIG_PATH="${CONFIG_PATH:-configs/arr_rev3_freeze.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATE_DIR="results/${RUN_TAG}/state"
REPORT_DIR="results/${RUN_TAG}/reports"
LOG_DIR="results/${RUN_TAG}/logs"
mkdir -p "$STATE_DIR" "$REPORT_DIR" "$LOG_DIR"
ORCH_START_UTC="$(cat "$STATE_DIR/orch_start_utc.txt")"
LOG_FILE="$LOG_DIR/aggregate_gpu3.log"

source ./scripts/may_arr_fixpack_common.sh

log_msg "Aggregator start | run_tag=$RUN_TAG"

MODEL_KEYS="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8'))
keys=[m['key'] for m in (cfg.get('core_models',[])+cfg.get('extension_models',[]))]
print(' '.join(keys))
PY
)"
for key in $MODEL_KEYS; do
  wait_for_flag "${key}_done"
  if [[ -f "$(flag_file "${key}_failed")" ]]; then
    log_msg "Detected failure flag for $key"
    exit 1
  fi
done

RUN_MAP_FULL="$STATE_DIR/run_map_full.json"
RUN_MAP_CORE="$STATE_DIR/run_map_core.json"

$PYTHON_BIN - "$STATE_DIR" "$CONFIG_PATH" "$RUN_TAG" "$RUN_MAP_FULL" "$RUN_MAP_CORE" <<'PY'
import json
import sys
from pathlib import Path
import yaml

state_dir = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
run_tag = sys.argv[3]
out_full = Path(sys.argv[4])
out_core = Path(sys.argv[5])

cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
core_cfg = cfg.get('core_models', [])
ext_cfg = cfg.get('extension_models', [])

KNOWN_SUFFIXES = [
    'exp01_mixed_run_dir',
    'exp01_stereoset_run_dir',
    'exp01_crows_run_dir',
    'exp02_strict_run_dir',
    'exp02_stereoset_run_dir',
    'exp02_crows_run_dir',
    'exp03_strict_run_dir',
    'exp09_strict_run_dir',
    'exp14_run_dir',
    'exp17_run_dir',
    'exp15_run_dir',
    'exp21_run_dir',
    'exp26_run_dir',
    'exp27_run_dir',
    'exp28_run_dir',
    'exp28_balanced_run_dir',
    'exp29_run_dir',
    'exp30_run_dir',
    'exp07_run_dir',
    'exp16_canonical_run_dir',
    'exp16_source_stereoset_run_dir',
    'exp16_source_crows_run_dir',
]


def read_txt(key: str, suffix: str) -> str:
    fp = state_dir / f"{key}_{suffix}.txt"
    return fp.read_text(encoding='utf-8').strip() if fp.exists() else ''


def collect_seed_runs(key: str, prefix: str):
    out = {}
    for fp in sorted(state_dir.glob(f"{key}_{prefix}_seed_*_run_dir.txt")):
        stem = fp.stem
        # e.g. gemma2b_exp14_crossfit_seed_101_run_dir
        parts = stem.split('_')
        if 'seed' not in parts:
            continue
        idx = parts.index('seed')
        if idx + 1 >= len(parts):
            continue
        seed = parts[idx + 1]
        out[str(seed)] = fp.read_text(encoding='utf-8').strip()
    return out


def build_payload(row, role: str):
    key = row['key']
    payload = {
        'key': key,
        'label': row.get('label', key),
        'model': row.get('model', ''),
        'gpu': row.get('gpu', ''),
        'role': role,
        'run_tag': run_tag,
    }
    for s in KNOWN_SUFFIXES:
        payload[s] = read_txt(key, s)
    payload['exp16_seed_runs'] = collect_seed_runs(key, 'exp16')
    payload['exp14_crossfit_seed_runs'] = collect_seed_runs(key, 'exp14_crossfit')
    payload['exp17_crossfit_seed_runs'] = collect_seed_runs(key, 'exp17_crossfit')
    return payload

models_full = {}
models_core = {}
for row in core_cfg:
    p = build_payload(row, role='core')
    models_full[row['model']] = p
    models_core[row['model']] = p
for row in ext_cfg:
    p = build_payload(row, role='extension')
    models_full[row['model']] = p

full = {
    'run_tag': run_tag,
    'config_path': str(cfg_path),
    'models': models_full,
}
core = {
    'run_tag': run_tag,
    'config_path': str(cfg_path),
    'models': models_core,
}
out_full.write_text(json.dumps(full, indent=2, sort_keys=True), encoding='utf-8')
out_core.write_text(json.dumps(core, indent=2, sort_keys=True), encoding='utf-8')
print(out_full)
print(out_core)
PY
log_msg "Wrote run maps: $RUN_MAP_FULL and $RUN_MAP_CORE"

# Core report suite
run_cmd "aggregate-seed-core" "$PYTHON_BIN tools/aggregate_seed_core.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR' --require-complete"
run_cmd "inference-robustness" "$PYTHON_BIN tools/build_inference_robustness_table.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "coverage-bias" "$PYTHON_BIN tools/coverage_bias_summary.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "multitoken-matched-summary" "$PYTHON_BIN tools/summarize_multitoken_matched_contrast.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "exp28-source-format" "$PYTHON_BIN tools/exp28_source_and_format_interaction.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "exp28-balanced" "$PYTHON_BIN tools/exp28_balanced_summary.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "cross-vs-same" "$PYTHON_BIN tools/cross_vs_same_interaction.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "occupancy-analysis" "$PYTHON_BIN tools/occupancy_setting_analysis.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR' --base-kind all"
run_cmd "boundary-crossing" "$PYTHON_BIN tools/boundary_crossing_summary.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "plausibility-proxy" "$PYTHON_BIN tools/plausibility_proxy_stratification.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "exp30-summary" "$PYTHON_BIN tools/summarize_exp30_injection_transfer.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "exp30-mechanism" "$PYTHON_BIN tools/exp30_backfire_mechanism_diagnostics.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "margin-calibration" "$PYTHON_BIN tools/margin_behavior_calibration.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "prompt-matched-contrast" "$PYTHON_BIN tools/summarize_prompt_matched_contrast.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "prompt-checklist" "$PYTHON_BIN tools/prompt_checklist_status.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "axis-representativeness" "$PYTHON_BIN tools/axis_representativeness_sensitivity.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "rank-sweep-basis" "$PYTHON_BIN tools/rank_sweep_basis_diagnostics.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "table2-provenance" "$PYTHON_BIN tools/table2_provenance_map.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "ci-vs-test" "$PYTHON_BIN tools/ci_vs_test_companion.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "crossfit-summary" "$PYTHON_BIN tools/crossfit_split_clean_summary.py --run-map '$RUN_MAP_CORE' --output-dir '$REPORT_DIR'"
run_cmd "render-rev3-auto-tables" "$PYTHON_BIN tools/render_rev3_auto_tables.py --run-map '$RUN_MAP_CORE' --report-dir '$REPORT_DIR' --output-dir 'paper/build/auto_tables'"

# Composition sensitivity on full model roster (core + extensions)
FULL_REPORT_DIR="$REPORT_DIR/full"
mkdir -p "$FULL_REPORT_DIR"
run_cmd "composition-sensitivity-full" "$PYTHON_BIN tools/composition_sensitivity_summary.py --run-map '$RUN_MAP_FULL' --output-dir '$FULL_REPORT_DIR'"
run_cmd "mistral-power" "$PYTHON_BIN tools/mistral_power_report.py --run-map '$RUN_MAP_FULL' --output-dir '$REPORT_DIR'"

# Figure/table regeneration and reproducibility bundle
run_cmd "make-paper-figures" "$PYTHON_BIN tools/make_paper_figures.py --run-map '$RUN_MAP_CORE'"
run_cmd "compile-results" "$PYTHON_BIN tools/compile_results.py"

MANIFEST_JSON="results/${RUN_TAG}/artifact_manifest.json"
run_cmd "write-artifact-manifest" "$PYTHON_BIN tools/write_artifact_manifest.py --run-tag '$RUN_TAG' --run-map '$RUN_MAP_FULL' --output '$MANIFEST_JSON'"
run_cmd "cleanroom-repro" "bash tools/cleanroom_repro_check.sh '$MANIFEST_JSON' 'results/${RUN_TAG}/cleanroom_check'"

run_cmd "latex-build" "cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex"

set_flag "aggregate_done"
log_msg "Aggregator complete"
