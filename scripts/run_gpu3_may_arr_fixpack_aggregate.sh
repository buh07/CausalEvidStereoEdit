#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

cd /jumbo/lisp/f004ndc/StereACL

RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
CONFIG_PATH="${CONFIG_PATH:-configs/may_arr_freeze_v1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STATE_DIR="results/${RUN_TAG}/state"
REPORT_DIR="results/${RUN_TAG}/reports"
LOG_DIR="results/${RUN_TAG}/logs"
mkdir -p "$STATE_DIR" "$REPORT_DIR" "$LOG_DIR"
ORCH_START_UTC="$(cat "$STATE_DIR/orch_start_utc.txt")"
LOG_FILE="$LOG_DIR/aggregate_gpu3.log"

source ./scripts/may_arr_fixpack_common.sh

log_msg "Aggregator start | run_tag=$RUN_TAG"

for key in gemma2b gemma2bit llama3b; do
  wait_for_flag "${key}_done"
  if [[ -f "$(flag_file "${key}_failed")" ]]; then
    log_msg "Detected failure flag for $key"
    exit 1
  fi
done

RUN_MAP_JSON="$STATE_DIR/run_map.json"
$PYTHON_BIN - "$STATE_DIR" "$CONFIG_PATH" "$RUN_TAG" "$RUN_MAP_JSON" <<'PY'
import json
import sys
from pathlib import Path
import yaml

state_dir = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
run_tag = sys.argv[3]
out_path = Path(sys.argv[4])

cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
models_cfg = cfg.get('core_models', [])

mapping = {
    'run_tag': run_tag,
    'config_path': str(cfg_path),
    'models': {},
}

for m in models_cfg:
    key = m['key']
    model_name = m['model']
    label = m['label']

    def read_txt(name: str) -> str:
        fp = state_dir / f"{key}_{name}.txt"
        return fp.read_text(encoding='utf-8').strip() if fp.exists() else ''

    seed_runs = {}
    for seed in (11, 29, 47):
        v = read_txt(f"exp16_seed_{seed}_run_dir")
        if v:
            seed_runs[str(seed)] = v

    mapping['models'][model_name] = {
        'key': key,
        'label': label,
        'gpu': m.get('gpu'),
        'exp01_mixed_run_dir': read_txt('exp01_mixed_run_dir'),
        'exp01_stereoset_run_dir': read_txt('exp01_stereoset_run_dir'),
        'exp01_crows_run_dir': read_txt('exp01_crows_run_dir'),
        'exp02_strict_run_dir': read_txt('exp02_strict_run_dir'),
        'exp02_stereoset_run_dir': read_txt('exp02_stereoset_run_dir'),
        'exp02_crows_run_dir': read_txt('exp02_crows_run_dir'),
        'exp03_strict_run_dir': read_txt('exp03_strict_run_dir'),
        'exp09_strict_run_dir': read_txt('exp09_strict_run_dir'),
        'exp14_run_dir': read_txt('exp14_run_dir'),
        'exp17_run_dir': read_txt('exp17_run_dir'),
        'exp15_run_dir': read_txt('exp15_run_dir'),
        'exp21_run_dir': read_txt('exp21_run_dir'),
        'exp26_run_dir': read_txt('exp26_run_dir'),
        'exp27_run_dir': read_txt('exp27_run_dir'),
        'exp28_run_dir': read_txt('exp28_run_dir'),
        'exp29_run_dir': read_txt('exp29_run_dir'),
        'exp30_run_dir': read_txt('exp30_run_dir'),
        'exp16_canonical_run_dir': read_txt('exp16_canonical_run_dir'),
        'exp16_source_stereoset_run_dir': read_txt('exp16_source_stereoset_run_dir'),
        'exp16_source_crows_run_dir': read_txt('exp16_source_crows_run_dir'),
        'exp16_seed_runs': seed_runs,
    }

out_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding='utf-8')
print(out_path)
PY
log_msg "Wrote run map: $RUN_MAP_JSON"

run_cmd "aggregate-seed-core" "$PYTHON_BIN tools/aggregate_seed_core.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR' --require-complete"
run_cmd "inference-robustness" "$PYTHON_BIN tools/build_inference_robustness_table.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "coverage-bias" "$PYTHON_BIN tools/coverage_bias_summary.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "multitoken-matched-summary" "$PYTHON_BIN tools/summarize_multitoken_matched_contrast.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "exp28-source-and-format" "$PYTHON_BIN tools/exp28_source_and_format_interaction.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "composition-sensitivity" "$PYTHON_BIN tools/composition_sensitivity_summary.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "cross-vs-same-interaction" "$PYTHON_BIN tools/cross_vs_same_interaction.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "boundary-crossing" "$PYTHON_BIN tools/boundary_crossing_summary.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"
run_cmd "table2-provenance" "$PYTHON_BIN tools/table2_provenance_map.py --run-map '$RUN_MAP_JSON' --output-dir '$REPORT_DIR'"

run_cmd "make-paper-figures" "$PYTHON_BIN tools/make_paper_figures.py --run-map '$RUN_MAP_JSON'"
run_cmd "compile-results" "$PYTHON_BIN tools/compile_results.py"

MANIFEST_JSON="results/${RUN_TAG}/artifact_manifest.json"
run_cmd "write-artifact-manifest" "$PYTHON_BIN tools/write_artifact_manifest.py --run-tag '$RUN_TAG' --run-map '$RUN_MAP_JSON' --output '$MANIFEST_JSON'"
run_cmd "cleanroom-repro" "bash tools/cleanroom_repro_check.sh '$MANIFEST_JSON' 'results/${RUN_TAG}/cleanroom_check'"

# Final summary bundle.
$PYTHON_BIN - "$RUN_TAG" "$REPORT_DIR" "$STATE_DIR" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

run_tag = sys.argv[1]
report_dir = Path(sys.argv[2])
state_dir = Path(sys.argv[3])

summary_md = report_dir / 'may_arr_fixpack_summary.md'
summary_csv = report_dir / 'may_arr_fixpack_summary.csv'

run_map = json.loads((state_dir / 'run_map.json').read_text(encoding='utf-8'))
seed_agg = report_dir / 'seed_core_asymmetry_aggregate.csv'
inf = report_dir / 'inference_robustness_core.csv'
ret = report_dir / 'alignment_retention_summary.csv'
cov = report_dir / 'coverage_bias_summary.csv'
exp28 = report_dir / 'multitoken_matched_contrast_summary.csv'
exp28_source = report_dir / 'exp28_source_stratified_summary.csv'
exp28_fmt = report_dir / 'exp28_single_vs_span_interaction.csv'
comp = report_dir / 'composition_sensitivity_exp16_pool_summary.csv'
comp_int = report_dir / 'composition_sensitivity_exp16_interaction.csv'
inter = report_dir / 'cross_vs_same_interaction.csv'
bound = report_dir / 'boundary_crossing_summary.csv'
bound_q = report_dir / 'boundary_crossing_quality.csv'
prov = report_dir / 'table2_provenance_map.csv'

lines = [
    '# May ARR Fixpack Summary',
    '',
    f'- RUN_TAG: `{run_tag}`',
    f'- Run map: `{state_dir / "run_map.json"}`',
    f'- Seed aggregate: `{seed_agg}`',
    f'- Inference robustness: `{inf}`',
    f'- Coverage/retention semantics: `{cov}`',
    f'- Multi-token matched summary: `{exp28}`',
    f'- Exp28 source stratification: `{exp28_source}`',
    f'- Exp28 single-vs-span interaction: `{exp28_fmt}`',
    f'- Composition sensitivity (direction pools): `{comp}`',
    f'- Composition sensitivity interactions: `{comp_int}`',
    f'- Cross-vs-same interaction: `{inter}`',
    f'- Boundary crossing summary: `{bound}`',
    f'- Boundary crossing quality: `{bound_q}`',
    f'- Table-2 provenance map: `{prov}`',
    '',
    '## Models',
]
for model, payload in run_map.get('models', {}).items():
    lines.append(f"- {payload.get('label', model)}: `{model}`")

summary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

rows = []
for model, payload in run_map.get('models', {}).items():
    rows.append({
        'model': model,
        'label': payload.get('label', model),
        'exp16_canonical_run_dir': payload.get('exp16_canonical_run_dir', ''),
        'exp26_run_dir': payload.get('exp26_run_dir', ''),
        'exp27_run_dir': payload.get('exp27_run_dir', ''),
        'exp28_run_dir': payload.get('exp28_run_dir', ''),
        'exp29_run_dir': payload.get('exp29_run_dir', ''),
        'exp30_run_dir': payload.get('exp30_run_dir', ''),
    })
pd.DataFrame(rows).to_csv(summary_csv, index=False)
print(summary_md)
print(summary_csv)
PY

set_flag "aggregate_done"
log_msg "Aggregator complete"
