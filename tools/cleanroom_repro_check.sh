#!/usr/bin/env bash
set -euo pipefail

cd /jumbo/lisp/f004ndc/StereACL

MANIFEST="${1:-}"
if [[ -z "$MANIFEST" ]]; then
  echo "Usage: tools/cleanroom_repro_check.sh <artifact_manifest.json> [report_dir]" >&2
  exit 2
fi

REPORT_DIR="${2:-}"
if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="$(dirname "$MANIFEST")/cleanroom_check"
fi
mkdir -p "$REPORT_DIR"

LOG="$REPORT_DIR/cleanroom_repro.log"
REPORT_JSON="$REPORT_DIR/cleanroom_repro_report.json"
REPORT_CSV="$REPORT_DIR/cleanroom_repro_report.csv"

{
  echo "[$(date -Iseconds)] START cleanroom repro check"
  echo "Manifest: $MANIFEST"
  python tools/make_paper_figures.py --artifact-manifest "$MANIFEST"
  python tools/compile_results.py

  python - "$MANIFEST" "$REPORT_JSON" "$REPORT_CSV" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_json = Path(sys.argv[2])
report_csv = Path(sys.argv[3])
root = Path('/jumbo/lisp/f004ndc/StereACL')

manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
rows = []
for item in manifest.get('checksums', []):
    rel = item.get('path', '')
    expected = item.get('sha256', '')
    fp = root / rel
    if not fp.exists() or not fp.is_file():
        rows.append({
            'path': rel,
            'status': 'missing',
            'expected_sha256': expected,
            'actual_sha256': '',
            'size_bytes': '',
        })
        continue
    h = hashlib.sha256()
    with fp.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    actual = h.hexdigest()
    rows.append({
        'path': rel,
        'status': 'match' if actual == expected else 'mismatch',
        'expected_sha256': expected,
        'actual_sha256': actual,
        'size_bytes': fp.stat().st_size,
    })

summary = {
    'manifest': str(manifest_path),
    'total_files': len(rows),
    'matches': sum(1 for r in rows if r['status'] == 'match'),
    'mismatches': sum(1 for r in rows if r['status'] == 'mismatch'),
    'missing': sum(1 for r in rows if r['status'] == 'missing'),
    'rows': rows,
}

report_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
with report_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['path', 'status', 'expected_sha256', 'actual_sha256', 'size_bytes'],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(json.dumps({
    'total_files': summary['total_files'],
    'matches': summary['matches'],
    'mismatches': summary['mismatches'],
    'missing': summary['missing'],
}, indent=2))
PY

  echo "[$(date -Iseconds)] DONE cleanroom repro check"
} | tee "$LOG"

