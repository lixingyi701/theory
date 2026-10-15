#!/usr/bin/env bash
set -euo pipefail
model_key="${1:-qwen3_8b}"
mode="${2:---plan}"
if [[ "$mode" != "--plan" && "$mode" != "--execute" ]]; then
  echo 'Usage: bash scripts/serve_model.sh [qwen3_8b|qwen3_14b] [--plan|--execute]' >&2
  exit 2
fi
mapfile -t model_fields < <(python - "$model_key" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path('configs/models.json').read_text())[sys.argv[1]]
for key in ['local_dir', 'served_name', 'revision']:
    print(m[key])
PY
)
model_path="${model_fields[0]}"
served_name="${model_fields[1]}"
revision="${model_fields[2]}"
cmd=(vllm serve "$model_path" --served-model-name "$served_name"
     --host 127.0.0.1 --port 8000 --dtype bfloat16
     --max-model-len 8192 --gpu-memory-utilization 0.85
     --tensor-parallel-size "${RELIABILITY_TP:-1}" --max-num-seqs 4
     --generation-config vllm --enable-prefix-caching)
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$mode" == "--plan" ]]; then exit 0; fi
python - "$model_path" "$revision" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
r = json.loads((root / 'download_receipt.json').read_text())
if r['revision'] != sys.argv[2]:
    raise SystemExit('Download revision mismatch')
for item in r['weight_files']:
    if (root / item['file']).stat().st_size != item['bytes']:
        raise SystemExit('Missing/changed weight shard')
PY
exec "${cmd[@]}"
