#!/usr/bin/env bash
# date_tagger_resilient_rate_limited.sbatch — generic rate limiting (always on) + safer key parsing + bucketized array
# Phase A (controller): bucketize & submit array
# Phase B (worker): activate venv and run date_tagger.py over one bucket
#
# WHAT'S NEW IN THIS VERSION
# - Rate limiting is ALWAYS enabled, regardless of provider toggles.
# - Generic env var names (no 'GROQ' in the limiter names).
# - Backwards-compat shim: if legacy GROQ_* vars are present, they are mapped into the new generic names.
# - Safer API key parsing/expansion; trims whitespace/CRLF and ignores blank/comment lines.
# - Optional % concurrency cap commented out (leave it if you want no cap).
#
# USAGE EXAMPLE (controller; Phase A):
#   sbatch --export=ALL,PROJECT_ROOT=/local/scratch/sbuongi/date_tagger,DATA_ROOT=/local/scratch/sbuongi/copied_data,BUCKET_SIZE=300 \
#          date_tagger_resilient_rate_limited.sbatch
#
# USAGE EXAMPLE (worker; Phase B direct debug):
#   export SLURM_ARRAY_TASK_ID=0; bash date_tagger_resilient_rate_limited.sbatch
#
# Assumes a Python script `date_tagger.py` that reads limits from the generic env vars defined below.

#SBATCH --job-name=tag_dates_full
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
# If you want to throttle burstiness to roughly the number of keys, uncomment and set e.g. %10
# #SBATCH --array=0-0%10

set -euo pipefail

# ---- Helper: lowercase truthy check ----
istrue() {
  case "${1,,}" in true|1|yes|y) return 0 ;; *) return 1 ;; esac
}

# ---- Modules (safe to source if present) ----
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true

# ---- Config (override via --export=ALL,VAR=val) ----
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_ROOT:=${PROJECT_ROOT}/data}"       # root folder holding CSVs (possibly many subdirs)
: "${OUT_BASE:=${PROJECT_ROOT}/tagging}"     # base dir for outputs (per-bucket)
: "${OUT_ROOT:=${PROJECT_ROOT}/outputs}"     # optional global outputs dir
: "${VENV_ACTIVATE:=${PROJECT_ROOT}/venvs/groq_tagging/bin/activate}"  # existing venv path (name doesn't matter)
: "${PY_SCRIPT:=${PROJECT_ROOT}/date_tagger.py}"
: "${KEYS_FILE:=${PROJECT_ROOT}/keys.txt}"   # one API key per line
: "${KEYS_EXPANDED:=${PROJECT_ROOT}/keys.expanded.txt}"
: "${TOKEN_STATE_DIR:=/local/scratch/${USER}/token_state}"  # shared, node-visible absolute path
: "${API_BASE_URL:=https://api.openai.com/v1/chat/completions}"  # or your local proxy/http://localhost:58112/v1/chat/completions
: "${MODEL_NAME:=llama-3.1-8b-instant}"
: "${CONTENT_COL:=English translation}"
: "${TAG_COL:=date_tagged}"

# Bucketing controls
: "${BUCKET_SIZE:=300}"       # number of CSVs per bucket (≈ one array task)
: "${BUCKET_LIST:=${PROJECT_ROOT}/bucket.list}"

# ---- ALWAYS-ON GENERIC RATE LIMIT VARS ----
# Preferred new names (what your Python should read):
: "${RATE_LIMIT_RPM:=30}"        # requests per minute per key
: "${RATE_LIMIT_TPM:=6000}"      # tokens per minute per key
: "${RATE_LIMIT_RPD:=14400}"     # requests per day per key
: "${RATE_LIMIT_TPD:=500000}"    # tokens per day per key

# Back-compat shim: If legacy names are set, map them into the generic ones (without mentioning 'groq').
if [[ -n "${GROQ_RPM:-}" ]];  then export RATE_LIMIT_RPM="${GROQ_RPM}";   fi
if [[ -n "${GROQ_TPM:-}" ]];  then export RATE_LIMIT_TPM="${GROQ_TPM}";   fi
if [[ -n "${GROQ_RPD:-}" ]];  then export RATE_LIMIT_RPD="${GROQ_RPD}";   fi
if [[ -n "${GROQ_TPD:-}" ]];  then export RATE_LIMIT_TPD="${GROQ_TPD}";   fi

# Optional global safety multipliers (e.g., run at 90% of the documented cap)
: "${RATE_SLACK:=0.90}"  # 0.0–1.0
python - <<'PY'
import os, math
def clamp(x, lo, hi): 
    return max(lo, min(hi, x))
slack = float(os.getenv("RATE_SLACK", "0.90"))
slack = clamp(slack, 0.05, 1.0)
def adj(name, default):
    val = float(os.getenv(name, str(default)))
    val = math.floor(val * slack) if "TPM" not in name else math.floor(val * slack)
    os.environ[name] = str(max(1, int(val)))
for n,d in [("RATE_LIMIT_RPM",30),("RATE_LIMIT_TPM",6000),("RATE_LIMIT_RPD",14400),("RATE_LIMIT_TPD",500000)]:
    adj(n,d)
# write back to shell
print(f'export RATE_LIMIT_RPM={os.environ["RATE_LIMIT_RPM"]}')
print(f'export RATE_LIMIT_TPM={os.environ["RATE_LIMIT_TPM"]}')
print(f'export RATE_LIMIT_RPD={os.environ["RATE_LIMIT_RPD"]}')
print(f'export RATE_LIMIT_TPD={os.environ["RATE_LIMIT_TPD"]}')
PY

# ---- Make shared state dir ----
mkdir -p "$TOKEN_STATE_DIR"
echo "[INFO] TOKEN_STATE_DIR=$TOKEN_STATE_DIR"

# ---- Phase detection: if no SLURM_ARRAY_TASK_ID, we're in controller (Phase A) ----
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[A] Controller phase: building buckets and submitting the array"

  # 1) Build list of CSVs
  mapfile -t all_csvs < <(find "$DATA_ROOT" -type f -name '*.csv' | sort)
  if (( ${#all_csvs[@]} == 0 )); then
    echo "[A] No CSV files found under $DATA_ROOT" >&2
    exit 2
  fi

  # 2) Bucketize
  rm -f "$BUCKET_LIST"
  total="${#all_csvs[@]}"
  idx=0
  bucket_idx=0
  while (( idx < total )); do
    start=$idx
    end=$(( idx + BUCKET_SIZE - 1 ))
    (( end >= total )) && end=$(( total - 1 ))
    bucket_dir="${OUT_BASE}/bucket_${bucket_idx}"
    mkdir -p "$bucket_dir"
    # Write a file list for this bucket
    bucket_list="${bucket_dir}/files.txt"
    : > "$bucket_list"
    for i in $(seq "$start" "$end"); do
      printf "%s\n" "${all_csvs[$i]}" >> "$bucket_list"
    done
    echo "$bucket_dir" >> "$BUCKET_LIST"
    idx=$(( end + 1 ))
    bucket_idx=$(( bucket_idx + 1 ))
  done
  num_buckets=$bucket_idx
  echo "[A] Bucketized ${total} files into ${num_buckets} buckets @ ${BUCKET_SIZE} each"

  # 3) Clean/expand keys file (ignore empty lines or lines starting with '#')
  if [[ -f "$KEYS_FILE" ]]; then
    awk 'NF && $0 !~ /^[[:space:]]*#/ {gsub(/\r/,""); print $0}' "$KEYS_FILE" > "$KEYS_EXPANDED"
  else
    echo "[A] WARNING: KEYS_FILE not found at $KEYS_FILE — continuing without keys.txt"
    : > "$KEYS_EXPANDED"
  fi
  keys_count=$(wc -l < "$KEYS_EXPANDED" || echo 0)
  echo "[A] keys_count=$keys_count (ok if 0 when using a local model without API keys)"

  # 4) Submit the array over all buckets (no % cap by default)
  # NOTE: add %N to throttle if you want, e.g., --array=0-$((num_buckets-1))%${keys_count:-1}
  sbatch --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",DATA_ROOT="$DATA_ROOT",OUT_BASE="$OUT_BASE",OUT_ROOT="$OUT_ROOT",\
VENV_ACTIVATE="$VENV_ACTIVATE",PY_SCRIPT="$PY_SCRIPT",KEYS_FILE="$KEYS_FILE",KEYS_EXPANDED="$KEYS_EXPANDED",\
TOKEN_STATE_DIR="$TOKEN_STATE_DIR",API_BASE_URL="$API_BASE_URL",MODEL_NAME="$MODEL_NAME",CONTENT_COL="$CONTENT_COL",TAG_COL="$TAG_COL",\
RATE_LIMIT_RPM="$RATE_LIMIT_RPM",RATE_LIMIT_TPM="$RATE_LIMIT_TPM",RATE_LIMIT_RPD="$RATE_LIMIT_RPD",RATE_LIMIT_TPD="$RATE_LIMIT_TPD" \
         --array=0-$((num_buckets-1)) "$0"
  echo "[A] Submitted array 0-$((num_buckets-1))"
  exit 0
fi

# ---- Phase B (worker) ----
echo "[B] Worker phase: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# 1) Activate venv
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "[B] WARNING: venv not found at $VENV_ACTIVATE; relying on system Python"
fi

# 2) Select bucket dir for this task
if [[ ! -f "$BUCKET_LIST" ]]; then
  echo "[B] ERROR: BUCKET_LIST not found at $BUCKET_LIST (controller should have created it)" >&2
  exit 3
fi
bucket_dir="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$BUCKET_LIST" | tr -d '\r\n')"
if [[ -z "$bucket_dir" || ! -d "$bucket_dir" ]]; then
  echo "[B] ERROR: bucket_dir for task ${SLURM_ARRAY_TASK_ID} not found" >&2
  exit 4
fi
echo "[B] bucket_dir=$bucket_dir"

# 3) Choose API key deterministically by task index (ok if keys_count==0 for local model)
key=""
if [[ -f "$KEYS_EXPANDED" ]]; then
  keys_count=$(wc -l < "$KEYS_EXPANDED" || echo 0)
  if (( keys_count > 0 )); then
    kidx=$(( SLURM_ARRAY_TASK_ID % keys_count ))
    key="$(sed -n "$((kidx+1))p" "$KEYS_EXPANDED" | tr -d '\r\n')"
  fi
fi
if [[ -n "$key" ]]; then
  export API_KEY="$key"          # generic name (no provider prefix)
else
  echo "[B] INFO: No API_KEY set (this is fine if using a local model/API that doesn't require a key)"
fi

# 4) Per-task output dir and state
DATA_DIR="$bucket_dir"
mkdir -p "$OUT_ROOT" "$TOKEN_STATE_DIR"

# 5) Diagnostics
echo "[B] MODEL_NAME=${MODEL_NAME}"
echo "[B] API_BASE_URL=${API_BASE_URL}"
echo "[B] RATE_LIMIT_RPM=${RATE_LIMIT_RPM}  RATE_LIMIT_TPM=${RATE_LIMIT_TPM}  RATE_LIMIT_RPD=${RATE_LIMIT_RPD}  RATE_LIMIT_TPD=${RATE_LIMIT_TPD}"
echo "[B] TOKEN_STATE_DIR=${TOKEN_STATE_DIR}"
echo "[B] CONTENT_COL='${CONTENT_COL}'  TAG_COL='${TAG_COL}'"

# 6) Run the Python tagger (uses srun for proper resource binding/logging)
# NOTE: Your Python should read:
#   - API_BASE_URL, API_KEY, MODEL_NAME
#   - RATE_LIMIT_RPM, RATE_LIMIT_TPM, RATE_LIMIT_RPD, RATE_LIMIT_TPD
#   - TOKEN_STATE_DIR (for cross-worker shared rate-budget files)
exec srun -u python "$PY_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --content-col "$CONTENT_COL" \
  --tag-col "$TAG_COL" \
  --model "$MODEL_NAME" \
  --out-root "$OUT_ROOT"
