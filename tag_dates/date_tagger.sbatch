#!/usr/bin/env bash
#
# date_tagger_array_resume.sbatch
# Phase A (controller): bucketize & submit array
# Phase B (worker): activate venv and run date_tagger.py over one bucket
#
#SBATCH --job-name=tag_dates_full
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
# Example throttle:  #SBATCH --array=0-0%10

set -euo pipefail

# ---- Helper ----
istrue() { case "${1,,}" in true|1|yes|y) return 0 ;; *) return 1 ;; esac; }
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true

: "${IS_GROQ:=""}"

# ---- Config (override any of these at submit time with --export=ALL,VAR=value) ----
: "${PROJECT_ROOT:=/local/scratch/sbuongi/date_tagger}"
: "${DATA_ROOT:=/local/scratch/sbuongi/copied_data}"   # holds CSVs (recursively)
: "${OUT_BASE:=${PROJECT_ROOT}/tagging}"               # per-bucket bookkeeping
: "${OUT_ROOT:=${PROJECT_ROOT}/outputs}"               # python outputs root
: "${VENV_ACTIVATE:=${PROJECT_ROOT}/venvs/groq_tagging/bin/activate}"
: "${PY_SCRIPT:=${PROJECT_ROOT}/date_tagger.py}"
: "${KEYS_FILE:=${PROJECT_ROOT}/keys.txt}"
: "${KEYS_EXPANDED:=${PROJECT_ROOT}/keys.expanded.txt}"
: "${TOKEN_STATE_DIR:=/local/scratch/${USER}/token_state}"

# API target + model
#: "${API_BASE_URL:=https://api.openai.com/v1/chat/completions}"
: "${API_BASE_URL:=http://localhost:58112/v1/chat/completions}"
#: "${API_BASE_URL:=https://api.groq.com/openai/v1/chat/completions}"
#: "${MODEL_NAME:=llama-3.1-8b-instant}"   # example for Groq
#: "${MODEL_NAME:=gpt-oss:20b}"             # example local model (e.g., llama3.1:8b)
: "${MODEL_NAME:=llama3.1:8b}"

# CSV fields + file matching
: "${CONTENT_COL:=English translation}"
: "${TAG_COL:=date_tagged}"
: "${FILE_GLOB:=*.csv}"
: "${ENCODING:=utf-8}"
: "${CHAR_PER_TOKEN:=4}"

# Concurrency
: "${MAX_PARALLEL_ARRAY_TASKS:=9}"   # SLURM array concurrency cap
: "${MAX_CONCURRENCY:=5}"            # per-task async concurrency for Python
export MAX_CONCURRENCY

# Guard toggles (NEW)
: "${RESUME:=true}"           # true → add --resume; false to disable
: "${SKIP_EXISTING:=false}"   # true → add --skip-existing
: "${OVERWRITE:=false}"       # true → add --overwrite (forces full rewrite)

# Bucketing
: "${BUCKET_SIZE:=300}"
: "${BUCKET_LIST:=${PROJECT_ROOT}/bucket.list}"

# Generic rate limits (names not Groq-specific)
: "${RATE_LIMIT_RPM:=30}"
: "${RATE_LIMIT_TPM:=6000}"
: "${RATE_LIMIT_RPD:=14400}"
: "${RATE_LIMIT_TPD:=500000}"

# Back-compat shims (if you still export GROQ_* they map here)
[[ -n "${GROQ_RPM:-}" ]] && export RATE_LIMIT_RPM="${GROQ_RPM}"
[[ -n "${GROQ_TPM:-}" ]] && export RATE_LIMIT_TPM="${GROQ_TPM}"
[[ -n "${GROQ_RPD:-}" ]] && export RATE_LIMIT_RPD="${GROQ_RPD}"
[[ -n "${GROQ_TPD:-}" ]] && export RATE_LIMIT_TPD="${GROQ_TPD}"

# Apply RATE_SLACK (use eval to actually set env from Python printouts)
: "${RATE_SLACK:=0.90}"
eval "$(
python3 - <<'PY'
import os, math
def clamp(x, lo, hi): return max(lo, min(hi, x))
slack = clamp(float(os.getenv("RATE_SLACK","0.90")), 0.05, 1.0)
for k, d in [("RATE_LIMIT_RPM",30),("RATE_LIMIT_TPM",6000),("RATE_LIMIT_RPD",14400),("RATE_LIMIT_TPD",500000)]:
    v = int(math.floor(float(os.getenv(k, str(d))) * slack))
    print(f"export {k}={max(1, v)}")
PY
)"

# ---- Shared state dir ----
mkdir -p "$TOKEN_STATE_DIR"

# ---- Phase A (controller) ----
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[A] Controller: DATA_ROOT=$DATA_ROOT"

  # 1) Find CSVs (recursive, case-sensitive *.csv)
  #mapfile -t all_csvs < <(find "$DATA_ROOT" -type f -name '*.csv' | sort)
  mapfile -t all_csvs < <(find -L "$DATA_ROOT" -type f -name "${FILE_GLOB:-*.csv}" | sort)
  if (( ${#all_csvs[@]} == 0 )); then
    echo "[A] No CSV files found under $DATA_ROOT" >&2
    exit 2
  fi

  # 2) Bucketize
  rm -f "$BUCKET_LIST"
  total="${#all_csvs[@]}"; idx=0; bucket_idx=0
  while (( idx < total )); do
    start=$idx
    end=$(( idx + BUCKET_SIZE - 1 ))
    (( end >= total )) && end=$(( total - 1 ))
    bucket_dir="${OUT_BASE}/bucket_${bucket_idx}"
    mkdir -p "$bucket_dir"
    bucket_list="${bucket_dir}/files.txt"; : > "$bucket_list"
    for i in $(seq "$start" "$end"); do
      printf "%s\n" "${all_csvs[$i]}" >> "$bucket_list"
    done
    echo "$bucket_dir" >> "$BUCKET_LIST"
    idx=$(( end + 1 ))
    bucket_idx=$(( bucket_idx + 1 ))
  done
  num_buckets=$bucket_idx
  echo "[A] Bucketized ${total} files into ${num_buckets} buckets @ ${BUCKET_SIZE} each"

  # 3) Expand keys (strip comments/CRLF; keep blank if no file)
  if [[ -f "$KEYS_FILE" ]]; then
    awk 'NF && $0 !~ /^[[:space:]]*#/ {gsub(/\r/,""); print $0}' "$KEYS_FILE" > "$KEYS_EXPANDED"
  else
    echo "[A] WARNING: KEYS_FILE not found at $KEYS_FILE — continuing without keys.txt"
    : > "$KEYS_EXPANDED"
  fi
  keys_count=$(wc -l < "$KEYS_EXPANDED" || echo 0)
  echo "[A] keys_count=$keys_count"

  # 4) Submit array with concurrency cap — robust env handoff
  array_spec="0-$((num_buckets-1))%${MAX_PARALLEL_ARRAY_TASKS}"

  # Export vars to Phase B, then use --export=ALL
  export PROJECT_ROOT DATA_ROOT OUT_BASE OUT_ROOT VENV_ACTIVATE PY_SCRIPT KEYS_FILE KEYS_EXPANDED \
         TOKEN_STATE_DIR API_BASE_URL MODEL_NAME CONTENT_COL TAG_COL FILE_GLOB ENCODING CHAR_PER_TOKEN \
         RATE_LIMIT_RPM RATE_LIMIT_TPM RATE_LIMIT_RPD RATE_LIMIT_TPD BUCKET_LIST BUCKET_SIZE \
         MAX_PARALLEL_ARRAY_TASKS MAX_CONCURRENCY RESUME SKIP_EXISTING OVERWRITE

  sbatch --export=ALL --array="$array_spec" "$0"
  echo "[A] Submitted array $array_spec"
  exit 0
fi

# ---- Phase B (worker) ----
echo "[B] Worker: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# 1) Activate venv
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "[B] WARNING: venv not found at $VENV_ACTIVATE; relying on system Python3"
fi

# 2) Resolve bucket dir
if [[ ! -f "$BUCKET_LIST" ]]; then
  echo "[B] ERROR: BUCKET_LIST not found at $BUCKET_LIST" >&2
  exit 3
fi
bucket_dir="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$BUCKET_LIST" | tr -d '\r\n')"
if [[ -z "$bucket_dir" || ! -d "$bucket_dir" ]]; then
  echo "[B] ERROR: bucket_dir for task ${SLURM_ARRAY_TASK_ID} not found" >&2
  exit 4
fi
echo "[B] bucket_dir=$bucket_dir"

# 2.5) Ensure Python sees CSVs: build per-bucket _files/ view
work_dir="$bucket_dir/_files"
mkdir -p "$work_dir"
while IFS= read -r p; do
  [[ -n "$p" && -f "$p" ]] || continue
  ln -sf "$p" "$work_dir/$(basename "$p")" 2>/dev/null || cp -n "$p" "$work_dir/"
done < "$bucket_dir/files.txt"

DATA_DIR="$work_dir"
mkdir -p "$OUT_ROOT" "$TOKEN_STATE_DIR"

# 3) Key selection (optional; exported as both API_KEY and OPENAI_API_KEY)
key=""
if [[ -f "$KEYS_EXPANDED" ]]; then
  keys_count=$(wc -l < "$KEYS_EXPANDED" || echo 0)
  if (( keys_count > 0 )); then
    kidx=$(( SLURM_ARRAY_TASK_ID % keys_count ))
    key="$(sed -n "$((kidx+1))p" "$KEYS_EXPANDED" | tr -d '\r\n')"
  fi
fi
if [[ -n "$key" ]]; then
  export API_KEY="$key"
  export OPENAI_API_KEY="$key"
  export GROQ_API_KEY="$key"
else
  echo "[B] INFO: No API_KEY set"
fi

# 4) Diagnostics
echo "[B] MODEL_NAME=${MODEL_NAME}"
echo "[B] API_BASE_URL=${API_BASE_URL}"
echo "[B] RATE_LIMIT_RPM=${RATE_LIMIT_RPM}  RATE_LIMIT_TPM=${RATE_LIMIT_TPM}  RATE_LIMIT_RPD=${RATE_LIMIT_RPD}  RATE_LIMIT_TPD=${RATE_LIMIT_TPD}"
echo "[B] TOKEN_STATE_DIR=${TOKEN_STATE_DIR}"
echo "[B] CONTENT_COL='${CONTENT_COL}'  TAG_COL='${TAG_COL}'"
echo "[B] DATA_DIR=${DATA_DIR}  FILE_GLOB=${FILE_GLOB}  ENCODING=${ENCODING}"
echo "[B] MAX_CONCURRENCY=${MAX_CONCURRENCY}  RESUME=${RESUME}  SKIP_EXISTING=${SKIP_EXISTING}  OVERWRITE=${OVERWRITE}"

# 5) Build args for Python
args=(
  --data-dir "$DATA_DIR"
  --out-root "$OUT_ROOT"
  --file-glob "$FILE_GLOB"
  --content-col "$CONTENT_COL"
  --tag-col "$TAG_COL"
  --encoding "$ENCODING"
  --model "$MODEL_NAME"
  --char-per-token "$CHAR_PER_TOKEN"
  --max-concurrency "$MAX_CONCURRENCY"
)

if istrue "$OVERWRITE"; then
  args+=(--overwrite)
elif istrue "$RESUME"; then
  args+=(--resume)
elif istrue "$SKIP_EXISTING"; then
  args+=(--skip-existing)
fi

# 6) Run (ensure RESUME/SKIP/OVERWRITE flags reach date_tagger.py)
exec srun -u python3 "$PY_SCRIPT" "${args[@]}"
