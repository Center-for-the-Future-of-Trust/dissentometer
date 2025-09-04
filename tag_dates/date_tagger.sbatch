#!/usr/bin/env bash
# date_tagger_resilient_venv.sbatch — handles 'ensurepip is not available' on HPC
# Phase A: bucketize & submit array
# Phase B: per-task runner that activates venv and calls date_tagger.py

#SBATCH --job-name=tag_dates_full
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

set -euo pipefail

# Optional modules (safe if your cluster doesn't use modules)
#module purge
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
#module load python/3.10

# ---- Config (override via --export=ALL,VAR=val when submitting) ----
: "${PROJECT_ROOT:=/local/scratch/sbuongi/date_tagger}"
: "${VENV_DIR:=${PROJECT_ROOT}/venvs/groq_tagging}"
: "${OUT_BASE:=/local/scratch/sbuongi/wiki_data_output/tagging}"
: "${OUT_ROOT:=/local/scratch/sbuongi/wiki_data_output/outputs}"
: "${SRC_DIRS:=/local/scratch/sbuongi/copied_data}"
: "${N:=300}"
: "${KEYS_FILE:=${PROJECT_ROOT}/keys.txt}"
: "${CONTENT_COL:=English translation}"
: "${TAG_COL:=date_tagged}"
: "${MODEL_NAME:=llama-3.1-8b-instant}"
: "${PY_SCRIPT:=date_tagger.py}"
: "${TOKEN_STATE_DIR:=${PROJECT_ROOT}/token_state}"

mkdir -p "${OUT_BASE}" "${OUT_ROOT}" "${TOKEN_STATE_DIR}"

# ---------- helper: find a python with venv+ensurepip, else try modules ----------
find_python() {
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" - <<'PY'
import sys
try:
    import venv, ensurepip  # both must be available
    ok = True
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
      then echo "$cand"; return 0; fi
    fi
  done
  if command -v module >/dev/null 2>&1; then
    module load python/3.10 >/dev/null 2>&1 || module load Python/3.10 >/dev/null 2>&1 || module load anaconda3 >/dev/null 2>&1 || true
    for cand in python3 python; do
      if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" - <<'PY'
import sys
try:
    import venv, ensurepip
    ok = True
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
        then echo "$cand"; return 0; fi
      fi
    done
  fi
  return 1
}

# ---------- create venv robustly ----------
create_venv() {
  local pybin="$1"
  echo "[INFO] Creating venv at ${VENV_DIR} using ${pybin}"
  mkdir -p "${VENV_DIR}"
  if "${pybin}" -m venv "${VENV_DIR}"; then
    return 0
  fi
  echo "[WARN] 'python -m venv' failed. Trying virtualenv fallback..."
  # Try pip-installing virtualenv to user site and create venv
  if "${pybin}" -m pip --version >/dev/null 2>&1; then
    "${pybin}" -m pip install --user --upgrade virtualenv >/dev/null 2>&1 || true
    # virtualenv could land in ~/.local/bin
    if command -v virtualenv >/dev/null 2>&1; then
      virtualenv "${VENV_DIR}" && return 0
    elif [[ -x "${HOME}/.local/bin/virtualenv" ]]; then
      "${HOME}/.local/bin/virtualenv" "${VENV_DIR}" && return 0
    fi
  fi
  echo "[ERROR] Could not create a virtual environment. Please load a Python module with venv/ensurepip support."
  return 1
}

# ===== Phase A =====
if [[ -z "${SLURM_ARRAY_TASK_ID+x}" ]]; then
  echo "[INFO] Phase A: bucketize and submit array"
  BUCKET_ROOT="${OUT_BASE}/buckets"
  mkdir -p "${BUCKET_ROOT}"
  : > "${OUT_BASE}/dirs.txt"

  # Pre-create N bucket dirs
  for i in $(seq 0 $((N-1))); do
    mkdir -p "${BUCKET_ROOT}/bucket_${i}"
  done

  # Round-robin copy CSVs from SRC_DIRS/*/*.csv into buckets
  i=0
  shopt -s nullglob
  for src in ${SRC_DIRS}/*; do
    for f in "${src}"/*.csv; do
      b=$(( i % N ))
      cp -n "$f" "${BUCKET_ROOT}/bucket_${b}/" 2>/dev/null || true
      i=$(( i + 1 ))
    done
  done
  shopt -u nullglob

  # List bucket paths for the array
  for i in $(seq 0 $((N-1))); do
    echo "${BUCKET_ROOT}/bucket_${i}" >> "${OUT_BASE}/dirs.txt"
  done

  # Expand keys so each array index has a key (round-robin over keys.txt)
  if [[ -f "${KEYS_FILE}" ]]; then mapfile -t KEYS < "${KEYS_FILE}"; else KEYS=(); fi
  : > "${OUT_BASE}/keys.expanded.txt"
  for i in $(seq 0 $((N-1))); do
    if (( ${#KEYS[@]} )); then
      idx=$(( i % ${#KEYS[@]} ))
      echo "${KEYS[$idx]}" >> "${OUT_BASE}/keys.expanded.txt"
    else
      echo "" >> "${OUT_BASE}/keys.expanded.txt"
    fi
  done

  TOTAL="$(wc -l < "${OUT_BASE}/dirs.txt" | tr -d ' ')"
  echo "[INFO] Prepared ${TOTAL} buckets under ${BUCKET_ROOT}"
  echo "[INFO] Submitting array 0-$((TOTAL-1))"

  sbatch \
    --array=0-$((TOTAL-1)) \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",VENV_DIR="${VENV_DIR}",OUT_BASE="${OUT_BASE}",OUT_ROOT="${OUT_ROOT}",SRC_DIRS="${SRC_DIRS}",N="${N}",KEYS_FILE="${KEYS_FILE}",CONTENT_COL="${CONTENT_COL}",TAG_COL="${TAG_COL}",MODEL_NAME="${MODEL_NAME}",PY_SCRIPT="${PY_SCRIPT}",TOKEN_STATE_DIR="${TOKEN_STATE_DIR}" \
    "$(realpath "$0")" -p "${PY_SCRIPT}"
  exit 0
fi

# ===== Phase B =====
echo "[INFO] Phase B: task ${SLURM_ARRAY_TASK_ID}"

# Find usable python, create venv if missing
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  PYBIN="$(find_python)" || { echo "[ERROR] No Python with venv+ensurepip found."; exit 127; }
  create_venv "${PYBIN}" || exit 127
fi

# Activate venv and install deps
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null 2>&1 || true

# Load Phase B inputs
mapfile -t DIRS < "${OUT_BASE}/dirs.txt"
mapfile -t KEYS < "${OUT_BASE}/keys.expanded.txt"
IDX="${SLURM_ARRAY_TASK_ID}"
TOTAL="${#DIRS[@]}"
(( IDX >= 0 && IDX < TOTAL )) || { echo "[ERROR] Index $IDX out of range (0..$((TOTAL-1)))"; exit 6; }

DATA_DIR="${DIRS[$IDX]}"
export GROQ_API_KEY="${KEYS[$IDX]:-}"

echo "[INFO] DATA_DIR: $DATA_DIR"
[[ -d "$DATA_DIR" ]] || { echo "[ERROR] Missing DATA_DIR: $DATA_DIR"; exit 7; }

mkdir -p "${TOKEN_STATE_DIR}"
export TOKEN_STATE_DIR

# Run the Python tagger (uses srun for proper resource binding/logging)
exec srun -u python "${PY_SCRIPT}" \
  --data-dir "$DATA_DIR" \
  --content-col "$CONTENT_COL" \
  --tag-col "$TAG_COL" \
  --model "$MODEL_NAME" \
  --out-root "$OUT_ROOT"
