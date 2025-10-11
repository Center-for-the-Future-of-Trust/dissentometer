#!/bin/bash -l
#SBATCH --job-name=wikiscrape_async_serial
#SBATCH --output=wikiscrape_%A_%a.out
#SBATCH --error=wikiscrape_%A_%a.err
#SBATCH --time=168:00:00          # 7 days
#SBATCH --ntasks=1                # one Python process per shard
#SBATCH --cpus-per-task=2         # CPU cores for the async event loop
#SBATCH --mem=30G
#SBATCH --array=0-7               # <-- runs all 8 shards (IDs 0..7)
#SBATCH --partition=debug         # <-- change to your real partition

set -euo pipefail

# Always run from the submit directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Python venv ---
VENV="$HOME/.venvs/wiki"
if [[ ! -f "$VENV/bin/activate" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Ensure deps (quietly)
python -m pip install -q --upgrade pip wheel
python - <<'PY'
import importlib, subprocess, sys
need=[]
for pkg in ("aiohttp","pycountry","unidecode"):
    try:
        importlib.import_module(pkg)
    except Exception:
        need.append(pkg)
if need:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *need])
print("[deps] ok")
PY

# --- Paths & script name ---
SCRIPT="wikipedia_scraper_async_seriallike.py"   # adjust if your filename differs
OUTDIR="$PWD/wikipedia_articles"
mkdir -p "$OUTDIR"

echo "[info] python = $(which python)"
python -V
echo "[info] outdir = $OUTDIR"
echo "[info] shard = ${SLURM_ARRAY_TASK_ID}"

# --- Run for this shard ---
srun "$VENV/bin/python" "$SCRIPT" \
  --output "$OUTDIR" \
  --concurrency 10 \
  --per-host 2 \
  --batch 50 \
  --max-depth 4 \
  --ascii-filenames \
  --shards 8 \
  --shard-id "${SLURM_ARRAY_TASK_ID}"
