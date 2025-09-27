#!/bin/bash -l
#SBATCH --job-name=wikiscrape_async_serial
#SBATCH --output=wikiscrape_%j.out
#SBATCH --error=wikiscrape_%j.err
#SBATCH --time=168:00:00          # 7 days
#SBATCH --ntasks=1                # one Python process
#SBATCH --cpus-per-task=2         # CPU cores for the async event loop
#SBATCH --mem=30G
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
SCRIPT="wikipedia_scraper.py"   # change if your filename differs
OUTDIR="$PWD/wikipedia_articles"
mkdir -p "$OUTDIR"

echo "[info] python = $(which python)"
python -V
echo "[info] outdir = $OUTDIR"

# --- Run ---
srun "$VENV/bin/python" "$SCRIPT"   --output "$OUTDIR"   --concurrency 10   --per-host 3   --batch 50   --max-depth 4   --ascii-filenames
