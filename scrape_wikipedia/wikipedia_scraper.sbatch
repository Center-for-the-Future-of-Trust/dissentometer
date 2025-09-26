#!/bin/bash -l
#SBATCH --job-name=wikipedia_scraper
#SBATCH --output=wikipedia_scraper_%j.out
#SBATCH --error=wikipedia_scraper_%j.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --partition=debug

# Use the system python3 (adjust path if different)
PY=/usr/bin/python3   # or: PY=$(which python3)

srun "$PY" wikipedia_scraper.py \
    --output /wikipedia_articles \
    --concurrency 16 \
    --per-host 5 \
    --batch-size 60