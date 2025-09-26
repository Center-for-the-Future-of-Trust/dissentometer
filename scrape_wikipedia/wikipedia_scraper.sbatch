#!/bin/bash
#SBATCH --job-name=wikipedia_scraper
#SBATCH --output=wikipedia_scraper_%j.out
#SBATCH --error=wikipedia_scraper_%j.err
#SBATCH --time=168:00:00            # walltime (7 days)
#SBATCH --ntasks=1                  # one task (single process)
#SBATCH --cpus-per-task=4           # 4 CPU cores reserved
#SBATCH --mem=16G                   # memory (adjust as needed)
#SBATCH --partition=standard        # partition/queue name (adjust for your cluster)

# Load modules if needed
module load python/3.10

# Activate your environment if you use one
# source ~/envs/wiki/bin/activate

# Run the scraper
srun python wikipedia_scraper.py \
    --output /wikipedia_articles \
    --concurrency 16 \
    --per-host 5 \
    --batch-size 60
