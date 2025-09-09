What is needed in pwd

If using Groq's API (`IS_GROQ` is set to true) you need keys.txt with one API key per line



file counter: 

`python3 count_completed_files.py --dir /local/scratch/sbuongi/date_tagger --pattern "tag_dates_full_*.out"`




I have to split the files 

Log the number of sentence_cores sent to the LLM 




Bucket the data and create number of shards: 
```
python3 split_and_bucket_updated.py \
  --data-root /local/scratch/sbuongi/date_tagger/copied_data \
  --out-base  /local/scratch/sbuongi/date_tagger/run_split3 \
  --file-glob '*.csv' \
  --bucket-size 300 \
  --num-shards 3
```

Use all ports 

```
BASE=/local/scratch/sbuongi/date_tagger/run_split3

# Shard 1 -> port 58112
sbatch --job-name=tag_s1 \
  --export=ALL,\
API_BASE_URL="http://127.0.0.1:58112/v1/chat/completions",\
DATA_ROOT="$BASE/shard1/inputs",\
OUT_BASE="$BASE/shard1/tagging",\
OUT_ROOT="$BASE/shard1/outputs",\
BUCKET_LIST="$BASE/shard1/bucket.list" \
  date_tagger.sbatch

# Shard 2 -> port 58114
sbatch --job-name=tag_s2 \
  --export=ALL,\
API_BASE_URL="http://127.0.0.1:58114/v1/chat/completions",\
DATA_ROOT="$BASE/shard2/inputs",\
OUT_BASE="$BASE/shard2/tagging",\
OUT_ROOT="$BASE/shard2/outputs",\
BUCKET_LIST="$BASE/shard2/bucket.list" \
  date_tagger.sbatch

# Shard 3 -> port 58116
sbatch --job-name=tag_s3 \
  --export=ALL,\
API_BASE_URL="http://127.0.0.1:58116/v1/chat/completions",\
DATA_ROOT="$BASE/shard3/inputs",\
OUT_BASE="$BASE/shard3/tagging",\
OUT_ROOT="$BASE/shard3/outputs",\
BUCKET_LIST="$BASE/shard3/bucket.list" \
  date_tagger.sbatch
```
