# !/bin/bash

: "${SRC_DIRS:=/local/scratch/sbuongi/copied_data/*}"

N=300
i=0
shopt -s nullglob
for src in ${SRC_DIRS}; do
  echo "Processing Dir = $src"
  for f in "${src}"/*.csv; do
    b=$(( i % N ))
#    cp -n "$f" "${BUCKET_ROOT}/bucket_${b}/" 2>/dev/null || true
    i=$(( i + 1 ))
  done
done
shopt -u nullglob
