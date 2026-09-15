#!/bin/bash
# Runs the full mode x start-method matrix sequentially with progress/ETA lines.
cd "$(dirname "$0")"
PY=${PY:-python}
MODES="list numpy sharedlist usm fastsharedlist sharedarray"
CTXS="fork forkserver spawn"
total=$(( $(echo $MODES | wc -w) * $(echo $CTXS | wc -w) ))
k=0; t0=$(date +%s)
echo "total units: $total (modes x start methods)"
for ctx in $CTXS; do for mode in $MODES; do
  k=$((k+1)); ts=$(date +%s)
  echo "[$k/$total] mode=$mode ctx=$ctx"
  $PY -u cow_bench.py --mode $mode --ctx $ctx 2>&1 | grep -v -i warning
  now=$(date +%s); el=$((now-t0)); avg=$((el/k)); eta=$(( avg*(total-k) ))
  echo "  elapsed ${el}s  avg/unit ${avg}s  ETA $((eta/60))m$((eta%60))s"
  [ $k -eq 1 ] && echo "  -> projected total $((avg*total/60))m"
done; done
echo "=== shareddict lookup ==="
$PY -u shareddict_lookup.py 2>&1 | grep -v -i warning
echo "MATRIX_DONE total $(( $(date +%s)-t0 ))s"
