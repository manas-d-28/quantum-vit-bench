#!/bin/sh
# Run every (model, lr, seed) combination on one task and append to one results folder.
#
#   sh tools/run_grid.sh TASK "MODELS" "LRS" "SEEDS" EPOCHS OUT
#
# Learning-rate search (2 seeds per lr, choose by validation with tools/stats.py --select-lr):
#   sh tools/run_grid.sh pair_same "classic uniform gaussian overlap_small" "0.0003 0.001 0.003" "0 1" 60 /kaggle/working/lr_search
# Final runs (5 seeds at each model's chosen lr):
#   sh tools/run_grid.sh pair_same "classic" "0.001" "0 1 2 3 4" 60 /kaggle/working/final
set -e
TASK=$1; MODELS=$2; LRS=$3; SEEDS=$4; EPOCHS=$5; OUT=$6
for m in $MODELS; do
  for lr in $LRS; do
    for s in $SEEDS; do
      python -u -m qvb.train --model "$m" --task "$TASK" --lr "$lr" --seed "$s" \
        --epochs "$EPOCHS" --out "$OUT" | grep RESULT
    done
  done
done
