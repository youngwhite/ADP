#!/bin/bash

seed=42
epochs=30

for fold in 1 2 3 4 5; do
  output_file="results/baseline-fold_${fold}-seed_${seed}.txt"
  python3 run-baseline.py \
    --fold "$fold" \
    --seed "$seed" \
    --epochs "$epochs"> "$output_file"
done
echo "--Results saved to $output_file"

# CUDA_VISIBLE_DEVICES=1 sh run-baseline.sh