#!/bin/bash

seed=42  # 可以根据需要更改种子值
epochs=60
echo "Greedy Layer Pruning AST seed $seed epochs $epochs"

for fold in 1 2 3 4 5; do
    output_file="results/glp-fold_${fold}-seed_${seed}.txt"
    python3 others/run-glp.py \
      --fold "$fold" \
      --seed "$seed" \
      --epochs "$epochs"> "$output_file"
    echo "--Results saved to $output_file"
done

# CUDA_VISIBLE_DEVICES=1 sh run-baseline.sh