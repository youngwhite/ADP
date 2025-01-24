#!/bin/bash

# Set fold
fold=1

# Layer-wise Head Pruning Loop
for pruning_ratio in $(seq 0.1 0.1 0.9); do
  for head_pruning_ratio in $(seq 0.25 0.25 0.75); do

    # Format output file name
    output_file=$(printf "results/layerwise-taylor-fold_%d-pruning_ratio_%.2f-head_pruning_ratio_%.2f.txt" \
      "$fold" "$pruning_ratio" "$head_pruning_ratio")

    # Skip if result already exists
    if [ -f "$output_file" ]; then
      echo "--Skipping existing result: $output_file"
      continue
    fi

    # Run the pruning script
    python3 others/run-pruner.py \
      --fold "$fold" \
      --test_before_pruning \
      --pruning_type 'taylor' \
      --cuda_id 2 \
      --pruning_ratio "$pruning_ratio" \
      --prune_head_dims \
      --prune_num_heads \
      --head_pruning_ratio "$head_pruning_ratio" \
      --post_train | tee "$output_file"

    # Log result
    echo "--Results saved to $output_file"

  done
done