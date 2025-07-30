#!/bin/bash

#!/bin/bash

# This script runs the plot_scores.py script for a range of cnt and layer values.

# Define the directory containing the .pt files
PT_DIR="./detailed_related_score/block_size_16/pts"

# Define the token and head slices
TOKEN_POS="[0]"
HEAD_POS="[:]"

# Define the range for cnt and the specific values for layer
CNTS=$(seq 1 1)
LAYERS=$(seq 0 10)

# Loop through each combination of cnt and layer
for cnt in $CNTS
do
  for layer in $LAYERS
  do
    echo "Generating plot for cnt=${cnt}, layer=${layer}"
    python3 plot_scores.py \
      --pt_dir "${PT_DIR}" \
      --cnt ${cnt} \
      --layer ${layer} \
      --token_pos "${TOKEN_POS}" \
      --head "${HEAD_POS}"
  done
done

echo "Finished generating all plots."
