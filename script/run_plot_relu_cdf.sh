#!/bin/bash

#!/bin/bash

# This script runs the plot_scores.py script for a range of cnt and layer values.

# Define the directory containing the .pt files
PT_DIR="./detailed_related_score/block_size_16/pts"

# Define the token and head slices
TOKEN_POS="[0]"
HEAD_POS="[0]"


# Define the range for cnt and the specific values for layer
CNTS=$(seq 1 1)
LAYERS=(23)
HEADS=(0 4 8 12 16 20 24 28)
TOKENS=$(seq 0 0)
# Loop through each combination of cnt and layer
for head in "${HEADS[@]}"
do
  for layer in "${LAYERS[@]}"
  do
    for pos in $TOKENS
    do
        for cnt in $CNTS
        do
            echo "Generating plot for cnt=${cnt}, layer=${layer}"
            python3 plot_relu_cdf.py \
            "${PT_DIR}" \
            --cnt ${cnt} \
            --layer ${layer} \
            --token_pos "[${pos}]" \
            --head "[${head}]"    
            #--head "[:]"    
        done
    done
  done
done

echo "Finished generating all plots."