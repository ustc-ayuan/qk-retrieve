#!/bin/bash

# 定义模型路径
#MODEL_PATH="/mnt/sda1/Meta-Llama-3-8B-Instruct"
MODEL_PATH="/mnt/sda1/Llama-3.1-8B"

# 定义 block_size 和 topk 的可能值
BLOCK_SIZES=(0)
TOPK_VALUES=(0.1 0.3 0.5 0.7 0.9)

# 遍历所有可能的组合
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    for TOPK in "${TOPK_VALUES[@]}"; do
        # 检查是否满足条件：block_size * topk <= 16 * 512
        #if (( BLOCK_SIZE * TOPK <= 18192 )); then
            # 定义输出文件名
            OUTPUT_FILE="output/L3_DOT_1_8B/per-head/ada_topk_no_relu_rrf/all_dy_log/output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
            ANSWER_FILE="output/L3_DOT_1_8B/per-head/ada_topk_no_relu_rrf/all_dy_ans/output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
            #OUTPUT_FILE="output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
            #ANSWER_FILE="ans_block_${BLOCK_SIZE}_topk_${TOPK}.txt"

            # 运行 Python 脚本并重定向输出
            CUDA_VISIBLE_DEVICES=0 python3 test_locom_retrieve_dynamic_size.py --model_path "$MODEL_PATH"  --topk_threshold "$TOPK" --output_path "$ANSWER_FILE" > "$OUTPUT_FILE" 2>&1

            echo "Completed block_size=${BLOCK_SIZE} topk=${TOPK}, output saved to $OUTPUT_FILE"
        #fi
    done
done
