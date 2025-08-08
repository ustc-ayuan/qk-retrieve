#!/bin/bash

# 定义模型路径
#MODEL_PATH="/mnt/sda1/Meta-Llama-3-8B-Instruct"
MODEL_PATH="/mnt/sda1/Llama-3.1-8B"

# 定义 block_size 和 topk 的可能值
BLOCK_SIZES=(8)
TOPK_VALUES=(0.1 0.3 0.5 0.7 0.9)
MTS_STRATEGIES=("MAX") # "SUM" "RRF" 

# 遍历所有可能的组合
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    for TOPK in "${TOPK_VALUES[@]}"; do
        for MTS_STRATEGY in "${MTS_STRATEGIES[@]}"; do
            # 检查是否满足条件：block_size * topk <= 16 * 512
            #if (( BLOCK_SIZE * TOPK <= 18192 )); then
                # 定义输出文件名
                LOG_DIR="logs_max/block_size_${BLOCK_SIZE}_threshold_${TOPK}"
                OUTPUT_FILE="output/L3_DOT_1_8B/per-head/ada_topk_no_relu_max/all_log/output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
                ANSWER_FILE="output/L3_DOT_1_8B/per-head/ada_topk_no_relu_max/all_ans/output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
                #LOG_DIR="test/block_size_${BLOCK_SIZE}_threshold_${TOPK}_mts_${MTS_STRATEGY}"
                #OUTPUT_FILE="test/output_block_${BLOCK_SIZE}_topk_${TOPK}.txt"
                #ANSWER_FILE="test/ans_block_${BLOCK_SIZE}_topk_${TOPK}.txt"

                # 运行 Python 脚本并重定向输出
                CUDA_VISIBLE_DEVICES=1 python3 test_locom_retrieve.py --model_path "$MODEL_PATH" --block_size "$BLOCK_SIZE" --topk_threshold "$TOPK" --mts_strategy "$MTS_STRATEGY" --log_dir "$LOG_DIR" --output_path "$ANSWER_FILE" > "$OUTPUT_FILE" 2>&1

                echo "Completed block_size=${BLOCK_SIZE} topk=${TOPK} mts=${MTS_STRATEGY}, output saved to $OUTPUT_FILE"
            #fi
        done
    done
done
