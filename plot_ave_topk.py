import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_f1_and_tokens_by_threshold(log_dir='logs'):
    """
    Analyzes and visualizes the F1 scores and average retrieved tokens by threshold for specific block sizes.

    This function scans a directory for experiment-specific subdirectories, reads the 
    'dynamic_topk.log' from each, and generates a combined bar and line plot to illustrate 
    the F1 scores and average retrieved tokens across different thresholds, grouped by block size.

    Args:
        log_dir (str): The root directory containing the log files for all experiments.
    """
    # 提供的 F1 数据
    block_sizes = [8, 16]
    topk_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    f1_scores = {
        8: [0.2224, 0.2826, 0.3099, 0.3322, 0.3702, 0.3510, 0.3513, 0.3758, 0.3912, 0.3905],
        16: [0.2015, 0.2482, 0.3364, 0.3613, 0.3500, 0.3622, 0.3608, 0.3757, 0.3772, 0.3773]
    }

    # 准备存储平均检索到的 Token 数
    avg_retrieved_tokens = {8: [], 16: []}

    # 找到所有实验子目录
    experiment_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    # 按照 block_size 和 threshold 的字典序排序
    experiment_dirs.sort(key=lambda x: (int(re.search(r'block_size_(\d+)_threshold_(\d+\.\d+)', x).group(1)),
                                        float(re.search(r'block_size_(\d+)_threshold_(\d+\.\d+)', x).group(2))))

    # 处理每个实验的日志文件
    for exp_dir in experiment_dirs:
        log_file = os.path.join(log_dir, exp_dir, 'dynamic_topk.log')
        
        if not os.path.exists(log_file):
            print(f"Log file not found for experiment: {exp_dir}")
            continue

        # 使用正则表达式提取 block_size 和 threshold
        match = re.search(r'block_size_(\d+)_threshold_(\d+\.\d+)', exp_dir)
        if match:
            block_size = int(match.group(1))
            threshold = float(match.group(2))
        else:
            print(f"Directory name '{exp_dir}' does not match expected format 'block_size_<int>_threshold_<float>'")
            continue

        # 读取并处理数据
        df = pd.read_csv(log_file)
        avg_topk = df['max_k'].mean()
        avg_retrieved_tokens[block_size].append(avg_topk * block_size)  # 计算平均检索到的 Token 数

    # 检查是否有数据
    if not avg_retrieved_tokens[8] or not avg_retrieved_tokens[16]:
        print("No valid data found. Please check the log files.")
        return

    # 设置图形大小
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 绘制直方图（F1 值）
    bar_width = 0.35
    index = np.arange(len(topk_values))

    # 绘制 F1 值的柱状图
    ax1.bar(index - bar_width/2, f1_scores[8], bar_width, label='Block 8 F1', color='skyblue')
    ax1.bar(index + bar_width/2, f1_scores[16], bar_width, label='Block 16 F1', color='orange')
    ax1.set_xlabel('Topk Threshold', fontsize=20)
    ax1.set_ylabel('Average F1 Score', color='tab:blue', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=18)
    ax1.set_xticks(index)
    ax1.set_xticklabels(topk_values, fontsize=18)

    # 创建第二个 Y 轴（Token 数）
    ax2 = ax1.twinx()

    # 绘制 Token 数的折线图
    ax2.plot(index, avg_retrieved_tokens[8], label='Block 8 Tokens', marker='o', color='green', linewidth=2, markersize=10)
    ax2.plot(index, avg_retrieved_tokens[16], label='Block 16 Tokens', marker='o', color='red', linewidth=2, markersize=10)
    ax2.set_ylabel('Average Retrieved Tokens', color='tab:red', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=18)

    # 添加一条值为 0.3709 的横虚线
    ax1.axhline(y=0.3709, color='r', linestyle='--', linewidth=2, label='Full Context Precision')

    # 添加图例
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=18)

    # 添加标题
    # plt.title('Average F1 Score and Retrieved Tokens by Threshold for Block Sizes 8 and 16', fontsize=20)

    # 保存图表
    output_path = os.path.join(log_dir, 'f1_and_tokens_by_threshold.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_f1_and_tokens_by_threshold()