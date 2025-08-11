import matplotlib.pyplot as plt
import numpy as np

def plot_f1_token_chart(label, topk, f1_scores, aver_token, baseline_value, title='F1 Score and Average Tokens vs. Top-K', output_path=None):
    """
    Generates and saves a combined line and bar chart with two y-axes and a baseline.

    Args:
        label (list of str): A list of labels for the different data series.
        topk (list of float): The values for the x-axis.
        f1_scores (list of list of float): A 2D list for the line chart (left y-axis).
        aver_token (list of list of float): A 2D list for the bar chart (right y-axis).
        baseline_value (float): The value for the baseline.
        title (str): The title of the chart.
        output_path (str, optional): The path to save the figure. If None, the plot is displayed.
    """
    # --- Basic Setup ---
    x = np.arange(len(topk))  # the label locations
    num_labels = len(label)
    width = 0.8 / num_labels  # the width of the bars for each group

    # Use a set of light, low-contrast colors for bars
    bar_colors = ['#FF9999', '#66B2FF', '#FFD700']  # Light Red, Light Blue, Light Yellow
    # Use a set of darker, high-contrast colors for lines
    line_colors = ['#FF0000', '#0000FF', '#FFFF00']  # Dark Red, Dark Blue, Dark Yellow

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Left Y-Axis (F1 Scores - Line Chart) ---
    ax1.set_xlabel('Top-K')
    ax1.set_ylabel('F1 Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(topk)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(0, max([max(series) for series in f1_scores]) * 1.1)  # 设置 y 轴范围从 0 开始

    # --- Right Y-Axis (Average Tokens - Bar Chart) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Tokens', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # --- Plot Bars ---
    for i in range(num_labels):
        # Calculate the position for each bar
        bar_position = x - (width * num_labels / 2) + (i * width) + width / 2
        ax2.bar(bar_position, aver_token[i], width, label=f'{label[i]} Tokens', color=bar_colors[i], alpha=0.5, edgecolor='grey', zorder=1)

    # --- Plot Lines ---
    for i in range(num_labels):
        ax1.plot(x, f1_scores[i], marker='o', linestyle='-', color=line_colors[i], label=f'{label[i]} F1', linewidth=2, markersize=8, zorder=3)

    # --- Plot Baseline ---
    ax1.axhline(y=baseline_value, color='tab:green', linestyle='--', label='Baseline', linewidth=2, zorder=4)

    # --- Final Touches ---
    fig.suptitle(title, fontsize=16)
    
    # Combine legends from both axes
    lines, labels_ax1 = ax1.get_legend_handles_labels()
    bars, labels_ax2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + bars, labels_ax1 + labels_ax2, loc='upper left')

    fig.tight_layout()  # Adjust layout to prevent labels from overlapping

    # --- Save or Show Plot ---
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # --- Example Data (as provided in the prompt) ---
    labels = ["MAX", "SUM", "RRF"]
    topk_values = [0.1, 0.3, 0.5, 0.7]
    
    # F1 scores for each model at each topk value
    f1_data = [
        [32.16, 40.93, 43.95, 44.68],
        [25.16, 38.04, 42.91, 44.55],
        [20.11, 36.07, 39.46, 40.74]
    ]
    
    # Average tokens for each model at each topk value
    avg_token_data = [
        [1141.99, 3743.83, 6668.42, 10037.07],
        [967.00, 3161.52, 5666.64, 8621.61],
        [414.57, 1813.19, 4011.00, 7439.29]
    ]

    # Baseline value
    baseline_value = 46.70

    # --- Generate the Plot ---
    plot_f1_token_chart(labels, topk_values, f1_data, avg_token_data, baseline_value, output_path='f1_vs_tokens.png')
