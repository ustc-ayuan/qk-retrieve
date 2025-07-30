import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def parse_slice_string(slice_str: str, max_dim: int) -> slice:
    """
    Parses a string representation of a slice and returns a slice object.
    Format can be '[start:end]' (inclusive), '[i]', or '[:]'.
    - '[:]' selects all elements.
    - '[i]' selects a single element at index i.
    - '[start:end]' selects elements from index start to end (inclusive).
    - '[:end]' selects elements from the beginning to end (inclusive).
    - '[start:]' selects elements from start to the end of the dimension.
    """
    slice_str = slice_str.strip()
    if not slice_str.startswith('[') or not slice_str.endswith(']'):
        raise ValueError(f"Invalid slice format: {slice_str}. Must be enclosed in brackets '[]'.")
    
    content = slice_str[1:-1].strip()
    
    if not content:
        raise ValueError("Empty slice '[]' is not allowed.")

    if ':' not in content:
        try:
            idx = int(content)
            if not (0 <= idx < max_dim):
                raise ValueError(f"Index {idx} is out of range for dimension of size {max_dim}.")
            return slice(idx, idx + 1)
        except ValueError:
            raise ValueError(f"Invalid index format: '{content}'. Must be an integer.")

    parts = content.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid slice format: {slice_str}. Must contain at most one ':'.")
        
    start_str, end_str = parts
    
    try:
        start = 0 if not start_str else int(start_str)
        # If end_str is empty, it means slice to the end.
        # The user wants inclusive end, so we add 1 to the parsed value.
        end = max_dim if not end_str else int(end_str) + 1
    except ValueError:
        raise ValueError(f"Invalid start/end format in '{slice_str}'. Must be integers.")

    if not (0 <= start < max_dim):
        raise ValueError(f"Start index {start} is out of range for dimension of size {max_dim}.")
    if not (0 < end <= max_dim):
        raise ValueError(f"End index {end-1} is out of range for dimension of size {max_dim}.")
    if start >= end:
        raise ValueError(f"Start index {start} must be less than end index {end-1}.")

    return slice(start, end)

def plot_scores(pt_dir, cnt, layer, token_pos_str, head_pos_str):
    """
    Loads scores from a .pt file and plots them based on specified token and head slices.
    """
    # 构建 .pt 文件路径
    pt_filename = f"cnt_{cnt}_layer_{layer}_related_score.pt"
    pt_path = os.path.join(pt_dir, pt_filename)

    # 确定 fig 保存目录
    base_dir = os.path.dirname(pt_dir)
    fig_dir = os.path.join(base_dir, "Origin_static/figs")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 加载数据
    scores_tensor = torch.load(pt_path, map_location='cpu')
    
    if scores_tensor.dim() == 4 and scores_tensor.shape[0] != 1:
        print(f"Warning: Expected bsz=1, but got shape {scores_tensor.shape}. Using first batch entry.")
    scores_tensor = scores_tensor[0] # (head_cnt, q_len, k_len)
    
    head_cnt, q_len, k_len = scores_tensor.shape

    # 解析切片字符串
    token_slice = parse_slice_string(token_pos_str, q_len)
    head_slice = parse_slice_string(head_pos_str, head_cnt)

    # 构建 fig 文件名
    tok_str_for_file = token_pos_str.replace(':', '-').replace('[','').replace(']','')
    head_str_for_file = head_pos_str.replace(':', '-').replace('[','').replace(']','')
    fig_filename = f"scores_cnt_{cnt}_layer_{layer}_tok({tok_str_for_file})_head({head_str_for_file}).png"
    fig_path = os.path.join(fig_dir, fig_filename)

    # 提取并合并分数
    # (num_heads, num_tokens, k_len)
    selected_scores = scores_tensor[head_slice, token_slice, :]
    
    # 累加分数
    # (k_len)
    aggregated_scores = selected_scores.sum(dim=0).sum(dim=0)
    
    # 排序
    sorted_scores, _ = torch.sort(aggregated_scores, descending=True)
    
    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Scores Analysis for {os.path.basename(pt_path)}\nTokens: {token_pos_str}, Heads: {head_pos_str}', fontsize=16)

    # --- 上子图: 降序排列直方图 ---
    ax1.bar(range(k_len), sorted_scores.numpy(), color='skyblue')
    ax1.set_title('Descending Sorted Scores')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Score')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 下子图: CDF ---
    cdf = torch.cumsum(sorted_scores, dim=0)
    total_sum = cdf[-1]
    if total_sum == 0:
        print("Warning: Total sum of scores is zero. Cannot plot CDF.")
        normalized_cdf = torch.zeros_like(cdf)
    else:
        normalized_cdf = cdf / total_sum
    
    ax2.plot(range(k_len), normalized_cdf.numpy(), marker='.', linestyle='-', color='coral')
    ax2.set_title('Cumulative Distribution Function (CDF) of Scores')
    ax2.set_xlabel('Top-K Documents')
    ax2.set_ylabel('Cumulative Score Percentage')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.05)

    # 标注百分位
    quantiles = [0.50, 0.75, 0.90, 0.99]
    for q in quantiles:
        top_k_count = torch.searchsorted(normalized_cdf, q).item() + 1
        if top_k_count <= k_len:
            ax2.axhline(y=q, color='grey', linestyle='--')
            ax2.axvline(x=top_k_count, color='grey', linestyle='--')
            ax2.text(top_k_count + k_len*0.01, q - 0.05, f'{int(q*100)}% at k={top_k_count}', color='black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot scores from .pt files with flexible token and head selection.",
        formatter_class=argparse.RawTextHelpFormatter # To allow for multiline help text
    )
    parser.add_argument("--pt_dir", type=str, required=True, help="Directory containing the .pt files.")
    parser.add_argument("--cnt", type=int, required=True, help="Count identifier for the .pt file.")
    parser.add_argument("--layer", type=int, required=True, help="Layer identifier for the .pt file.")
    parser.add_argument(
        "--token_pos", 
        type=str, 
        required=True, 
        help="""Specify the token positions to analyze.
The format is a string representing a Python slice. The end index is inclusive.
Examples:
  '[5]'     : Selects only the 6th token (index 5).
  '[0:10]'  : Selects the first 11 tokens (indices 0 through 10).
  '[:]'     : Selects all tokens.
  '[10:]'   : Selects tokens from index 10 to the end."""
    )
    parser.add_argument(
        "--head", 
        type=str, 
        required=True, 
        help="""Specify the heads to analyze.
The format is a string representing a Python slice. The end index is inclusive.
Examples:
  '[10]'    : Selects only the 11th head (index 10).
  '[0:15]'  : Selects the first 16 heads (indices 0 through 15).
  '[:]'     : Selects all heads.
  '[16:]'   : Selects heads from index 16 to the end."""
    )
    
    args = parser.parse_args()
    
    plot_scores(args.pt_dir, args.cnt, args.layer, args.token_pos, args.head)
