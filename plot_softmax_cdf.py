import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def parse_slice_string(slice_str: str, max_dim: int) -> slice:
    """
    Parses a string representation of a slice and returns a slice object.
    Format can be '[start:end]' (inclusive), '[i]', or '[:]'.
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

def plot_softmax_cdf(pt_dir, cnt, layer, token_pos_str, head_pos_str):
    """
    Loads scores, applies softmax, and plots the resulting distribution and CDF.
    """
    # Build .pt file path
    pt_filename = f"cnt_{cnt}_layer_{layer}_related_score.pt"
    pt_path = os.path.join(pt_dir, pt_filename)

    if not os.path.exists(pt_path):
        print(f"Error: File not found at {pt_path}")
        return

    # Determine output directory
    pts_dir = os.path.dirname(pt_path)
    base_dir = os.path.dirname(pts_dir)
    fig_dir = os.path.join(base_dir, "Softmax_static/figs")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data
    scores_tensor = torch.load(pt_path, map_location='cpu')
    
    if scores_tensor.dim() == 4 and scores_tensor.shape[0] != 1:
        print(f"Warning: Expected bsz=1, but got shape {scores_tensor.shape}. Using first batch entry.")
    scores_tensor = scores_tensor[0] # (head_cnt, q_len, k_len)
    
    head_cnt, q_len, k_len = scores_tensor.shape

    # Parse slice strings
    token_slice = parse_slice_string(token_pos_str, q_len)
    head_slice = parse_slice_string(head_pos_str, head_cnt)

    # Build figure filename
    tok_str_for_file = token_pos_str.replace(':', '-').replace('[','').replace(']','')
    head_str_for_file = head_pos_str.replace(':', '-').replace('[','').replace(']','')
    fig_filename = f"cnt_{cnt}_layer_{layer}_tok({tok_str_for_file})_head({head_str_for_file})_softmax_cdf.png"
    fig_path = os.path.join(fig_dir, fig_filename)

    # Select and aggregate scores
    selected_scores = scores_tensor[head_slice, token_slice, :]
    aggregated_scores = selected_scores.sum(dim=0).sum(dim=0) # Shape: (k_len)
    
    # Sort original aggregated scores
    sorted_original_scores, _ = torch.sort(aggregated_scores, descending=True)

    # Apply stable softmax
    if aggregated_scores.numel() > 0:
        stable_scores = aggregated_scores - torch.max(aggregated_scores)
        softmax_scores = torch.exp(stable_scores) / torch.sum(torch.exp(stable_scores))
    else:
        print(f"Warning: No scores selected for {pt_path}. Skipping plot.")
        return

    # Sort post-softmax scores
    sorted_softmax_scores, _ = torch.sort(softmax_scores, descending=True)
    
    # Create plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Score Analysis for {os.path.basename(pt_path)}\nTokens: {token_pos_str}, Heads: {head_pos_str}', fontsize=16)

    # --- Top subplot: Bar chart of original sorted scores ---
    ax1.bar(range(k_len), sorted_original_scores.numpy(), color='cornflowerblue')
    ax1.set_title('Original Aggregated Scores (Descending)')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Score')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Middle subplot: Bar chart of sorted softmax scores ---
    ax2.bar(range(k_len), sorted_softmax_scores.numpy(), color='deepskyblue')
    ax2.set_title('Descending Sorted Softmax Scores (Probabilities)')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Probability')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Bottom subplot: CDF of softmax scores ---
    cdf = torch.cumsum(sorted_softmax_scores, dim=0)
    
    ax3.plot(range(k_len), cdf.numpy(), marker='.', linestyle='-', color='tomato')
    ax3.set_title('Cumulative Distribution Function (CDF) of Softmax Scores')
    ax3.set_xlabel('Top-K Documents')
    ax3.set_ylabel('Cumulative Probability')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(0, 1.05)

    # Annotate quantiles on the CDF plot
    quantiles = [0.50, 0.75, 0.90, 0.99]
    for q in quantiles:
        top_k_count = torch.searchsorted(cdf, q).item() + 1
        if top_k_count <= k_len:
            ax3.axhline(y=q, color='grey', linestyle='--')
            ax3.axvline(x=top_k_count, color='grey', linestyle='--')
            ax3.text(top_k_count + k_len*0.01, q - 0.05, f'{int(q*100)}% at k={top_k_count}', color='black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot original scores, softmax-normalized scores, and their CDF from .pt files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pt_dir", type=str, help="Directory containing the .pt score files.")
    parser.add_argument("--cnt", type=int, required=True, help="Count identifier for the score file.")
    parser.add_argument("--layer", type=int, required=True, help="Layer identifier for the score file.")
    parser.add_argument(
        "--token_pos", 
        type=str, 
        required=True, 
        help="""Specify token positions to analyze (inclusive slice).
Examples: '[5]', '[0:10]', '[:]', '[10:]'"""
    )
    parser.add_argument(
        "--head", 
        type=str, 
        required=True, 
        help="""Specify heads to analyze (inclusive slice).
Examples: '[10]', '[0:15]', '[:]', '[16:]'"""
    )
    
    args = parser.parse_args()
    
    plot_softmax_cdf(args.pt_dir, args.cnt, args.layer, args.token_pos, args.head)
