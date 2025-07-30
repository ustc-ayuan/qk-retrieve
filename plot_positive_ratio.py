import torch
import matplotlib.pyplot as plt
import argparse
import os
import glob
import numpy as np

def process_and_plot_file(pt_path: str):
    """
    Loads a score tensor, calculates the positive score ratio per head and token,
    and saves the result as a heatmap.
    """
    # Determine the output directory for figures
    pts_dir = os.path.dirname(pt_path)
    base_dir = os.path.dirname(pts_dir)
    fig_dir = os.path.join(base_dir, "Origin_static/figs_positive_ratio")
    os.makedirs(fig_dir, exist_ok=True)

    # Define the output figure path
    pt_filename = os.path.basename(pt_path)
    fig_filename = os.path.splitext(pt_filename)[0] + "_positive_ratio.png"
    fig_path = os.path.join(fig_dir, fig_filename)

    # Load the score tensor
    try:
        scores_tensor = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file {pt_path}: {e}")
        return

    # Validate tensor dimensions (bsz, head_cnt, q_len, k_len)
    if scores_tensor.dim() != 4:
        print(f"Skipping {pt_path}: Expected a 4D tensor, but got {scores_tensor.dim()}D.")
        return
        
    if scores_tensor.shape[0] != 1:
        print(f"Warning for {pt_path}: Expected batch size of 1, but got {scores_tensor.shape[0]}. Using the first batch entry.")
    
    scores_tensor = scores_tensor[0]  # Shape: (head_cnt, q_len, k_len)
    
    head_cnt, q_len, k_len = scores_tensor.shape

    if k_len == 0:
        print(f"Skipping {pt_path}: k_len is zero.")
        return

    # Calculate the ratio of positive scores for each head and token position
    # The result is a 2D tensor where each element is the ratio for that (head, token) pair
    positive_ratio = (scores_tensor > 0).float().mean(dim=-1) # Shape: (head_cnt, q_len)

    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(max(10, q_len / 5), max(8, head_cnt / 4)))
    im = ax.imshow(positive_ratio.numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Positive Score Ratio", rotation=-90, va="bottom")

    # Set labels and title
    ax.set_xlabel("Token Position (q_len)")
    ax.set_ylabel("Head Index")
    ax.set_title(f"Positive Score Ratio per Head/Token\nFile: {pt_filename}")

    # Set ticks for axes
    ax.set_xticks(np.arange(q_len))
    ax.set_yticks(np.arange(head_cnt))
    ax.set_xticklabels(np.arange(q_len))
    ax.set_yticklabels(np.arange(head_cnt))
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(head_cnt):
        for j in range(q_len):
            # Do not annotate if the plot is too dense
            if q_len < 30 and head_cnt < 30:
                text = ax.text(j, i, f"{positive_ratio[i, j]:.2f}",
                               ha="center", va="center", color="w", fontsize=8)

    fig.tight_layout()
    
    # Save the figure
    plt.savefig(fig_path)
    print(f"Heatmap saved to {fig_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze score files and plot the ratio of positive values as a heatmap.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "score_dir", 
        type=str, 
        help="Directory containing the .pt score files."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.score_dir):
        print(f"Error: Directory not found at {args.score_dir}")
        return

    # Find all .pt files in the specified directory
    pt_files = glob.glob(os.path.join(args.score_dir, "*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {args.score_dir}")
        return
        
    print(f"Found {len(pt_files)} score files to process.")

    for pt_file in pt_files:
        process_and_plot_file(pt_file)

if __name__ == "__main__":
    main()
