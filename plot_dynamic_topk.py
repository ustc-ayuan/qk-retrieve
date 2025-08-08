import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_dynamic_topk_distribution(log_dir='logs_sum'):
    """
    Analyzes and visualizes the distribution of dynamic top-K values from log files.

    This function scans a directory for experiment-specific subdirectories, reads the 
    'dynamic_topk.log' from each, and generates box plots to illustrate the distribution 
    of top-K values across different layers for each experimental configuration.

    Args:
        log_dir (str): The root directory containing the log files for all experiments.
    """
    # Find all experiment subdirectories in the log directory
    experiment_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    # Sort experiment directories by block_size and threshold
    def sort_key(exp_dir):
        # Use regular expressions to extract block_size and threshold
        match = re.search(r'block_size_(\d+)_threshold_(\d+\.\d+)', exp_dir)
        if match:
            block_size = int(match.group(1))
            threshold = float(match.group(2))
            return (block_size, threshold)
        else:
            raise ValueError(f"Directory name '{exp_dir}' does not match expected format 'block_size_<int>_threshold_<float>'")

    experiment_dirs.sort(key=sort_key)

    # Prepare a figure to hold all the plots
    num_experiments = len(experiment_dirs)
    if num_experiments == 0:
        print("No experiment log directories found.")
        return
        
    fig, axes = plt.subplots(num_experiments, 1, figsize=(15, 8 * num_experiments), squeeze=False)
    fig.suptitle('Distribution of Dynamic Top-K Values Across Layers', fontsize=20)

    stats_data = []
    # Process each experiment's log file
    for i, exp_dir in enumerate(experiment_dirs):
        log_file = os.path.join(log_dir, exp_dir, 'dynamic_topk.log')
        
        if not os.path.exists(log_file):
            print(f"Log file not found for experiment: {exp_dir}")
            continue

        # Read and process the data
        df = pd.read_csv(log_file)
        
        # Calculate overall statistics for the experiment
        overall_mean = df['max_k'].mean()
        overall_p90 = df['max_k'].quantile(0.9)

        stats_data.append({
            'Experiment': exp_dir,
            'Average Top-K': f'{overall_mean:.2f}',
            'P90 Top-K': f'{overall_p90:.2f}'
        })

        # Create a box plot for the current experiment
        ax = axes[i, 0]
        sns.boxplot(x='layer_idx', y='max_k', data=df, ax=ax)
        ax.set_title(f'Experiment: {exp_dir}', fontsize=18)
        ax.set_xlabel('Layer Index', fontsize=18)
        ax.set_ylabel('Dynamic Top-K (max_k)', fontsize=18)
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Print the statistics table
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print("Summary of Dynamic Top-K Statistics:")
        print(stats_df.to_string(index=False))

    # Save the combined plot
    output_path = os.path.join(log_dir, 'dynamic_topk_distribution.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_dynamic_topk_distribution()
