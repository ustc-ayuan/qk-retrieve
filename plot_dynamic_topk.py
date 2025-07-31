import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dynamic_topk_distribution(log_dir='logs'):
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

    # Prepare a figure to hold all the plots
    num_experiments = len(experiment_dirs)
    if num_experiments == 0:
        print("No experiment log directories found.")
        return
        
    fig, axes = plt.subplots(num_experiments, 1, figsize=(15, 8 * num_experiments), squeeze=False)
    fig.suptitle('Distribution of Dynamic Top-K Values Across Layers', fontsize=16)

    # Process each experiment's log file
    for i, exp_dir in enumerate(experiment_dirs):
        log_file = os.path.join(log_dir, exp_dir, 'dynamic_topk.log')
        
        if not os.path.exists(log_file):
            print(f"Log file not found for experiment: {exp_dir}")
            continue

        # Read and process the data
        df = pd.read_csv(log_file)
        
        # Create a box plot for the current experiment
        ax = axes[i, 0]
        sns.boxplot(x='layer_idx', y='max_k', data=df, ax=ax)
        ax.set_title(f'Experiment: {exp_dir}')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Dynamic Top-K (max_k)')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the combined plot
    output_path = os.path.join(log_dir, 'dynamic_topk_distribution.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_dynamic_topk_distribution()
