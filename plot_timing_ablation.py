import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_timing_ablation(log_dir='logs'):
    """
    Analyzes and visualizes timing data for an ablation study.

    This function scans a directory for experiment-specific subdirectories, reads the 
    'timing.log' from each, and generates a stacked bar chart to compare the performance 
    of different components (retrieve, attention, other) across various experimental 
    configurations.

    Args:
        log_dir (str): The root directory containing the log files for all experiments.
    """
    # Find all experiment subdirectories in the log directory
    experiment_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    if not experiment_dirs:
        print("No experiment log directories found.")
        return

    # Process each experiment's log file
    all_data = []
    for exp_dir in experiment_dirs:
        log_file = os.path.join(log_dir, exp_dir, 'timing.log')
        
        if not os.path.exists(log_file):
            print(f"Log file not found for experiment: {exp_dir}")
            continue

        # Read and process the data
        df = pd.read_csv(log_file)
        avg_times = df[['total_topk_time', 'total_concat_time', 'total_attn_time', 'other_time']].mean()
        avg_times['experiment'] = exp_dir
        all_data.append(avg_times)

    if not all_data:
        print("No timing data could be processed.")
        return

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(all_data)
    plot_df.set_index('experiment', inplace=True)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_df.plot(kind='bar', stacked=True, ax=ax)

    ax.set_title('Performance Ablation Study: Time per Request Component')
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Average Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Time Component')

    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(log_dir, 'timing_ablation.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_timing_ablation()
