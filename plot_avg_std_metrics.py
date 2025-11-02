"""
Academic-quality plot for Mean ± Std metrics comparison
This script creates a grouped bar chart showing performance metrics with error bars
for different algorithms/classifiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
from typing import Dict, List, Tuple, Optional

# Set matplotlib parameters for academic publication quality
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Professional color palette
ALGORITHM_COLORS = {
    'bLRRIME-KNN': '#FFFFFF',     # Steel Blue
    'bLRRIME-SVM': '#e0e0e0',     # Royal Purple
    'bLRRIME-RF': '#e6ebf5',      # Tangerine
    'bLRRIME-CART': '#bdc2cc',    # Vermillion
    'bLRRIME-XGBOOST': '#d3e8ff', # Forest Green
    'bLRRIME-MLP': '#F0F5FF'      # Dusty Rose
}      

# Simplified algorithm names for display
ALGORITHM_DISPLAY_NAMES = {
    'bLRRIME-KNN': 'KNN',
    'bLRRIME-SVM': 'SVM',
    'bLRRIME-RF': 'RF',
    'bLRRIME-CART': 'CART',
    'bLRRIME-XGBOOST': 'XGBoost',
    'bLRRIME-MLP': 'MLP'
}

# Metrics to plot (in order)
DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC']

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and ensure all required columns are present.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
    
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_cols = ['Algorithm', 'Fold'] + DEFAULT_METRICS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
    
    return df

def calculate_statistics(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate mean and std for each algorithm and metric.
    
    Args:
        df: Input DataFrame
        metrics: List of metrics to calculate
    
    Returns:
        Nested dictionary with statistics
    """
    stats = {}
    algorithms = df['Algorithm'].unique()
    
    for algorithm in algorithms:
        algo_data = df[df['Algorithm'] == algorithm]
        stats[algorithm] = {}
        
        for metric in metrics:
            if metric in algo_data.columns:
                values = algo_data[metric].dropna()
                stats[algorithm][metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'n': len(values)
                }
            else:
                stats[algorithm][metric] = {
                    'mean': 0,
                    'std': 0,
                    'n': 0
                }
    
    return stats

def create_metrics_avg_std_plot(
    csv_path: str,
    output_dir: str = None,
    dataset_name: str = 'Dataset',
    metrics: List[str] = None,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    show_values: bool = True,
    y_limit: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create an academic-quality grouped bar chart with error bars.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save the plot (if None, uses same directory as csv)
        dataset_name: Name of the dataset for the title
        metrics: List of metrics to plot (if None, uses DEFAULT_METRICS)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        show_values: Whether to show values on top of bars
        y_limit: Optional y-axis limits (min, max)
    """
    # Load data
    df = load_and_process_data(csv_path)
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    # Filter metrics to only those present in the data
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        raise ValueError("No valid metrics found in the data")
    
    # Calculate statistics
    stats_data = calculate_statistics(df, available_metrics)
    
    # Use the order from ALGORITHM_DISPLAY_NAMES if defined, otherwise use what's in the data
    # This ensures bars follow the predefined display sequence
    ordered_algorithms = []
    for algo_key in ALGORITHM_DISPLAY_NAMES.keys():
        if algo_key in stats_data:
            ordered_algorithms.append(algo_key)
    
    # Add any algorithms not in ALGORITHM_DISPLAY_NAMES (for flexibility)
    for algo_key in stats_data.keys():
        if algo_key not in ordered_algorithms:
            ordered_algorithms.append(algo_key)
    
    sorted_algorithms = ordered_algorithms
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Set up bar positions
    n_algorithms = len(sorted_algorithms)
    n_metrics = len(available_metrics)
    width = 0.8 / n_algorithms  # Width of each bar
    x = np.arange(n_metrics)  # Label positions
    
    # Plot bars for each algorithm
    for i, algorithm in enumerate(sorted_algorithms):
        positions = x + (i - n_algorithms/2 + 0.5) * width
        
        means = [stats_data[algorithm][metric]['mean'] for metric in available_metrics]
        stds = [stats_data[algorithm][metric]['std'] for metric in available_metrics]
        
        # Get color for this algorithm
        color = ALGORITHM_COLORS.get(algorithm, '#808080')
        display_name = ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)
        
        # Plot bars with error bars
        bars = ax.bar(
            positions, means, width,
            yerr=stds,
            label=display_name,
            color=color,
            alpha=0.9,
            capsize=3,
            error_kw={
                'elinewidth': 1.5,
                'capthick': 1.5,
                'alpha': 0.7
            },
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add value labels on top of bars if requested
        if show_values:
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}',
                    ha='center', va='bottom',
                    fontsize=8,
                    rotation=0
                )
    
    # Customize the plot
    ax.set_xlabel('Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score', fontweight='bold', fontsize=14)
    ax.set_title(
        f'{dataset_name} - Performance Comparison (Mean ± Std)',
        fontweight='bold',
        fontsize=16,
        pad=20
    )
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics, fontsize=12)
    
    # Set y-axis
    if y_limit:
        ax.set_ylim(y_limit)
    else:
        ax.set_ylim(0, 1.05)  # Default for normalized metrics
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=False,
        shadow=False,
        borderpad=0.5,
        framealpha=1,
        edgecolor='black',
        ncol=1
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Add subtle background
    ax.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_metrics_avg_std_comparison.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for academic publications
    pdf_path = os.path.join(output_dir, f'{dataset_name}_metrics_avg_std_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    plt.close()

def create_metrics_table(csv_path: str, output_dir: str = None, dataset_name: str = 'Dataset') -> None:
    """
    Create a LaTeX table of the metrics for academic papers.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save the table
        dataset_name: Name of the dataset
    """
    df = load_and_process_data(csv_path)
    stats_data = calculate_statistics(df, DEFAULT_METRICS)
    
    # Create table
    table_lines = []
    table_lines.append('\\begin{table}[htbp]')
    table_lines.append('\\centering')
    table_lines.append(f'\\caption{{Performance Comparison - {dataset_name}}}')
    table_lines.append('\\begin{tabular}{l' + 'c' * len(DEFAULT_METRICS) + '}')
    table_lines.append('\\toprule')
    
    # Header
    header = 'Algorithm & ' + ' & '.join(DEFAULT_METRICS) + ' \\\\'
    table_lines.append(header)
    table_lines.append('\\midrule')
    
    # Data rows
    for algorithm in sorted(stats_data.keys()):
        display_name = ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)
        row = display_name
        for metric in DEFAULT_METRICS:
            mean = stats_data[algorithm][metric]['mean']
            std = stats_data[algorithm][metric]['std']
            row += f' & ${mean:.3f} \\pm {std:.3f}$'
        row += ' \\\\'
        table_lines.append(row)
    
    table_lines.append('\\bottomrule')
    table_lines.append('\\end{tabular}')
    table_lines.append('\\end{table}')
    
    # Save table
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    table_path = os.path.join(output_dir, f'{dataset_name}_metrics_table.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"LaTeX table saved to: {table_path}")

def main():
    parser = argparse.ArgumentParser(description='Create academic-quality metrics comparison plot')
    parser.add_argument('csv_path', help='Path to all_fold_results.csv file')
    parser.add_argument('--output_dir', help='Directory to save output files', default=None)
    parser.add_argument('--dataset_name', help='Name of the dataset', default='PulmonaryHypertension')
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot', default=None)
    parser.add_argument('--figsize', nargs=2, type=float, help='Figure size (width height)', default=[14, 8])
    parser.add_argument('--dpi', type=int, help='DPI for saved figure', default=300)
    parser.add_argument('--no_values', action='store_true', help='Do not show values on bars')
    parser.add_argument('--y_min', type=float, help='Minimum y-axis value', default=None)
    parser.add_argument('--y_max', type=float, help='Maximum y-axis value', default=None)
    parser.add_argument('--create_table', action='store_true', help='Also create LaTeX table')
    
    args = parser.parse_args()
    
    # Set y limits if provided
    y_limit = None
    if args.y_min is not None or args.y_max is not None:
        y_limit = (args.y_min or 0, args.y_max or 1.05)
    
    # Create plot
    create_metrics_avg_std_plot(
        args.csv_path,
        args.output_dir,
        args.dataset_name,
        args.metrics,
        tuple(args.figsize),
        args.dpi,
        not args.no_values,
        y_limit
    )
    
    # Create table if requested
    if args.create_table:
        create_metrics_table(args.csv_path, args.output_dir, args.dataset_name)

if __name__ == '__main__':
    # For testing/demonstration, you can run directly with:
    # Example usage:
    test_csv = r'D:\Github\fs-plot\clf_ablation_06_36_08-SINGLE_TF_MULTI_CLF-PulmonaryHypertension\detailed_data\all_fold_results.csv'
    
    if os.path.exists(test_csv):
        create_metrics_avg_std_plot(
            csv_path=test_csv,
            dataset_name='PulmonaryHypertension',
            show_values=False  # Set to False for cleaner academic look
        )
    else:
        print("Please run with command line arguments or update the test_csv path")
        main()
