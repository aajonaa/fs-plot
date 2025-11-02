"""
Academic-quality performance metrics comparison plot with 6 subplots
This script creates a detailed comparison with individual metric panels for different algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
import argparse
from typing import List, Tuple, Optional, Dict
import seaborn as sns

# Set matplotlib parameters for academic publication quality
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid'] = False

# # CLF-Ablation Professional color palette
# ALGORITHM_COLORS = {
#     'bLRRIME-KNN': '#E6CCB2',  
#     'bLRRIME-SVM': '#E2C2A2',  
#     'bLRRIME-RF': '#DDB892',     
#     'bLRRIME-CART': '#D2AD88',  
#     'bLRRIME-MLP': '#CDA783',      
#     'bLRRIME-XGBOOST': '#C7A17D' 
# }      

# # Simplified algorithm names for display
# ALGORITHM_DISPLAY_NAMES = {
#     'bLRRIME-KNN': 'bLRRIME-V1-KNN',
#     'bLRRIME-SVM': 'bLRRIME-V1-SVM',
#     'bLRRIME-RF': 'bLRRIME-V1-RF',
#     'bLRRIME-CART': 'bLRRIME-V1-CART',
#     'bLRRIME-MLP': 'bLRRIME-V1-MLP',
#     'bLRRIME-XGBOOST': 'bLRRIME-V1-XGBoost'
# }

# # FS-Ablation Professional color palette
# ALGORITHM_COLORS = {
#     'bLRRIME-XGBOOST': '#ebe2cd',   
#     'XGBOOST': '#c2baa6',   
# }      

# # Simplified algorithm names for display
# ALGORITHM_DISPLAY_NAMES = {
#     'bLRRIME-XGBOOST': 'bLRRIME-V1-XGBoost',
#     'XGBOOST': 'XGBOOST',
# }

# # TF-Ablation Professional color palette (Mansully modify Algorithm col needed)
# ALGORITHM_COLORS = {
#      'bLRRIME-KNN-s1': '#0077B6',
#      'bLRRIME-KNN-s2': '#0096C7',   
#      'bLRRIME-KNN-s3': '#00B4D8',     
#      'bLRRIME-KNN-s4': '#48CAE4',    
#      'bLRRIME-KNN-v1': '#6CD5EA',     
#      'bLRRIME-KNN-v2': '#90E0EF',
#      'bLRRIME-KNN-v3': '#ADE8F4',      
#      'bLRRIME-KNN-v4': '#CAF0F8' 
# }      

# # Simplified algorithm names for display
# ALGORITHM_DISPLAY_NAMES = {
#      'bLRRIME-KNN-s1': 'bLRRIME-S1-KNN',    
#      'bLRRIME-KNN-s2': 'bLRRIME-S2-KNN',     
#      'bLRRIME-KNN-s3': 'bLRRIME-S3-KNN',     
#      'bLRRIME-KNN-s4': 'bLRRIME-S4-KNN',   
#      'bLRRIME-KNN-v1': 'bLRRIME-V1-KNN',    
#      'bLRRIME-KNN-v2': 'bLRRIME-V2-KNN',
#      'bLRRIME-KNN-v3': 'bLRRIME-V3-KNN',  
#      'bLRRIME-KNN-v4': 'bLRRIME-V4-KNN'
# }

# Comparison Professional color palette (Mansully modify Algorithm col needed)
ALGORITHM_COLORS = {
     'bLRRIME-XGBOOST': '#89B09A',
     'bRIME-XGBOOST': '#92B296',   
     'bHHO-XGBOOST': '#9AB391',     
     'bPSO-XGBOOST': '#A2B592',    
     'bGWO-XGBOOST': '#AAB693',     
     'bJADE-XGBOOST': '#B1BB9B',
     'bSADE-XGBOOST': '#BFC4AC',      
     'bSHADE-XGBOOST': '#CDCEBD' 
}      

# Simplified algorithm names for display
ALGORITHM_DISPLAY_NAMES = {
     'bLRRIME-XGBOOST': 'bLRRIME-V1-XGBOOST',
     'bRIME-XGBOOST': 'bRIME-V1-XGBOOST',   
     'bHHO-XGBOOST': 'bHHO-V1-XGBOOST',     
     'bPSO-XGBOOST': 'bPSO-V1-XGBOOST',    
     'bGWO-XGBOOST': 'bGWO-V1-XGBOOST',     
     'bJADE-XGBOOST': 'bJADE-V1-XGBOOST',
     'bSADE-XGBOOST': 'bSADE-V1-XGBOOST',      
     'bSHADE-XGBOOST': 'bSHADE-V1-XGBOOST' 
}

# Metrics to plot (in order)
DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC']

# Metric display properties
METRIC_PROPERTIES = {
    'Accuracy': {'ylabel': 'Accuracy', 'ylim': (0.5, 1.0), 'highlight': True},
    'Precision': {'ylabel': 'Precision', 'ylim': (0.5, 1.0), 'highlight': False},
    'Recall': {'ylabel': 'Recall', 'ylim': (0.5, 1.0), 'highlight': True},
    'F1': {'ylabel': 'F1-Score', 'ylim': (0.5, 1.0), 'highlight': True},
    'Specificity': {'ylabel': 'Specificity', 'ylim': (0.5, 1.0), 'highlight': False},
    'MCC': {'ylabel': 'MCC', 'ylim': (0.0, 1.0), 'highlight': False},
}

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

def calculate_algorithm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean metrics for each algorithm.
    
    Args:
        df: Input DataFrame with fold-level results
    
    Returns:
        DataFrame with mean metrics per algorithm
    """
    available_metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    
    # Group by algorithm and calculate mean
    algo_metrics = df.groupby('Algorithm')[available_metrics].agg(['mean', 'std'])
    
    return algo_metrics

def create_performance_comparison_subplots(
    csv_path: str,
    output_dir: str = None,
    dataset_name: str = 'Dataset',
    metrics: List[str] = None,
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 300,
    show_error_bars: bool = True,
    layout: Tuple[int, int] = (2, 3)
) -> None:
    """
    Create a multi-panel figure with separate subplot for each metric.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the title
        metrics: List of metrics to plot
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        show_error_bars: Whether to show error bars
        layout: Grid layout (rows, cols)
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
    algo_metrics = calculate_algorithm_metrics(df)
    
    # Use the order from ALGORITHM_DISPLAY_NAMES if defined, otherwise use what's in the data
    # This ensures bars follow the predefined display sequence
    data_algorithms = df['Algorithm'].unique()
    ordered_algorithms = []
    for algo_key in ALGORITHM_DISPLAY_NAMES.keys():
        if algo_key in data_algorithms:
            ordered_algorithms.append(algo_key)
    
    # Add any algorithms not in ALGORITHM_DISPLAY_NAMES (for flexibility)
    for algo_key in data_algorithms:
        if algo_key not in ordered_algorithms:
            ordered_algorithms.append(algo_key)
    
    algorithms = ordered_algorithms
    
    # Create figure with subplots
    n_rows, n_cols = layout
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Use GridSpec for better control
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)
    
    # Create subplots for each metric
    for idx, metric in enumerate(available_metrics[:n_rows * n_cols]):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get metric properties
        props = METRIC_PROPERTIES.get(metric, {
            'ylabel': metric,
            'ylim': (0, 1),
            'highlight': False
        })
        
        # Prepare data for this metric
        metric_data = []
        metric_std = []
        colors = []
        display_names = []
        
        for algorithm in algorithms:
            if algorithm in algo_metrics.index:
                mean_val = algo_metrics.loc[algorithm, (metric, 'mean')]
                std_val = algo_metrics.loc[algorithm, (metric, 'std')]
                metric_data.append(mean_val)
                metric_std.append(std_val)
                colors.append(ALGORITHM_COLORS.get(algorithm, '#808080'))
                display_names.append(ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm))
            else:
                metric_data.append(0)
                metric_std.append(0)
                colors.append('#808080')
                display_names.append(algorithm)
        
        # Create bar plot
        x_pos = np.arange(len(algorithms))
        
        if show_error_bars:
            bars = ax.bar(x_pos, metric_data, color=colors, alpha=0.85,
                          yerr=metric_std, capsize=4,
                          error_kw={'elinewidth': 1.2, 'capthick': 1.2},
                          edgecolor='black', linewidth=0.8)
        else:
            bars = ax.bar(x_pos, metric_data, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=0.8)
        
        # Highlight best performer
        if props.get('highlight', False):
            best_idx = np.argmax(metric_data)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2.5)
            
            # Add star marker for best
            ax.text(x_pos[best_idx], metric_data[best_idx] + 
                   (metric_std[best_idx] if show_error_bars else 0) + 0.02,
                   '*', ha='center', va='bottom', fontsize=16, color='gold', fontweight='bold')
        
        # Customize subplot
        ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12, pad=10)
        ax.set_ylabel(props['ylabel'], fontsize=11)
        ax.set_ylim(props['ylim'])
        
        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        
        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars, metric_data, metric_std)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{val:.3f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set background
        ax.set_facecolor('#FAFAFA')
    
    # Add main title
    fig.suptitle(f'{dataset_name} - Detailed Performance Comparison',
                fontweight='bold', fontsize=16, y=0.98)
    
    # Save the figure
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_performance_comparison_bars.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, f'{dataset_name}_performance_comparison_bars.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    plt.close()

def create_performance_heatmap(
    csv_path: str,
    output_dir: str = None,
    dataset_name: str = 'Dataset',
    metrics: List[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 300,
    enhance_gradient: bool = True,
    gradient_percentiles: Tuple[float, float] = (5, 95)
) -> None:
    """
    Create a heatmap showing all metrics for all algorithms.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset
        metrics: List of metrics to include
        figsize: Figure size
        dpi: Resolution
        enhance_gradient: Whether to enhance color gradient for small differences
        gradient_percentiles: Percentiles to use for gradient range (min, max)
    """
    # Load data
    df = load_and_process_data(csv_path)
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    # Filter metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Calculate mean metrics
    algo_metrics = df.groupby('Algorithm')[available_metrics].mean()
    
    # Use the order from ALGORITHM_DISPLAY_NAMES if defined, otherwise use what's in the data
    # This ensures the heatmap follows the predefined display sequence
    ordered_algorithms = []
    for algo_key in ALGORITHM_DISPLAY_NAMES.keys():
        if algo_key in algo_metrics.index:
            ordered_algorithms.append(algo_key)
    
    # Add any algorithms not in ALGORITHM_DISPLAY_NAMES (for flexibility)
    for algo_key in algo_metrics.index:
        if algo_key not in ordered_algorithms:
            ordered_algorithms.append(algo_key)
    
    # Reorder the dataframe according to our desired sequence
    algo_metrics = algo_metrics.loc[ordered_algorithms]
    
    # Rename algorithms for display
    algo_metrics.index = [ALGORITHM_DISPLAY_NAMES.get(algo, algo) for algo in algo_metrics.index]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Prepare data for better color mapping
    data_matrix = algo_metrics.values.T
    
    # Calculate color range based on enhancement settings
    if enhance_gradient:
        # Use percentiles to focus on the actual data range and enhance small differences
        data_min = np.percentile(data_matrix, gradient_percentiles[0])
        data_max = np.percentile(data_matrix, gradient_percentiles[1])
        
        # Add some padding to ensure we don't cut off extreme values
        range_padding = (data_max - data_min) * 0.1
        vmin = max(0, data_min - range_padding)
        vmax = min(1, data_max + range_padding)
        
        # If the range is too small (all values very similar), force a minimum range
        if (vmax - vmin) < 0.1:
            center = (data_matrix.min() + data_matrix.max()) / 2
            vmin = max(0, center - 0.05)
            vmax = min(1, center + 0.05)
    else:
        # Use full 0-1 range without enhancement
        vmin, vmax = 0, 1
    
    # Create heatmap with enhanced gradient
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(algo_metrics.index)))
    ax.set_yticks(np.arange(len(available_metrics)))
    ax.set_xticklabels(algo_metrics.index, fontsize=11)
    ax.set_yticklabels(available_metrics, fontsize=11)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar with range information
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(f'Performance Score\n(Range: {vmin:.3f} - {vmax:.3f})', 
    #                rotation=270, labelpad=25, fontsize=12)
    cbar.set_label('Performance Score', 
                   rotation=270, labelpad=25, fontsize=12)                   
    
    # Add text annotations
    for i in range(len(available_metrics)):
        for j in range(len(algo_metrics.index)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Set title and labels
    # ax.set_title(f'{dataset_name} - Performance Metrics Heatmap',
    #             fontweight='bold', fontsize=14, pad=20)
    ax.set_title('Algorithm Performance Metrics Heatmap',
                fontweight='bold', fontsize=14, pad=20)    

    # Tight layout
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_performance_heatmap.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {output_path}")
    
    plt.show()
    plt.close()

def create_boxplot_comparison(
    csv_path: str,
    output_dir: str = None,
    dataset_name: str = 'Dataset',
    metrics: List[str] = None,
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 300,
    layout: Tuple[int, int] = (2, 3)
) -> None:
    """
    Create boxplots for each metric showing distribution across folds.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset
        metrics: List of metrics to plot
        figsize: Figure size
        dpi: Resolution
        layout: Grid layout
    """
    # Load data
    df = load_and_process_data(csv_path)
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    # Filter metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Create figure
    n_rows, n_cols = layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='white')
    axes = axes.flatten()
    
    # Create boxplot for each metric
    for idx, metric in enumerate(available_metrics[:n_rows * n_cols]):
        ax = axes[idx]
        
        # Prepare data for boxplot
        data_for_plot = []
        labels = []
        colors = []
        
        # Use the predefined order from ALGORITHM_DISPLAY_NAMES
        data_algorithms = df['Algorithm'].unique()
        ordered_algorithms = []
        for algo_key in ALGORITHM_DISPLAY_NAMES.keys():
            if algo_key in data_algorithms:
                ordered_algorithms.append(algo_key)
        
        # Add any algorithms not in ALGORITHM_DISPLAY_NAMES
        for algo_key in data_algorithms:
            if algo_key not in ordered_algorithms:
                ordered_algorithms.append(algo_key)
        
        for algorithm in ordered_algorithms:
            algo_data = df[df['Algorithm'] == algorithm][metric].values
            data_for_plot.append(algo_data)
            labels.append(ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm))
            colors.append(ALGORITHM_COLORS.get(algorithm, '#808080'))
        
        # Create boxplot
        bp = ax.boxplot(data_for_plot, tick_labels=labels, patch_artist=True,
                       notch=True, showmeans=True,
                       meanprops=dict(marker='o', markerfacecolor='red', markersize=6),
                       medianprops=dict(linewidth=2, color='black'),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize subplot
        ax.set_title(f'{metric} Distribution', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove empty subplots
    for idx in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add main title
    fig.suptitle(f'{dataset_name} - Performance Distribution Across Folds',
                fontweight='bold', fontsize=16, y=1.02)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_performance_boxplots.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Boxplots saved to: {output_path}")
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f'{dataset_name}_performance_boxplots.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create academic-quality performance comparison plots')
    parser.add_argument('csv_path', help='Path to all_fold_results.csv file')
    parser.add_argument('--output_dir', help='Directory to save output files', default=None)
    parser.add_argument('--dataset_name', help='Name of the dataset', default='PulmonaryHypertension')
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot', default=None)
    parser.add_argument('--figsize', nargs=2, type=float, help='Figure size', default=[16, 10])
    parser.add_argument('--dpi', type=int, help='DPI for saved figure', default=300)
    parser.add_argument('--no_error_bars', action='store_true', help='Do not show error bars')
    parser.add_argument('--layout', nargs=2, type=int, help='Grid layout (rows cols)', default=[2, 3])
    parser.add_argument('--plot_type', choices=['bars', 'heatmap', 'boxplot', 'all'], 
                       default='bars', help='Type of plot to create')
    
    args = parser.parse_args()
    
    if args.plot_type == 'bars' or args.plot_type == 'all':
        create_performance_comparison_subplots(
            args.csv_path,
            args.output_dir,
            args.dataset_name,
            args.metrics,
            tuple(args.figsize),
            args.dpi,
            not args.no_error_bars,
            tuple(args.layout)
        )
    
    if args.plot_type == 'heatmap' or args.plot_type == 'all':
        create_performance_heatmap(
            args.csv_path,
            args.output_dir,
            args.dataset_name,
            args.metrics,
            tuple(args.figsize),
            args.dpi
        )
    
    if args.plot_type == 'boxplot' or args.plot_type == 'all':
        create_boxplot_comparison(
            args.csv_path,
            args.output_dir,
            args.dataset_name,
            args.metrics,
            tuple(args.figsize),
            args.dpi,
            tuple(args.layout)
        )

if __name__ == '__main__':
    # For testing/demonstration
    test_csv = r'D:\Github\fs-plot\clf_ablation_06_36_08-SINGLE_TF_MULTI_CLF-PulmonaryHypertension\detailed_data\all_fold_results.csv'
    test_csv = r'D:\Github\fs-plot\comparison_23_17_42-SINGLE_TF_SINGLE_CLF-PulmonaryHypertension\detailed_data\all_fold_results.csv'
    
    if os.path.exists(test_csv):
        # Create all three types of plots
        print("Creating performance comparison bar charts...")
        create_performance_comparison_subplots(
            csv_path=test_csv,
            dataset_name='PulmonaryHypertension'
        )
        
        print("\nCreating performance heatmap...")
        create_performance_heatmap(
            csv_path=test_csv,
            dataset_name='PulmonaryHypertension'
        )
        
        print("\nCreating performance boxplots...")
        create_boxplot_comparison(
            csv_path=test_csv,
            dataset_name='PulmonaryHypertension'
        )
    else:
        print("Please run with command line arguments or update the test_csv path")
        main()
