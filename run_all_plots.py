"""
Example script to generate all academic-quality plots from experimental results.
This script demonstrates how to use both plotting modules to create publication-ready figures.
"""

import os
import sys
from plot_avg_std_metrics import create_metrics_avg_std_plot, create_metrics_table
from plot_performance_metrics import (
    create_performance_comparison_subplots,
    create_performance_heatmap,
    create_boxplot_comparison
)

def generate_all_plots(csv_path, output_dir=None, dataset_name='PulmonaryHypertension'):
    """
    Generate all types of plots from the experimental results.
    
    Args:
        csv_path: Path to the all_fold_results.csv file
        output_dir: Directory to save all plots (if None, saves to same dir as CSV)
        dataset_name: Name of the dataset for titles
    """
    
    # Verify the CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING ACADEMIC-QUALITY PLOTS")
    print("=" * 60)
    
    # 1. Mean ± Std Comparison Plot (Grouped Bar Chart)
    print("\n1. Creating Mean ± Std Comparison Plot...")
    print("-" * 40)
    try:
        create_metrics_avg_std_plot(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            show_values=False,  # Cleaner look without values
            dpi=300
        )
        print("✓ Mean ± Std plot created successfully")
    except Exception as e:
        print(f"✗ Error creating Mean ± Std plot: {e}")
    
    # 2. LaTeX Table for Papers
    print("\n2. Creating LaTeX Table...")
    print("-" * 40)
    try:
        create_metrics_table(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_name=dataset_name
        )
        print("✓ LaTeX table created successfully")
    except Exception as e:
        print(f"✗ Error creating LaTeX table: {e}")
    
    # 3. Performance Comparison with 6 Subplots
    print("\n3. Creating Performance Comparison Subplots...")
    print("-" * 40)
    try:
        create_performance_comparison_subplots(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            figsize=(16, 10),
            dpi=300,
            show_error_bars=True
        )
        print("✓ Performance comparison subplots created successfully")
    except Exception as e:
        print(f"✗ Error creating performance comparison subplots: {e}")
    
    # 4. Performance Heatmap
    print("\n4. Creating Performance Heatmap...")
    print("-" * 40)
    try:
        create_performance_heatmap(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            figsize=(12, 8),
            dpi=300
        )
        print("✓ Performance heatmap created successfully")
    except Exception as e:
        print(f"✗ Error creating performance heatmap: {e}")
    
    # 5. Boxplot Distribution Comparison
    print("\n5. Creating Boxplot Distribution Comparison...")
    print("-" * 40)
    try:
        create_boxplot_comparison(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            figsize=(16, 10),
            dpi=300,
            layout=(2, 3)
        )
        print("✓ Boxplot comparison created successfully")
    except Exception as e:
        print(f"✗ Error creating boxplot comparison: {e}")
    
    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

def batch_process_multiple_experiments(base_dir, pattern='**/detailed_data/all_fold_results.csv'):
    """
    Process multiple experiments in batch mode.
    
    Args:
        base_dir: Base directory containing experiment folders
        pattern: Glob pattern to find CSV files
    """
    import glob
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    
    print(f"Found {len(csv_files)} experiments to process")
    
    for i, csv_path in enumerate(csv_files, 1):
        # Extract experiment name from path
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        
        print(f"\n{'='*60}")
        print(f"Processing experiment {i}/{len(csv_files)}: {exp_name}")
        print(f"{'='*60}")
        
        # Create output directory for this experiment
        output_dir = os.path.join(base_dir, 'academic_plots', exp_name)
        
        # Generate all plots
        generate_all_plots(csv_path, output_dir, exp_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate all academic-quality plots from experimental results'
    )
    
    # Add arguments
    parser.add_argument(
        'csv_path', 
        help='Path to all_fold_results.csv file'
    )
    parser.add_argument(
        '--output_dir', 
        help='Directory to save output plots',
        default=None
    )
    parser.add_argument(
        '--dataset_name', 
        help='Name of the dataset for plot titles',
        default='PulmonaryHypertension'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple experiments in batch mode'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        batch_process_multiple_experiments(args.csv_path)
    else:
        # Single file processing
        generate_all_plots(args.csv_path, args.output_dir, args.dataset_name)
    
    # Example usage (when running directly):
    # python run_all_plots.py "path/to/all_fold_results.csv" --dataset_name "MyDataset"
    
    # For testing with the available data:
    if len(sys.argv) == 1:  # No arguments provided, use test data
        test_csv = r'D:\Github\fs-plot\clf_ablation_06_36_08-SINGLE_TF_MULTI_CLF-PulmonaryHypertension\detailed_data\all_fold_results.csv'
        if os.path.exists(test_csv):
            print("Running with test data...")
            generate_all_plots(test_csv, None, 'PulmonaryHypertension')
