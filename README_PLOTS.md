# Academic-Quality Plotting Scripts for Machine Learning Experiments

This repository contains Python scripts for creating publication-ready plots from machine learning experimental results. The scripts are designed to work with fold-level results stored in CSV format.

## 📊 Available Scripts

### 1. `plot_avg_std_metrics.py`
Creates grouped bar charts showing mean ± standard deviation for different performance metrics across algorithms.

**Features:**
- Professional color palette optimized for academic publications
- Error bars showing standard deviation
- LaTeX table generation for papers
- PDF output for vector graphics
- Customizable metrics and figure sizes

### 2. `plot_performance_metrics.py`
Creates detailed performance comparisons with multiple visualization options.

**Features:**
- **6-Panel Subplot Comparison**: Individual panels for each metric
- **Heatmap Visualization**: Overview of all metrics across algorithms
- **Boxplot Distribution**: Shows performance distribution across folds
- Highlights best performers with visual indicators
- Multiple layout options

### 3. `run_all_plots.py`
Convenience script to generate all plot types at once.

## 🚀 Quick Start

### Basic Usage

```bash
# Generate Mean ± Std comparison plot
python plot_avg_std_metrics.py "path/to/all_fold_results.csv" --dataset_name "YourDataset"

# Generate 6-panel performance comparison
python plot_performance_metrics.py "path/to/all_fold_results.csv" --plot_type bars

# Generate all plots at once
python run_all_plots.py "path/to/all_fold_results.csv" --dataset_name "YourDataset"
```

### Advanced Usage

```bash
# Customize metrics to plot
python plot_avg_std_metrics.py "data.csv" --metrics Accuracy F1 MCC --figsize 12 6

# Create heatmap visualization
python plot_performance_metrics.py "data.csv" --plot_type heatmap

# Batch process multiple experiments
python run_all_plots.py "base/directory" --batch
```

## 📁 Required Data Format

The scripts expect a CSV file (`all_fold_results.csv`) with the following columns:

- `Fold`: Fold number (e.g., 1-10)
- `Algorithm`: Algorithm name (e.g., bLRRIME-KNN)
- `Classifier`: Classifier type (e.g., knn, svm, rf)
- **Performance Metrics**: Accuracy, Precision, Recall, F1, Specificity, MCC, ROC_AUC
- Optional: Dataset, TransferFunction, Run

## 🎨 Customization

### Color Schemes

The scripts use a carefully selected color palette for different algorithms:

```python
ALGORITHM_COLORS = {
    'bLRRIME-KNN': '#2E86AB',     # Steel Blue
    'bLRRIME-SVM': '#A23B72',     # Royal Purple
    'bLRRIME-RF': '#F18F01',      # Tangerine
    'bLRRIME-CART': '#C73E1D',    # Vermillion
    'bLRRIME-XGBOOST': '#6A994E', # Forest Green
    'bLRRIME-MLP': '#BC4B51'      # Dusty Rose
}
```

### Figure Properties

All plots use Times New Roman font and academic-standard formatting:
- High DPI (300) for publication quality
- PDF output for vector graphics
- Customizable figure sizes
- Professional axis labels and legends

## 📈 Output Files

Each script generates multiple output files:

1. **PNG Files** (300 DPI): For presentations and web
2. **PDF Files**: For academic publications (vector format)
3. **LaTeX Tables** (.tex): For direct inclusion in papers

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

## 📝 Command Line Options

### plot_avg_std_metrics.py

```
Arguments:
  csv_path              Path to all_fold_results.csv file

Options:
  --output_dir DIR      Directory to save output files
  --dataset_name NAME   Name of the dataset (default: PulmonaryHypertension)
  --metrics M1 M2 ...   Specific metrics to plot
  --figsize W H         Figure size in inches (default: 14 8)
  --dpi DPI            Resolution for saved figure (default: 300)
  --no_values          Don't show values on bars
  --y_min MIN          Minimum y-axis value
  --y_max MAX          Maximum y-axis value
  --create_table       Also create LaTeX table
```

### plot_performance_metrics.py

```
Arguments:
  csv_path              Path to all_fold_results.csv file

Options:
  --output_dir DIR      Directory to save output files
  --dataset_name NAME   Name of the dataset
  --metrics M1 M2 ...   Specific metrics to plot
  --figsize W H         Figure size in inches (default: 16 10)
  --dpi DPI            Resolution (default: 300)
  --no_error_bars      Don't show error bars
  --layout ROWS COLS    Grid layout (default: 2 3)
  --plot_type TYPE      Type: bars, heatmap, boxplot, all (default: bars)
```

## 📚 Examples

### Example 1: Create all plots for a single experiment

```bash
python run_all_plots.py "clf_ablation/detailed_data/all_fold_results.csv" \
    --dataset_name "PulmonaryHypertension" \
    --output_dir "publication_figures"
```

### Example 2: Create custom metric comparison

```bash
python plot_avg_std_metrics.py "data.csv" \
    --metrics Accuracy F1 MCC \
    --figsize 10 6 \
    --y_min 0.7 \
    --y_max 1.0 \
    --create_table
```

### Example 3: Batch process multiple experiments

```bash
python run_all_plots.py "experiments_folder" --batch
```

## 🏆 Best Practices

1. **For Publications**: Always save both PNG and PDF formats
2. **For Presentations**: Use PNG with 300 DPI
3. **For Papers**: Use PDF for vector graphics and LaTeX tables
4. **Color Accessibility**: The chosen color palette is colorblind-friendly
5. **Figure Size**: Default sizes are optimized for two-column academic papers

## 📧 Support

For issues or questions about these plotting scripts, please check the data format requirements and ensure your CSV file contains all necessary columns.

## 📄 License

These scripts are provided for academic use. Please cite appropriately if used in publications.
