"""
Academic-quality plot for Transfer Functions (S1-S4 and V1-V4)
This script creates a professional visualization of 8 binary transfer functions
used in metaheuristic optimization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib.gridspec as gridspec

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
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

# Define color palettes for the two families
SIGMOID_COLORS = {
    'S1': '#0077B6',  # Deep Blue
    'S2': '#00B4D8',  # Light Blue
    'S3': '#48CAE4',  # Cyan Blue
    'S4': '#90E0EF'   # Pale Blue
}

V_SHAPE_COLORS = {
    'V1': '#D62828',  # Deep Red
    'V2': '#F77F00',  # Orange
    'V3': '#FCBF49',  # Yellow-Orange
    'V4': '#EAE2B7'   # Light Yellow
}

# Transfer function probability calculations (without binary conversion)
def s1_prob(x):
    """S1: Sigmoid with a=2"""
    return 1 / (1 + np.exp(-2 * x))

def s2_prob(x):
    """S2: Sigmoid with a=1"""
    return 1 / (1 + np.exp(-x))

def s3_prob(x):
    """S3: Sigmoid with a=0.5"""
    return 1 / (1 + np.exp(-x/2))

def s4_prob(x):
    """S4: Sigmoid with a≈0.333"""
    return 1 / (1 + np.exp(-x/3))

def v1_prob(x):
    """V1: Error function"""
    return np.abs(erf((np.sqrt(np.pi)/2) * x))

def v2_prob(x):
    """V2: Hyperbolic tangent"""
    return np.abs(np.tanh(x))

def v3_prob(x):
    """V3: Algebraic function"""
    return np.abs(x / np.sqrt(1 + x**2))

def v4_prob(x):
    """V4: Arctangent"""
    return np.abs((2/np.pi) * np.arctan((np.pi/2) * x))

def create_transfer_functions_plot(
    save_path: str = None,
    figsize: tuple = (14, 8),
    dpi: int = 300,
    x_range: tuple = (-8, 8),
    show_grid: bool = True,
    subplot_layout: bool = False
):
    """
    Create a professional plot of all transfer functions.
    
    Args:
        save_path: Path to save the figure (if None, only displays)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        x_range: Range of x values to plot
        show_grid: Whether to show grid
        subplot_layout: If True, creates 2 subplots for each family
    """
    
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # Initialize axes variable
    axes = None
    
    if subplot_layout:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        axes = (ax1, ax2)
        
        # Plot Sigmoid family (S1-S4) on left subplot
        ax1.plot(x, s1_prob(x), color=SIGMOID_COLORS['S1'], label='S1 (a=2)', linewidth=2.5)
        ax1.plot(x, s2_prob(x), color=SIGMOID_COLORS['S2'], label='S2 (a=1)', linewidth=2.5, linestyle='--')
        ax1.plot(x, s3_prob(x), color=SIGMOID_COLORS['S3'], label='S3 (a=0.5)', linewidth=2.5, linestyle=':')
        ax1.plot(x, s4_prob(x), color=SIGMOID_COLORS['S4'], label='S4 (a≈0.333)', linewidth=2.5, linestyle='-.')
        
        ax1.set_xlabel('Decision Variable', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Transfer Probability', fontweight='bold', fontsize=13)
        ax1.set_title('Sigmoid Family (S1-S4)', fontweight='bold', fontsize=14)
        ax1.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
        ax1.grid(show_grid, alpha=0.3, linestyle='--')
        ax1.set_xlim(x_range)
        ax1.set_ylim(-0.05, 1.05)
        
        # Add box frame
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        # Plot V-shape family (V1-V4) on right subplot
        ax2.plot(x, v1_prob(x), color=V_SHAPE_COLORS['V1'], label='V1 (erf)', linewidth=2.5)
        ax2.plot(x, v2_prob(x), color=V_SHAPE_COLORS['V2'], label='V2 (tanh)', linewidth=2.5, linestyle='--')
        ax2.plot(x, v3_prob(x), color=V_SHAPE_COLORS['V3'], label='V3 (algebraic)', linewidth=2.5, linestyle=':')
        ax2.plot(x, v4_prob(x), color=V_SHAPE_COLORS['V4'], label='V4 (arctan)', linewidth=2.5, linestyle='-.')
        
        ax2.set_xlabel('Decision Variable', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Transfer Probability', fontweight='bold', fontsize=13)
        ax2.set_title('V-Shape Family (V1-V4)', fontweight='bold', fontsize=14)
        ax2.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
        ax2.grid(show_grid, alpha=0.3, linestyle='--')
        ax2.set_xlim(x_range)
        ax2.set_ylim(-0.05, 1.05)
        
        # Add box frame
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        # Add main title
        fig.suptitle('Binary Transfer Functions for Metaheuristic Optimization', 
                    fontweight='bold', fontsize=16, y=1.02)
        
    else:
        # Create single plot with all functions
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        axes = ax
        
        # Plot Sigmoid family with solid lines
        ax.plot(x, s1_prob(x), color=SIGMOID_COLORS['S1'], label='S1: Sigmoid (a=2)', 
                linewidth=2.5, linestyle='-')
        ax.plot(x, s2_prob(x), color=SIGMOID_COLORS['S2'], label='S2: Sigmoid (a=1)', 
                linewidth=2.5, linestyle='-')
        ax.plot(x, s3_prob(x), color=SIGMOID_COLORS['S3'], label='S3: Sigmoid (a=0.5)', 
                linewidth=2.5, linestyle='-')
        ax.plot(x, s4_prob(x), color=SIGMOID_COLORS['S4'], label='S4: Sigmoid (a≈0.333)', 
                linewidth=2.5, linestyle='-')
        
        # Plot V-shape family with dashed lines
        ax.plot(x, v1_prob(x), color=V_SHAPE_COLORS['V1'], label='V1: Error Function', 
                linewidth=2.5, linestyle='--')
        ax.plot(x, v2_prob(x), color=V_SHAPE_COLORS['V2'], label='V2: Tanh', 
                linewidth=2.5, linestyle='--')
        ax.plot(x, v3_prob(x), color=V_SHAPE_COLORS['V3'], label='V3: Algebraic', 
                linewidth=2.5, linestyle='--')
        ax.plot(x, v4_prob(x), color=V_SHAPE_COLORS['V4'], label='V4: Arctangent', 
                linewidth=2.5, linestyle='--')
        
        # Customize plot
        ax.set_xlabel('Decision Variable', fontweight='bold', fontsize=14)
        ax.set_ylabel('Transfer Probability', fontweight='bold', fontsize=14)
        ax.set_title('Binary Transfer Functions for Metaheuristic Optimization', 
                    fontweight='bold', fontsize=16, pad=20)
        
        # Add legend with two columns
        legend = ax.legend(loc='center left', ncol=2, frameon=True, 
                          fancybox=False, shadow=False,
                          title='Transfer Functions', title_fontsize=11)
        legend.get_frame().set_linewidth(1.5)
        
        # Set axis limits
        ax.set_xlim(x_range)
        ax.set_ylim(-0.05, 1.05)
        
        # Add grid
        ax.grid(show_grid, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add box frame (all spines visible)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        # Add horizontal reference lines
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.axhline(y=0.5, color='gray', linewidth=0.8, alpha=0.3, linestyle=':')
        ax.axhline(y=1, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")
        
        # Also save as PDF for academic publications
        if save_path.endswith('.png'):
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    # Return figure and axes
    return fig, axes

def create_comparison_table():
    """
    Create a comparison table of transfer functions for reference.
    """
    print("\n" + "="*80)
    print("TRANSFER FUNCTIONS COMPARISON TABLE")
    print("="*80)
    print("\nSigmoid Family (S-shaped curves):")
    print("-"*40)
    print("S1: f(x) = 1/(1 + exp(-2x))         - Steepest slope")
    print("S2: f(x) = 1/(1 + exp(-x))          - Standard sigmoid")
    print("S3: f(x) = 1/(1 + exp(-x/2))        - Moderate slope")
    print("S4: f(x) = 1/(1 + exp(-x/3))        - Gentlest slope")
    
    print("\nV-Shape Family (Symmetric around origin):")
    print("-"*40)
    print("V1: f(x) = |erf(√π/2 * x)|          - Error function")
    print("V2: f(x) = |tanh(x)|                - Hyperbolic tangent")
    print("V3: f(x) = |x/√(1 + x²)|            - Algebraic function")
    print("V4: f(x) = |2/π * arctan(π/2 * x)|  - Arctangent")
    print("="*80)

def main():
    """
    Main function to create all transfer function plots.
    """
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "transfer_functions_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Creating Transfer Functions Visualizations...")
    print("-" * 50)
    
    # Create single plot with all functions
    print("\n1. Creating combined plot...")
    create_transfer_functions_plot(
        save_path=os.path.join(output_dir, "transfer_functions_combined.png"),
        figsize=(14, 8),
        subplot_layout=False
    )
    
    # Create subplot version
    print("\n2. Creating subplot version...")
    create_transfer_functions_plot(
        save_path=os.path.join(output_dir, "transfer_functions_subplots.png"),
        figsize=(16, 7),
        subplot_layout=True
    )
    
    # Print comparison table
    create_comparison_table()
    
    print("\n✓ All plots have been created successfully!")
    print(f"✓ Files saved in: {output_dir}/")

if __name__ == "__main__":
    main()
