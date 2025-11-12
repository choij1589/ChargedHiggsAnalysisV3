#!/usr/bin/env python3
"""
Generate heatmaps showing discrimination variable correctness across MHc-MA grid.
Creates one heatmap per variable with MHc on x-axis and MA on y-axis.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

WORKDIR = os.environ['WORKDIR']

# Mass point structure (irregular grid)
MHC_VALUES = [70, 85, 100, 115, 130, 145, 160]
MA_MAP = {
    70: [15, 18, 40, 55, 65],
    85: [15, 21, 70, 80],
    100: [15, 24, 60, 75, 95],
    115: [15, 27, 87, 110],
    130: [15, 30, 55, 83, 90, 100, 125],
    145: [15, 35, 92, 140],
    160: [15, 50, 85, 98, 120, 135, 155]
}

# All 27 discrimination variables
DISCRIMINATION_VARIABLES = [
    "acoplanarity_correct",
    "scalarPtSum_correct",
    "ptAsymmetry_correct",
    "gammaFactor_smaller_correct",
    "gammaFactor_larger_correct",
    "gammaAcop_smaller_correct",
    "gammaAcop_larger_correct",
    "deltaR_pair_mu3rd_larger_correct",
    "deltaR_pair_mu3rd_smaller_correct",
    "deltaPhi_pair_mu3rd_larger_correct",
    "ptRatio_mu3rd_smaller_correct",
    "deltaR_nearestBjet_smaller_correct",
    "deltaR_nearestBjet_larger_correct",
    "deltaR_leadingNonBjet_smaller_correct",
    "deltaR_leadingNonBjet_larger_correct",
    "deltaPhi_pair_MET_smaller_correct",
    "deltaPhi_pair_MET_larger_correct",
    "MT_pair_MET_smaller_correct",
    "MT_pair_MET_larger_correct",
    "deltaPhi_muSS_MET_larger_correct",
    "deltaPhi_muSS_MET_smaller_correct",
    "MT_muSS_MET_larger_correct",
    "MT_muSS_MET_smaller_correct",
    "MT_asymmetry_smaller_correct",
    "MT_asymmetry_larger_correct",
    "mass_smaller_correct",
    "mass_larger_correct",
]


def format_variable_name(var_name):
    """Clean up variable name for display."""
    # Remove "_correct" suffix
    name = var_name.replace('_correct', '')

    # Format special cases
    replacements = {
        'deltaR': 'ΔR',
        'deltaPhi': 'Δφ',
        'deltaEta': 'Δη',
        'muSS': 'μ_SS',
        'mu3rd': 'μ_3rd',
        'MET': 'E_T^miss',
        'MT': 'M_T',
        'pair': 'pair',
        'gamma': 'γ',
        'Acop': 'Acop',
        '_': ' '
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name.split())

    return name


def create_data_matrix(data, variable):
    """
    Create 2D matrix for heatmap with MHc on x-axis and MA on y-axis.
    Uses all unique MA values as fixed y-axis labels.

    Args:
        data: Dictionary with structure {mass_point: {variable: metrics}}
        variable: Discrimination variable name

    Returns:
        tuple: (data_matrix, ma_values_sorted) where data_matrix is 2D numpy array
               and ma_values_sorted is list of all unique MA values
    """
    # Collect all unique MA values across all mass points
    all_ma_values = set()
    for ma_list in MA_MAP.values():
        all_ma_values.update(ma_list)

    # Sort MA values
    ma_values_sorted = sorted(all_ma_values)

    # Initialize matrix with NaN (for missing combinations)
    data_matrix = np.full((len(ma_values_sorted), len(MHC_VALUES)), np.nan)

    # Fill in data for existing MHc-MA combinations
    for mhc_idx, mhc in enumerate(MHC_VALUES):
        for ma in MA_MAP[mhc]:
            ma_idx = ma_values_sorted.index(ma)
            mass_point = f"MHc{mhc}_MA{ma}"
            if mass_point in data and variable in data[mass_point]:
                correctness = data[mass_point][variable]['correctness']
                data_matrix[ma_idx, mhc_idx] = correctness

    return data_matrix, ma_values_sorted


def plot_heatmap(data, variable, output_path):
    """
    Generate heatmap for a single discrimination variable.

    Args:
        data: Dictionary with discrimination summary data
        variable: Variable name to plot
        output_path: Output file path
    """
    # Create data matrix with fixed MA values
    data_matrix, ma_values_sorted = create_data_matrix(data, variable)

    # Create figure (adjust height based on number of MA values)
    fig_height = max(8, len(ma_values_sorted) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Create custom colormap with gray for NaN values
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#EEEEEE')  # Light gray for missing values

    # Plot heatmap
    im = ax.imshow(data_matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto',
                   interpolation='nearest')

    # Set x-axis (MHc values)
    ax.set_xticks(np.arange(len(MHC_VALUES)))
    ax.set_xticklabels([str(mhc) for mhc in MHC_VALUES])
    ax.set_xlabel(r'$M_{H^{\pm}}$ [GeV]', fontsize=12)

    # Set y-axis (fixed MA values)
    ax.set_yticks(np.arange(len(ma_values_sorted)))
    ax.set_yticklabels([str(ma) for ma in ma_values_sorted])
    ax.set_ylabel(r'$M_A$ [GeV]', fontsize=12)

    # Add text annotations
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            value = data_matrix[i, j]
            if not np.isnan(value):
                # Always use black text
                text = ax.text(j, i, f'{value:.1f}',
                              ha="center", va="center",
                              color='black', fontsize=9,
                              weight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Correctness (%)', rotation=270, labelpad=25, fontsize=12)

    # Add title
    clean_name = format_variable_name(variable)
    ax.set_title(f'{clean_name}\nSignal Pair Selection Correctness',
                fontsize=14, pad=20, weight='bold')

    # Add grid for clarity
    ax.set_xticks(np.arange(len(MHC_VALUES)) - 0.5, minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0]) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_all_heatmaps(input_file, era, channel):
    """
    Generate heatmaps for all discrimination variables.

    Args:
        input_file: Path to discrimination_summary.json
        era: Data era
        channel: Analysis channel
    """
    # Read data
    print(f"Reading: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded data for {len(data)} mass points")

    # Create output directory
    output_dir = f"{WORKDIR}/SignalKinematics/plots/DiscriminationHeatmaps/{era}/{channel}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Generate heatmap for each variable
    print(f"Generating {len(DISCRIMINATION_VARIABLES)} heatmaps...")
    for i, variable in enumerate(DISCRIMINATION_VARIABLES, 1):
        output_path = os.path.join(output_dir, f"{variable}.png")
        print(f"[{i}/{len(DISCRIMINATION_VARIABLES)}] {variable}")

        try:
            plot_heatmap(data, variable, output_path)
        except Exception as e:
            print(f"  ERROR: Failed to generate heatmap: {e}")
            continue

    print(f"\nAll heatmaps saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate discrimination variable heatmaps (MHc vs MA)"
    )
    parser.add_argument("--input", required=True, type=str,
                       help="Input discrimination_summary.json file")
    parser.add_argument("--era", default="2017", type=str,
                       help="Data era")
    parser.add_argument("--channel", default="SR3Mu", type=str,
                       help="Analysis channel")

    args = parser.parse_args()

    print("=" * 60)
    print("Discrimination Variable Heatmap Generator")
    print("=" * 60)
    print(f"Era: {args.era}")
    print(f"Channel: {args.channel}")
    print("=" * 60)

    # Check input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Generate heatmaps
    generate_all_heatmaps(args.input, args.era, args.channel)

    print("=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
