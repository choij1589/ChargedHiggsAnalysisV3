#!/usr/bin/env python3
"""
Filter impacts.json to remove statistical bin nuisances and produce cleaner impact plots.

Usage:
    filterImpacts.py -i impacts.json -o impacts_filtered.json
    filterImpacts.py -i impacts.json -o impacts_top30.json --top 30
    filterImpacts.py -i impacts.json -o impacts_filtered.json --exclude "prop_bin.*" "autoMCStat.*"
    filterImpacts.py -i impacts.json -o impacts_filtered.json --include ".*lumi.*" ".*SF.*"

After filtering, use plotImpacts.py to generate the plot:
    plotImpacts.py -i impacts_filtered.json -o impacts_filtered
"""
import os
import json
import logging
import argparse
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# Default patterns to exclude (statistical bin-by-bin uncertainties)
DEFAULT_EXCLUDE_PATTERNS = [
    r'^prop_bin',      # Barlow-Beeston lite parameters
    r'^autoMCStat',    # autoMCStats parameters
]


def filter_impacts(input_file, output_file, exclude_patterns=None, include_patterns=None, top_n=None):
    """
    Filter impacts.json to remove specified nuisance parameters.

    Args:
        input_file: Path to input impacts.json
        output_file: Path to output filtered JSON
        exclude_patterns: List of regex patterns to exclude (default: stat bin nuisances)
        include_patterns: List of regex patterns to include (if specified, only matching params kept)
        top_n: If specified, keep only top N parameters by impact
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        data = json.load(f)

    if 'params' not in data:
        raise ValueError(f"Invalid impacts.json format: missing 'params' key")

    if 'POIs' not in data or len(data['POIs']) == 0:
        raise ValueError(f"Invalid impacts.json format: missing 'POIs' key")

    # Use default exclude patterns if none specified
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

    original_count = len(data['params'])
    poi_name = data['POIs'][0]['name']

    # Filter parameters
    filtered_params = []
    excluded_count = 0
    for param in data['params']:
        name = param['name']

        # Check exclusion patterns
        excluded = any(re.match(pat, name) for pat in exclude_patterns)
        if excluded:
            excluded_count += 1
            continue

        # Check inclusion patterns (if specified, only keep matching params)
        if include_patterns:
            included = any(re.search(pat, name) for pat in include_patterns)
            if not included:
                continue

        filtered_params.append(param)

    # Sort by impact magnitude (descending)
    impact_key = f'impact_{poi_name}'
    filtered_params.sort(key=lambda x: abs(x.get(impact_key, 0)), reverse=True)

    # Take top N if specified
    if top_n is not None and top_n > 0:
        filtered_params = filtered_params[:top_n]

    data['params'] = filtered_params

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logging.info(f"Original parameters: {original_count}")
    logging.info(f"Excluded (stat bins): {excluded_count}")
    logging.info(f"Filtered parameters: {len(filtered_params)}")
    logging.info(f"Output written to: {output_file}")

    return len(filtered_params)


def main():
    parser = argparse.ArgumentParser(
        description='Filter impacts.json to remove statistical bin nuisances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove stat bin nuisances (default)
  %(prog)s -i impacts.json -o impacts_filtered.json

  # Keep only top 30 impacts
  %(prog)s -i impacts.json -o impacts_top30.json --top 30

  # Custom exclude patterns
  %(prog)s -i impacts.json -o out.json --exclude "prop_bin.*" "CMS_.*_mcstat"

  # Include only specific patterns
  %(prog)s -i impacts.json -o out.json --include ".*lumi.*" ".*SF.*"

After filtering, generate plot with:
  plotImpacts.py -i impacts_filtered.json -o impacts_filtered
"""
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input impacts.json file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output filtered JSON file')
    parser.add_argument('--top', type=int, default=None,
                        help='Keep only top N parameters by impact')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Regex patterns to exclude (default: prop_bin.*, autoMCStat.*)')
    parser.add_argument('--include', nargs='+', default=None,
                        help='Regex patterns to include (only keep matching params)')
    parser.add_argument('--no-default-exclude', action='store_true',
                        help='Do not apply default exclusion patterns')

    args = parser.parse_args()

    # Determine exclude patterns
    if args.no_default_exclude:
        exclude_patterns = args.exclude if args.exclude else []
    else:
        exclude_patterns = args.exclude if args.exclude else DEFAULT_EXCLUDE_PATTERNS

    filter_impacts(
        input_file=args.input,
        output_file=args.output,
        exclude_patterns=exclude_patterns,
        include_patterns=args.include,
        top_n=args.top
    )


if __name__ == '__main__':
    main()
