#!/usr/bin/env python3
"""
Split combined discrimination summary JSON into individual mass point files.
"""

import os
import sys
import argparse
import json

WORKDIR = os.environ['WORKDIR']


def split_discrimination_json(input_file, output_dir):
    """
    Split combined JSON into individual mass point files.

    Args:
        input_file: Path to combined discrimination_summary.json
        output_dir: Directory to save individual JSON files
    """
    # Read combined JSON
    print(f"Reading: {input_file}")
    with open(input_file, 'r') as f:
        combined_data = json.load(f)

    print(f"Found {len(combined_data)} mass points")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split into individual files
    for mass_point, variables in combined_data.items():
        # Extract MHc and MA values
        # Format: "MHc70_MA15" -> MHc=70, MA=15
        parts = mass_point.split('_')
        mhc = int(parts[0].replace('MHc', ''))
        ma = int(parts[1].replace('MA', ''))

        # Create individual JSON structure
        individual_data = {
            "mass_point": mass_point,
            "MHc": mhc,
            "MA": ma,
            "variables": variables
        }

        # Save to individual file
        output_file = os.path.join(output_dir, f"{mass_point}.json")
        with open(output_file, 'w') as f:
            json.dump(individual_data, f, indent=2)

        print(f"  Created: {output_file}")

    print(f"\nSuccessfully split into {len(combined_data)} individual JSON files")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split combined discrimination JSON into individual files"
    )
    parser.add_argument("--input", required=True, type=str,
                       help="Input combined JSON file")
    parser.add_argument("--output", required=True, type=str,
                       help="Output directory for individual JSON files")

    args = parser.parse_args()

    print("=" * 60)
    print("Split Discrimination JSON")
    print("=" * 60)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Split JSON
    split_discrimination_json(args.input, args.output)

    print("=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
