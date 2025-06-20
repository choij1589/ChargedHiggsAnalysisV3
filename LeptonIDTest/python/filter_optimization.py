#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np

def filter_and_display(csv_file, fixed_var, fixed_value, region=None):
    """
    Filter CSV optimization results and display efficiency table
    
    Args:
        csv_file: Path to CSV file with optimization results
        fixed_var: Variable to fix ('miniiso' or 'sip3d')  
        fixed_value: Value of the fixed variable
        era: Era to filter (optional, if None shows all eras)
        region: Region to filter (optional, if None shows all regions)
    """
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Filter by region if specified  
    if region:
        df = df[df['Region'] == region]
        if df.empty:
            print(f"No data found for region: {region}")
            return
    
    # Filter by fixed variable
    if fixed_var.lower() == 'miniiso':
        df_filtered = df[df['miniIso_cut'] == fixed_value]
        changing_var = 'sip3d_cut'
        fixed_name = 'miniIso'
    elif fixed_var.lower() == 'sip3d':
        df_filtered = df[df['sip3d_cut'] == fixed_value]
        changing_var = 'miniIso_cut'
        fixed_name = 'SIP3D'
    else:
        print("Error: fixed_var must be 'miniiso' or 'sip3d'")
        return
    
    if df_filtered.empty:
        print(f"No data found for {fixed_name} = {fixed_value}")
        return
    
    # Get unique values for changing variable and lepton types
    changing_values = sorted(df_filtered[changing_var].unique())
    lepton_types = sorted(df_filtered['LeptonType'].unique())
    
    # Create pivot table for each era and region combination
    era_region_combinations = df_filtered[['Era', 'Region']].drop_duplicates()
    
    for _, combo in era_region_combinations.iterrows():
        era_name = combo['Era']
        region_name = combo['Region']
        
        # Filter for this era-region combination
        subset = df_filtered[(df_filtered['Era'] == era_name) & 
                           (df_filtered['Region'] == region_name)]
        
        if subset.empty:
            continue
            
        print(f"\n{'='*80}")
        print(f"Efficiency Table - Era: {era_name}, Region: {region_name}")
        print(f"Fixed: {fixed_name} = {fixed_value}")
        print(f"{'='*80}")
        
        # Create efficiency matrix
        eff_matrix = np.zeros((len(lepton_types), len(changing_values)))
        
        for i, lep_type in enumerate(lepton_types):
            for j, change_val in enumerate(changing_values):
                row = subset[(subset['LeptonType'] == lep_type) & 
                           (subset[changing_var] == change_val)]
                if not row.empty:
                    eff_matrix[i, j] = row['Efficiency'].iloc[0]
                else:
                    eff_matrix[i, j] = np.nan
        
        # Create DataFrame for nice formatting
        eff_df = pd.DataFrame(eff_matrix, 
                             index=lepton_types, 
                             columns=[f"{changing_var.replace('_cut', '')}={val}" for val in changing_values])
        
        # Format as percentage with 1 decimal place
        print(eff_df.map(lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"))
        
        # Also print event counts for reference
        print(f"\nEvent Counts (Pass/Total):")
        for i, lep_type in enumerate(lepton_types):
            print(f"{lep_type:>10}: ", end="")
            for j, change_val in enumerate(changing_values):
                row = subset[(subset['LeptonType'] == lep_type) & 
                           (subset[changing_var] == change_val)]
                if not row.empty:
                    pass_events = int(row['Pass_Events'].iloc[0])
                    total_events = int(row['Total_Events'].iloc[0])
                    print(f"{pass_events:>4}/{total_events:<6}", end=" ")
                else:
                    print("   N/A    ", end=" ")
            print()

def main():
    parser = argparse.ArgumentParser(description="Filter and display optimization results")
    parser.add_argument("--fix", required=True, choices=['miniiso', 'sip3d'], 
                       help="Variable to fix (miniiso or sip3d)")
    parser.add_argument("--value", required=True, type=float,
                       help="Value of the fixed variable")
    parser.add_argument("--era", help="Filter by specific era")
    parser.add_argument("--region", help="Filter by specific region")
    parser.add_argument("--object", choices=['muon', 'electron'], 
                       help="Object type filter (muon or electron)")
    
    args = parser.parse_args()

    csv_file = f"optimization/tightID_optimization_{args.object}_{args.era}.csv"
    
    try:
        filter_and_display(csv_file, args.fix, args.value, args.region)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.csv_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()