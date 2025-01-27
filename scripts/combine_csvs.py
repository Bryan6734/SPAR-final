"""Combine multiple CSV files into one, keeping only the latest row for each question."""

import os
import pandas as pd
from glob import glob

def combine_csvs(input_dir: str, output_file: str):
    """Combine all CSVs in input_dir into a single CSV, keeping latest rows."""
    
    # Read all CSVs
    all_files = glob(os.path.join(input_dir, "*.csv"))
    dfs = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # Add source file for debugging
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid CSV files found!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Keep only the latest row for each question
    # Sort by source_file (which contains timestamp) and drop duplicates
    latest_df = combined_df.sort_values('source_file').drop_duplicates(
        subset=['Question'], 
        keep='last'
    )
    
    # Drop the source_file column
    latest_df = latest_df.drop('source_file', axis=1)
    
    # Save to output file
    latest_df.to_csv(output_file, index=False)
    print(f"\nCombined {len(all_files)} files into {output_file}")
    print(f"Final dataset has {len(latest_df)} rows")

if __name__ == '__main__':
    input_dir = "to_be_joined"
    output_file = "combined_results.csv"
    combine_csvs(input_dir, output_file)
