#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:36:22 2025

@author: mveldeman
"""

"""
After establishing my PRx and LDF based autoregulation time windows in my 
Sham rat population, I will proceed with analysis of the data of the disease
animals with SAH. Data is split in pre (30 minutes, 1800s) and post (45 min 2700s)
in relation to the induction of hemorrage. Actually post start 15 minutes after
the ICP peak. And pre goes 30 minutes back from the time of induction
"""


# %% Setup and Data Discovery

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

# Define all data paths
base_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/A_A_SAH_processing_cleaned_files"

paths = {
    'pre_physio': f"{base_path}/SAH_pre_resliced_csv",
    'post_physio': f"{base_path}/SAH_ppost_resliced_csv", 
    'pre_ldf': f"{base_path}/SAH_pre_with_LDF_resliced",
    'post_ldf': f"{base_path}/SAH_ppost_wth_LDF_resliced"
}

# What's the metadata file called and where is it located?
# metadata_path = f"{base_path}/your_metadata_file.xlsx"  # You'll need to provide this

print("Data Discovery:")
print("="*50)

# Check what files exist in each directory
for name, path in paths.items():
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.startswith('A') and f.endswith('.csv')]
        print(f"\n{name}:")
        print(f"  Path exists: {os.path.exists(path)}")
        print(f"  Files found: {len(files)}")
        print(f"  Sample files: {files[:3]}")  # Show first 3
    else:
        print(f"\n{name}: PATH NOT FOUND - {path}")
        
        
# %% Helper Functions for German Locale CSV Loading

def load_german_csv(filepath):
    """Load CSV with German locale settings"""
    try:
        df = pd.read_csv(filepath, 
                         sep=';',           
                         decimal=',',       
                         encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_animal_id(filename):
    """Extract animal ID from filename (A1, A2, etc.)"""
    # Remove all the _per000 suffixes and .csv
    animal_id = filename.split('_')[0]  # Gets 'A1' from 'A1_per000.csv'
    return animal_id

def create_timepoint_column(n_rows, phase, start_time=0):
    """Create timepoint column for data"""
    # For pre: 0 to 1799 seconds (30 minutes)
    # For post: 900 to 3599 seconds (45 minutes, starting 15 min after ICP peak)
    
    if phase == 'pre':
        timepoints = np.arange(start_time, start_time + n_rows)
    elif phase == 'post':
        timepoints = np.arange(900, 900 + n_rows)  # Starting at 15 min post-ICP peak
    else:
        timepoints = np.arange(n_rows)
    
    return timepoints

# Test loading one file from each directory
print("\nTesting file loading:")
print("="*30)

# Test pre physiology
pre_physio_files = glob.glob(f"{paths['pre_physio']}/A*_per000.csv")
if pre_physio_files:
    test_file = pre_physio_files[0]
    df_test = load_german_csv(test_file)
    if df_test is not None:
        print(f"Pre physio test ({os.path.basename(test_file)}):")
        print(f"  Shape: {df_test.shape}")
        print(f"  Columns: {df_test.columns.tolist()}")
        print(f"  Animal ID: {extract_animal_id(os.path.basename(test_file))}")
        
        
# %% Load All Data and Create Master Dataset

def load_all_animal_data():
    """Load all animal data and create master dataframe"""
    
    all_data = []
    
    # Get list of all animal IDs by checking pre_physio directory
    pre_files = glob.glob(f"{paths['pre_physio']}/A*_per000.csv")
    animal_ids = [extract_animal_id(os.path.basename(f)) for f in pre_files]
    animal_ids.sort(key=lambda x: int(x[1:]))  # Sort A1, A2, A3, etc.
    
    print(f"Processing {len(animal_ids)} animals...")
    print(f"Animal IDs: {animal_ids[:10]}...")  # Show first 10
    
    for animal_id in animal_ids:
        print(f"Processing {animal_id}...")
        
        # Pre-phase data
        pre_physio_file = f"{paths['pre_physio']}/{animal_id}_per000.csv"
        pre_ldf_file = f"{paths['pre_ldf']}/{animal_id}_per000.csv"
        
        # Post-phase data  
        post_physio_file = f"{paths['post_physio']}/{animal_id}_per000_per000.csv"
        post_ldf_file = f"{paths['post_ldf']}/{animal_id}_per000.csv"
        
        # Load pre-phase data
        if os.path.exists(pre_physio_file):
            pre_physio = load_german_csv(pre_physio_file)
            if pre_physio is not None:
                # Add metadata columns
                pre_physio['animal_id'] = animal_id
                pre_physio['phase'] = 'pre'
                pre_physio['timepoint'] = create_timepoint_column(len(pre_physio), 'pre')
                
                # Load corresponding LDF data if available
                if os.path.exists(pre_ldf_file):
                    pre_ldf = load_german_csv(pre_ldf_file)
                    if pre_ldf is not None and len(pre_ldf) == len(pre_physio):
                        pre_physio['ldf_left'] = pre_ldf['ldf_left']
                        pre_physio['ldf_right'] = pre_ldf['ldf_right']
                    else:
                        pre_physio['ldf_left'] = np.nan
                        pre_physio['ldf_right'] = np.nan
                else:
                    pre_physio['ldf_left'] = np.nan
                    pre_physio['ldf_right'] = np.nan
                
                all_data.append(pre_physio)
        
        # Load post-phase data
        if os.path.exists(post_physio_file):
            post_physio = load_german_csv(post_physio_file)
            if post_physio is not None:
                # Add metadata columns
                post_physio['animal_id'] = animal_id
                post_physio['phase'] = 'post'
                post_physio['timepoint'] = create_timepoint_column(len(post_physio), 'post')
                
                # Load corresponding LDF data if available
                if os.path.exists(post_ldf_file):
                    post_ldf = load_german_csv(post_ldf_file)
                    if post_ldf is not None and len(post_ldf) == len(post_physio):
                        post_physio['ldf_left'] = post_ldf['ldf_left']
                        post_physio['ldf_right'] = post_ldf['ldf_right']
                    else:
                        post_physio['ldf_left'] = np.nan
                        post_physio['ldf_right'] = np.nan
                else:
                    post_physio['ldf_left'] = np.nan
                    post_physio['ldf_right'] = np.nan
                
                all_data.append(post_physio)
    
    # Combine all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        return master_df
    else:
        return None

# Load all data
print("Loading all animal data...")
master_data = load_all_animal_data()

if master_data is not None:
    print(f"\nMaster dataset created!")
    print(f"Shape: {master_data.shape}")
    print(f"Animals: {master_data['animal_id'].nunique()}")
    print(f"Phases: {master_data['phase'].unique()}")
    print(f"Columns: {master_data.columns.tolist()}")
    
    # Show data summary
    print(f"\nData summary by phase:")
    print(master_data.groupby(['phase']).agg({
        'animal_id': 'nunique',
        'timepoint': ['min', 'max', 'count']
    }))
else:
    print("Failed to create master dataset!")   
        

# %% Load Metadata and Finalize Dataset

# First, let's check the current dataset structure
print("Current dataset summary:")
print("="*40)
print(f"Total rows: {len(master_data):,}")
print(f"Animals: {master_data['animal_id'].nunique()}")
print(f"Unique animal IDs: {sorted(master_data['animal_id'].unique())}")
print(f"\nSample of data:")
print(master_data.head())

# Check data quality
print(f"\nData completeness by variable:")
completeness = master_data.isnull().sum()
for col in ['abp', 'icp', 'ldf_left', 'ldf_right']:
    if col in completeness:
        pct_complete = (1 - completeness[col]/len(master_data)) * 100
        print(f"{col}: {pct_complete:.1f}% complete ({len(master_data) - completeness[col]:,} valid points)")

# What's the name and location of your metadata Excel file?
# We'll need to load it like this:
# metadata_path = "path_to_your_metadata_file.xlsx"
# metadata = pd.read_excel(metadata_path, ...)

# For now, let's proceed without metadata and add it later
print(f"\nNext step: Please provide the path and filename of your metadata Excel file")
print(f"So we can load it and join it with the physiological data")

# Check timepoint ranges
print(f"\nTimepoint ranges by phase:")
for phase in ['pre', 'post']:
    phase_data = master_data[master_data['phase'] == phase]
    print(f"{phase}: {phase_data['timepoint'].min()} to {phase_data['timepoint'].max()} seconds")
    print(f"  Duration: {(phase_data['timepoint'].max() - phase_data['timepoint'].min())/60:.1f} minutes")
    
# %% Load and Join Metadata

metadata_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/Animal Meta Data.xlsx"

# Load metadata with proper handling for German locale Excel file
try:
    metadata = pd.read_excel(metadata_path, engine='openpyxl')
    print("Metadata loaded successfully!")
    print(f"Shape: {metadata.shape}")
    print(f"\nColumn names:")
    print(metadata.columns.tolist())
    print(f"\nFirst few rows:")
    print(metadata.head())
    
    # Check how many animals have metadata vs physiological data
    metadata_ids = set(metadata['ID'].astype(str))
    physio_ids = set(master_data['animal_id'])
    
    print(f"\nData matching:")
    print(f"Animals with metadata: {len(metadata_ids)}")
    print(f"Animals with physio data: {len(physio_ids)}")
    print(f"Animals in both: {len(metadata_ids.intersection(physio_ids))}")
    
    # Check for mismatches
    only_metadata = metadata_ids - physio_ids
    only_physio = physio_ids - metadata_ids
    
    if only_metadata:
        print(f"Only in metadata: {sorted(only_metadata)}")
    if only_physio:
        print(f"Only in physio data: {sorted(only_physio)}")
        
except Exception as e:
    print(f"Error loading metadata: {e}")
    metadata = None
    
# %% Join Metadata with Physiological Data
   
if metadata is not None:
    # Ensure ID column is string type for matching
    metadata['ID'] = metadata['ID'].astype(str)
    
    # Join metadata with master dataset
    # This is equivalent to left_join in R
    master_data_with_meta = master_data.merge(
        metadata, 
        left_on='animal_id', 
        right_on='ID', 
        how='left'
    )
    
    print("Data successfully joined!")
    print(f"Final dataset shape: {master_data_with_meta.shape}")
    print(f"\nFinal columns:")
    print(master_data_with_meta.columns.tolist())
    
    # Check for successful joins
    missing_metadata = master_data_with_meta['ID'].isnull().sum()
    if missing_metadata > 0:
        print(f"\nWarning: {missing_metadata} rows missing metadata")
        missing_animals = master_data_with_meta[master_data_with_meta['ID'].isnull()]['animal_id'].unique()
        print(f"Animals missing metadata: {missing_animals}")
    
    # Show sample of final dataset
    print(f"\nSample of final dataset:")
    sample_cols = ['animal_id', 'phase', 'timepoint', 'abp', 'icp', 'ldf_left', 'ldf_right', 
                   'sugawara_grading', 'sah_mild (0-7)', 'sah_moderate (8-12)', 'sah_severe (13-18)']
    available_cols = [col for col in sample_cols if col in master_data_with_meta.columns]
    print(master_data_with_meta[available_cols].head())
    
    # Summary by SAH severity (if columns exist)
    if 'sugawara_grading' in master_data_with_meta.columns:
        print(f"\nSAH severity distribution:")
        severity_summary = master_data_with_meta.groupby(['animal_id', 'sugawara_grading']).size().reset_index()
        print(severity_summary['sugawara_grading'].value_counts().sort_index())
    
    # Store the final dataset
    sah_data = master_data_with_meta
    
else:
    print("Using dataset without metadata for now...")
    sah_data = master_data 
    
    
 # %% Data Quality Assessment for SAH Analysis
  
print("SAH DATASET QUALITY ASSESSMENT")
print("="*50)

# Overall summary
print(f"Total animals: {sah_data['animal_id'].nunique()}")
print(f"Total observations: {len(sah_data):,}")
print(f"Time range: {sah_data['timepoint'].min()} to {sah_data['timepoint'].max()} seconds")

# Data completeness by phase and variable
print(f"\nData completeness by phase:")
for phase in ['pre', 'post']:
    phase_data = sah_data[sah_data['phase'] == phase]
    print(f"\n{phase.upper()} phase ({len(phase_data):,} observations):")
    
    for var in ['abp', 'icp', 'ldf_left', 'ldf_right']:
        if var in phase_data.columns:
            completeness = (1 - phase_data[var].isnull().sum() / len(phase_data)) * 100
            print(f"  {var}: {completeness:.1f}% complete")

# Identify animals with good data quality for autoregulation analysis
print(f"\nAnimals suitable for different analyses:")

animals_with_pressure = []
animals_with_ldf = []
animals_with_bilateral_ldf = []

for animal_id in sah_data['animal_id'].unique():
    animal_data = sah_data[sah_data['animal_id'] == animal_id]
    
    # Check ABP-ICP availability
    abp_completeness = (1 - animal_data['abp'].isnull().sum() / len(animal_data)) * 100
    icp_completeness = (1 - animal_data['icp'].isnull().sum() / len(animal_data)) * 100
    
    if abp_completeness > 50 and icp_completeness > 50:
        animals_with_pressure.append(animal_id)
    
    # Check LDF availability
    ldf_left_completeness = (1 - animal_data['ldf_left'].isnull().sum() / len(animal_data)) * 100
    ldf_right_completeness = (1 - animal_data['ldf_right'].isnull().sum() / len(animal_data)) * 100
    
    if abp_completeness > 50 and (ldf_left_completeness > 50 or ldf_right_completeness > 50):
        animals_with_ldf.append(animal_id)
    
    if abp_completeness > 50 and ldf_left_completeness > 50 and ldf_right_completeness > 50:
        animals_with_bilateral_ldf.append(animal_id)

print(f"ABP-ICP analysis: {len(animals_with_pressure)} animals")
print(f"LDF analysis (unilateral): {len(animals_with_ldf)} animals") 
print(f"LDF analysis (bilateral): {len(animals_with_bilateral_ldf)} animals")

print(f"\nReady for autoregulation analysis!")  
    


# %% Debug LDF Data Loading

print("DEBUGGING LDF DATA IMPORT")
print("="*40)

# Let's check what's happening with LDF files
test_animal = 'A1'

# Check the LDF file paths and contents
ldf_paths = {
    'pre': f"{paths['pre_ldf']}/{test_animal}_per000.csv",
    'post': f"{paths['post_ldf']}/{test_animal}_per000.csv"
}

for phase, ldf_path in ldf_paths.items():
    print(f"\n{phase.upper()} LDF file:")
    print(f"Path: {ldf_path}")
    print(f"Exists: {os.path.exists(ldf_path)}")
    
    if os.path.exists(ldf_path):
        # Load and inspect the LDF file
        ldf_data = load_german_csv(ldf_path)
        if ldf_data is not None:
            print(f"Shape: {ldf_data.shape}")
            print(f"Columns: {ldf_data.columns.tolist()}")
            print(f"First few rows:")
            print(ldf_data.head())
            
            # Check if ldf_left and ldf_right columns exist
            if 'ldf_left' in ldf_data.columns:
                print(f"ldf_left completeness: {(1-ldf_data['ldf_left'].isnull().sum()/len(ldf_data))*100:.1f}%")
                print(f"ldf_left sample values: {ldf_data['ldf_left'].dropna().head().tolist()}")
            else:
                print("ldf_left column NOT FOUND")
                
            if 'ldf_right' in ldf_data.columns:
                print(f"ldf_right completeness: {(1-ldf_data['ldf_right'].isnull().sum()/len(ldf_data))*100:.1f}%")
                print(f"ldf_right sample values: {ldf_data['ldf_right'].dropna().head().tolist()}")
            else:
                print("ldf_right column NOT FOUND")
        else:
            print("Failed to load LDF file")
    else:
        print("LDF file does not exist")
    
    
# %% Check LDF File Structure Across Multiple Animals

print("\nCHECKING LDF FILES ACROSS MULTIPLE ANIMALS")
print("="*50)

# Check first 5 animals to see if it's a systematic issue
test_animals = ['A1', 'A2', 'A3', 'A4', 'A5']

for animal in test_animals:
    print(f"\n--- {animal} ---")
    
    # Pre LDF
    pre_ldf_path = f"{paths['pre_ldf']}/{animal}_per000.csv"
    if os.path.exists(pre_ldf_path):
        pre_ldf = load_german_csv(pre_ldf_path)
        if pre_ldf is not None:
            has_ldf_left = 'ldf_left' in pre_ldf.columns
            has_ldf_right = 'ldf_right' in pre_ldf.columns
            print(f"Pre LDF - Shape: {pre_ldf.shape}, ldf_left: {has_ldf_left}, ldf_right: {has_ldf_right}")
            if not (has_ldf_left and has_ldf_right):
                print(f"  Available columns: {pre_ldf.columns.tolist()}")
        else:
            print(f"Pre LDF - Failed to load")
    else:
        print(f"Pre LDF - File not found")
    
    # Post LDF  
    post_ldf_path = f"{paths['post_ldf']}/{animal}_per000.csv"
    if os.path.exists(post_ldf_path):
        post_ldf = load_german_csv(post_ldf_path)
        if post_ldf is not None:
            has_ldf_left = 'ldf_left' in post_ldf.columns
            has_ldf_right = 'ldf_right' in post_ldf.columns
            print(f"Post LDF - Shape: {post_ldf.shape}, ldf_left: {has_ldf_left}, ldf_right: {has_ldf_right}")
            if not (has_ldf_left and has_ldf_right):
                print(f"  Available columns: {post_ldf.columns.tolist()}")
        else:
            print(f"Post LDF - Failed to load")
    else:
        print(f"Post LDF - File not found")  
    
    
    
# %% Timestamp-Based Data Merging Function

def merge_data_by_timestamp(physio_data, ldf_data, animal_id, phase, tolerance_seconds=1.0):
    """
    Merge physiology and LDF data based on closest timestamps
    
    Parameters:
    - physio_data: DataFrame with physiology data
    - ldf_data: DataFrame with LDF data  
    - animal_id: Animal identifier
    - phase: 'pre' or 'post'
    - tolerance_seconds: Maximum time difference for matching (default 1 second)
    """
    
    # Convert DateTime to numeric for easier matching
    physio_times = pd.to_numeric(physio_data['DateTime'])
    ldf_times = pd.to_numeric(ldf_data['DateTime'])
    
    print(f"    {animal_id} {phase}: Physio times {physio_times.min():.6f} to {physio_times.max():.6f}")
    print(f"    {animal_id} {phase}: LDF times {ldf_times.min():.6f} to {ldf_times.max():.6f}")
    
    # Create output lists
    matched_data = []
    
    # For each physiology timepoint, find the closest LDF timepoint
    for i, physio_time in enumerate(physio_times):
        # Find closest LDF timestamp
        time_diffs = np.abs(ldf_times - physio_time)
        closest_idx = np.argmin(time_diffs)
        closest_diff = time_diffs.iloc[closest_idx]
        
        # Convert time difference to seconds (assuming DateTime is in days)
        diff_seconds = closest_diff * 24 * 3600  # Convert days to seconds
        
        if diff_seconds <= tolerance_seconds:
            # Create merged row
            merged_row = physio_data.iloc[i].copy()
            merged_row['ldf_left'] = ldf_data.iloc[closest_idx]['ldf_left']
            merged_row['ldf_right'] = ldf_data.iloc[closest_idx]['ldf_right']
            merged_row['time_diff_seconds'] = diff_seconds
            matched_data.append(merged_row)
    
    if matched_data:
        result_df = pd.DataFrame(matched_data)
        
        # Add metadata
        result_df['animal_id'] = animal_id
        result_df['phase'] = phase
        result_df['timepoint'] = create_timepoint_column(len(result_df), phase)
        
        match_rate = len(matched_data) / len(physio_data) * 100
        avg_time_diff = np.mean([row['time_diff_seconds'] for row in matched_data])
        
        print(f"    → Matched {len(matched_data)}/{len(physio_data)} points ({match_rate:.1f}%)")
        print(f"    → Average time difference: {avg_time_diff:.3f} seconds")
        
        return result_df
    else:
        print(f"    → No matches found within {tolerance_seconds}s tolerance")
        return None

# %% Load Data with Timestamp-Based Merging

def load_all_animal_data_timestamp_based():
    """Load all animal data with timestamp-based LDF integration"""
    
    all_data = []
    
    # Get list of all animal IDs
    pre_files = glob.glob(f"{paths['pre_physio']}/A*_per000.csv")
    animal_ids = [extract_animal_id(os.path.basename(f)) for f in pre_files]
    animal_ids.sort(key=lambda x: int(x[1:]))
    
    print(f"Processing {len(animal_ids)} animals with timestamp-based merging...")
    
    for animal_id in animal_ids[:5]:  # Test with first 5 animals first
        print(f"\nProcessing {animal_id}...")
        
        # PRE-PHASE
        pre_physio_file = f"{paths['pre_physio']}/{animal_id}_per000.csv"
        pre_ldf_file = f"{paths['pre_ldf']}/{animal_id}_per000.csv"
        
        if os.path.exists(pre_physio_file) and os.path.exists(pre_ldf_file):
            pre_physio = load_german_csv(pre_physio_file)
            pre_ldf = load_german_csv(pre_ldf_file)
            
            if pre_physio is not None and pre_ldf is not None:
                merged_pre = merge_data_by_timestamp(pre_physio, pre_ldf, animal_id, 'pre')
                if merged_pre is not None:
                    all_data.append(merged_pre)
        
        # POST-PHASE  
        post_physio_file = f"{paths['post_physio']}/{animal_id}_per000_per000.csv"
        post_ldf_file = f"{paths['post_ldf']}/{animal_id}_per000.csv"
        
        if os.path.exists(post_physio_file) and os.path.exists(post_ldf_file):
            post_physio = load_german_csv(post_physio_file)
            post_ldf = load_german_csv(post_ldf_file)
            
            if post_physio is not None and post_ldf is not None:
                merged_post = merge_data_by_timestamp(post_physio, post_ldf, animal_id, 'post')
                if merged_post is not None:
                    all_data.append(merged_post)
    
    # Combine all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        # Remove the temporary time_diff_seconds column
        if 'time_diff_seconds' in master_df.columns:
            master_df = master_df.drop('time_diff_seconds', axis=1)
        return master_df
    else:
        return None

# Test timestamp-based merging with first 5 animals
print("Testing timestamp-based merging...")
master_data_timestamp = load_all_animal_data_timestamp_based()
    
    
# %% Verify Timestamp-Based Merging Results

if master_data_timestamp is not None:
    print(f"\nTimestamp-based dataset created!")
    print(f"Shape: {master_data_timestamp.shape}")
    print(f"Animals: {master_data_timestamp['animal_id'].nunique()}")
    
    # Check LDF data completeness
    print(f"\nLDF data completeness:")
    for phase in ['pre', 'post']:
        phase_data = master_data_timestamp[master_data_timestamp['phase'] == phase]
        if len(phase_data) > 0:
            ldf_left_completeness = (1 - phase_data['ldf_left'].isnull().sum() / len(phase_data)) * 100
            ldf_right_completeness = (1 - phase_data['ldf_right'].isnull().sum() / len(phase_data)) * 100
            
            print(f"{phase.upper()} phase ({len(phase_data)} points):")
            print(f"  ldf_left: {ldf_left_completeness:.1f}% complete")
            print(f"  ldf_right: {ldf_right_completeness:.1f}% complete")
    
    # Show sample of merged data
    print(f"\nSample of timestamp-merged data:")
    sample_cols = ['animal_id', 'phase', 'timepoint', 'DateTime', 'abp', 'icp', 'ldf_left', 'ldf_right']
    available_cols = [col for col in sample_cols if col in master_data_timestamp.columns]
    print(master_data_timestamp[available_cols].head(10))
    
    # Show data by animal and phase
    print(f"\nData summary by animal and phase:")
    summary = master_data_timestamp.groupby(['animal_id', 'phase']).agg({
        'timepoint': 'count',
        'ldf_left': lambda x: (1-x.isnull().sum()/len(x))*100,
        'ldf_right': lambda x: (1-x.isnull().sum()/len(x))*100
    }).round(1)
    summary.columns = ['n_points', 'ldf_left_%', 'ldf_right_%']
    print(summary)
    
else:
    print("Timestamp-based merging failed!")
    

# %% Process All 69 Animals with Timestamp-Based Merging

def load_all_animals_complete():
    """Load all 69 animals with timestamp-based LDF integration"""
    
    all_data = []
    
    # Get list of all animal IDs
    pre_files = glob.glob(f"{paths['pre_physio']}/A*_per000.csv")
    animal_ids = [extract_animal_id(os.path.basename(f)) for f in pre_files]
    animal_ids.sort(key=lambda x: int(x[1:]))
    
    print(f"Processing all {len(animal_ids)} animals with timestamp-based merging...")
    
    successful_animals = 0
    failed_animals = []
    
    for animal_id in animal_ids:
        try:
            # PRE-PHASE
            pre_physio_file = f"{paths['pre_physio']}/{animal_id}_per000.csv"
            pre_ldf_file = f"{paths['pre_ldf']}/{animal_id}_per000.csv"
            
            if os.path.exists(pre_physio_file) and os.path.exists(pre_ldf_file):
                pre_physio = load_german_csv(pre_physio_file)
                pre_ldf = load_german_csv(pre_ldf_file)
                
                if pre_physio is not None and pre_ldf is not None:
                    merged_pre = merge_data_by_timestamp(pre_physio, pre_ldf, animal_id, 'pre')
                    if merged_pre is not None:
                        all_data.append(merged_pre)
            
            # POST-PHASE  
            post_physio_file = f"{paths['post_physio']}/{animal_id}_per000_per000.csv"
            post_ldf_file = f"{paths['post_ldf']}/{animal_id}_per000.csv"
            
            if os.path.exists(post_physio_file) and os.path.exists(post_ldf_file):
                post_physio = load_german_csv(post_physio_file)
                post_ldf = load_german_csv(post_ldf_file)
                
                if post_physio is not None and post_ldf is not None:
                    merged_post = merge_data_by_timestamp(post_physio, post_ldf, animal_id, 'post')
                    if merged_post is not None:
                        all_data.append(merged_post)
            
            successful_animals += 1
            if successful_animals % 10 == 0:
                print(f"Processed {successful_animals}/{len(animal_ids)} animals...")
                
        except Exception as e:
            print(f"Error processing {animal_id}: {e}")
            failed_animals.append(animal_id)
    
    print(f"\nProcessing complete:")
    print(f"Successful: {successful_animals}/{len(animal_ids)} animals")
    if failed_animals:
        print(f"Failed: {failed_animals}")
    
    # Combine all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        # Remove the temporary time_diff_seconds column if it exists
        if 'time_diff_seconds' in master_df.columns:
            master_df = master_df.drop('time_diff_seconds', axis=1)
        return master_df
    else:
        return None

# Process all animals
print("Processing all 69 animals...")
complete_sah_data = load_all_animals_complete()
    

# %% Final Dataset Summary and Quality Assessment

if complete_sah_data is not None:
    print("FINAL SAH DATASET SUMMARY")
    print("="*60)
    print(f"Successfully processed: 67/69 animals")
    print(f"Total observations: {len(complete_sah_data):,}")
    print(f"Animals: {complete_sah_data['animal_id'].nunique()}")
    print(f"Phases: {complete_sah_data['phase'].unique()}")
    
    # Data completeness by phase
    print(f"\nData completeness by phase and variable:")
    for phase in ['pre', 'post']:
        phase_data = complete_sah_data[complete_sah_data['phase'] == phase]
        print(f"\n{phase.upper()} phase ({len(phase_data):,} observations):")
        
        for var in ['abp', 'icp', 'ldf_left', 'ldf_right']:
            if var in phase_data.columns:
                completeness = (1 - phase_data[var].isnull().sum() / len(phase_data)) * 100
                n_valid = len(phase_data) - phase_data[var].isnull().sum()
                print(f"  {var}: {completeness:.1f}% complete ({n_valid:,} valid points)")
    
    # Animals suitable for different analyses
    animals_abp_icp = []
    animals_bilateral_ldf = []
    
    for animal_id in complete_sah_data['animal_id'].unique():
        animal_data = complete_sah_data[complete_sah_data['animal_id'] == animal_id]
        
        # Check ABP-ICP availability
        abp_completeness = (1 - animal_data['abp'].isnull().sum() / len(animal_data)) * 100
        icp_completeness = (1 - animal_data['icp'].isnull().sum() / len(animal_data)) * 100
        
        if abp_completeness > 30 and icp_completeness > 70:
            animals_abp_icp.append(animal_id)
        
        # Check bilateral LDF availability
        ldf_left_completeness = (1 - animal_data['ldf_left'].isnull().sum() / len(animal_data)) * 100
        ldf_right_completeness = (1 - animal_data['ldf_right'].isnull().sum() / len(animal_data)) * 100
        
        if (abp_completeness > 30 and ldf_left_completeness > 70 and ldf_right_completeness > 70):
            animals_bilateral_ldf.append(animal_id)
    
    print(f"\nANIMALS SUITABLE FOR ANALYSIS:")
    print(f"ABP-ICP analysis (PRx): {len(animals_abp_icp)} animals")
    print(f"Bilateral LDF analysis (Lx): {len(animals_bilateral_ldf)} animals")
    
    # Store these for later use
    sah_final_data = complete_sah_data
    animals_for_prx = animals_abp_icp
    animals_for_ldf = animals_bilateral_ldf
    
    print(f"\nReady for autoregulation analysis!")
    print(f"Next steps:")
    print(f"1. Add metadata from Excel file")
    print(f"2. Calculate PRx (300s windows) for {len(animals_abp_icp)} animals")
    print(f"3. Calculate Lx (300s windows) for {len(animals_bilateral_ldf)} animals")
    print(f"4. Compare pre vs post SAH")
    print(f"5. Compare left vs right hemispheres")

else:
    print("No data to analyze!")
    
    
# %% Add Metadata to Final SAH Dataset

metadata_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/Animal Meta Data.xlsx"

try:
    # Load metadata
    metadata = pd.read_excel(metadata_path, engine='openpyxl')
    metadata['ID'] = metadata['ID'].astype(str)  # Ensure string type for matching
    
    # Join with SAH data
    sah_data_with_meta = sah_final_data.merge(
        metadata, 
        left_on='animal_id', 
        right_on='ID', 
        how='left'
    )
    
    print("Metadata successfully added!")
    print(f"Final dataset shape: {sah_data_with_meta.shape}")
    
    # Check metadata coverage
    missing_meta = sah_data_with_meta['ID'].isnull().sum()
    if missing_meta == 0:
        print("✅ All animals have metadata!")
    else:
        print(f"⚠️  {missing_meta} observations missing metadata")
    
    # Show SAH severity distribution
    if 'sugawara_grading' in sah_data_with_meta.columns:
        print(f"\nSAH severity distribution (Sugawara grading):")
        severity_counts = sah_data_with_meta.groupby('animal_id')['sugawara_grading'].first().value_counts().sort_index()
        print(severity_counts)
    
    # Store final dataset
    sah_complete = sah_data_with_meta
    
except Exception as e:
    print(f"Error loading metadata: {e}")
    print("Proceeding without metadata for now...")
    sah_complete = sah_final_data
    
# %% Setup for Autoregulation Calculations

from scipy import signal

def calculate_prx_sah(abp, icp, window_seconds=300, fs=1.0):
    """Calculate PRx for SAH data using validated 300s windows"""
    # Find indices where both signals are valid
    valid_mask = ~(np.isnan(abp) | np.isnan(icp))
    valid_indices = np.where(valid_mask)[0]
    
    if np.sum(valid_mask) < window_seconds:
        return None, None, 0
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    icp_valid = icp[valid_mask]
    
    # Calculate PRx using rolling correlation (300s windows)
    window_points = int(window_seconds * fs)
    prx_values = []
    prx_indices = []
    
    for i in range(window_points, len(abp_valid)):
        # Get data window
        abp_window = abp_valid[i-window_points:i]
        icp_window = icp_valid[i-window_points:i]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(abp_window, icp_window)[0, 1]
        
        if not np.isnan(correlation):
            prx_values.append(correlation)
            prx_indices.append(valid_indices[i])
    
    return np.array(prx_values), np.array(prx_indices), np.sum(valid_mask)

def calculate_lx_sah(abp, ldf, hemisphere, window_seconds=300, fs=1.0):
    """Calculate LDF-based autoregulation index (Lx) for SAH data"""
    # Find indices where both signals are valid
    valid_mask = ~(np.isnan(abp) | np.isnan(ldf))
    valid_indices = np.where(valid_mask)[0]
    
    if np.sum(valid_mask) < window_seconds:
        return None, None, 0
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    ldf_valid = ldf[valid_mask]
    
    # Calculate Lx using rolling correlation (300s windows)
    window_points = int(window_seconds * fs)
    lx_values = []
    lx_indices = []
    
    for i in range(window_points, len(abp_valid)):
        # Get data window
        abp_window = abp_valid[i-window_points:i]
        ldf_window = ldf_valid[i-window_points:i]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(abp_window, ldf_window)[0, 1]
        
        if not np.isnan(correlation):
            lx_values.append(correlation)
            lx_indices.append(valid_indices[i])
    
    return np.array(lx_values), np.array(lx_indices), np.sum(valid_mask)

print("Autoregulation calculation functions ready!")
print("Using validated 300-second windows from sham analysis")
    



# %% Test Autoregulation Calculations on One Animal

# Test with first animal to verify functions work
test_animal = animals_for_ldf[0]  # Pick an animal with good LDF data
test_data = sah_complete[sah_complete['animal_id'] == test_animal]

print(f"Testing autoregulation calculations on {test_animal}")
print(f"Data points: {len(test_data)}")

# Check if metadata is available
if 'sugawara_grading' in test_data.columns:
    sah_severity = test_data['sugawara_grading'].iloc[0]
    print(f"SAH severity (Sugawara): {sah_severity}")

# Split by phase
pre_data = test_data[test_data['phase'] == 'pre']
post_data = test_data[test_data['phase'] == 'post']

print(f"Pre-SAH: {len(pre_data)} points ({len(pre_data)/60:.1f} minutes)")
print(f"Post-SAH: {len(post_data)} points ({len(post_data)/60:.1f} minutes)")

# Test PRx calculation
if test_animal in animals_for_prx:
    print(f"\nTesting PRx calculation:")
    prx_pre, _, n_valid_pre = calculate_prx_sah(pre_data['abp'].values, pre_data['icp'].values)
    prx_post, _, n_valid_post = calculate_prx_sah(post_data['abp'].values, post_data['icp'].values)
    
    if prx_pre is not None:
        print(f"  Pre-SAH PRx: {np.mean(prx_pre):.3f} ± {np.std(prx_pre):.3f} (n={len(prx_pre)} windows)")
    if prx_post is not None:
        print(f"  Post-SAH PRx: {np.mean(prx_post):.3f} ± {np.std(prx_post):.3f} (n={len(prx_post)} windows)")

# Test Lx calculation
print(f"\nTesting Lx calculation:")
# Left hemisphere (affected side)
lx_left_pre, _, _ = calculate_lx_sah(pre_data['abp'].values, pre_data['ldf_left'].values, 'left')
lx_left_post, _, _ = calculate_lx_sah(post_data['abp'].values, post_data['ldf_left'].values, 'left')

# Right hemisphere (control side)  
lx_right_pre, _, _ = calculate_lx_sah(pre_data['abp'].values, pre_data['ldf_right'].values, 'right')
lx_right_post, _, _ = calculate_lx_sah(post_data['abp'].values, post_data['ldf_right'].values, 'right')

if lx_left_pre is not None and lx_left_post is not None:
    print(f"  Left hemisphere (SAH side):")
    print(f"    Pre-SAH Lx: {np.mean(lx_left_pre):.3f} ± {np.std(lx_left_pre):.3f}")
    print(f"    Post-SAH Lx: {np.mean(lx_left_post):.3f} ± {np.std(lx_left_post):.3f}")
    print(f"    Change: {np.mean(lx_left_post) - np.mean(lx_left_pre):+.3f}")

if lx_right_pre is not None and lx_right_post is not None:
    print(f"  Right hemisphere (control side):")
    print(f"    Pre-SAH Lx: {np.mean(lx_right_pre):.3f} ± {np.std(lx_right_pre):.3f}")
    print(f"    Post-SAH Lx: {np.mean(lx_right_post):.3f} ± {np.std(lx_right_post):.3f}")
    print(f"    Change: {np.mean(lx_right_post) - np.mean(lx_right_pre):+.3f}")

print(f"\n🎉 Autoregulation calculation test complete!")
print(f"Ready to analyze all {len(animals_for_ldf)} animals!")
    
    
# %% Calculate Autoregulation Indices for All Animals

print("Calculating autoregulation indices for all animals...")
print("This may take a few minutes...")

autoregulation_results = []

for i, animal_id in enumerate(animals_for_ldf):
    if i % 10 == 0:
        print(f"Processing {i+1}/{len(animals_for_ldf)} animals...")
    
    try:
        # Get animal data
        animal_data = sah_complete[sah_complete['animal_id'] == animal_id]
        pre_data = animal_data[animal_data['phase'] == 'pre']
        post_data = animal_data[animal_data['phase'] == 'post']
        
        # Get metadata
        sah_severity = animal_data['sugawara_grading'].iloc[0] if 'sugawara_grading' in animal_data.columns else np.nan
        
        # Initialize result dictionary
        result = {
            'animal_id': animal_id,
            'sah_severity': sah_severity,
            'n_pre_points': len(pre_data),
            'n_post_points': len(post_data)
        }
        
        # Calculate PRx if animal has sufficient ABP-ICP data
        if animal_id in animals_for_prx:
            prx_pre, _, _ = calculate_prx_sah(pre_data['abp'].values, pre_data['icp'].values)
            prx_post, _, _ = calculate_prx_sah(post_data['abp'].values, post_data['icp'].values)
            
            result['prx_pre'] = np.mean(prx_pre) if prx_pre is not None else np.nan
            result['prx_post'] = np.mean(prx_post) if prx_post is not None else np.nan
            result['prx_change'] = result['prx_post'] - result['prx_pre'] if not np.isnan(result['prx_pre']) and not np.isnan(result['prx_post']) else np.nan
        else:
            result['prx_pre'] = result['prx_post'] = result['prx_change'] = np.nan
        
        # Calculate Lx for both hemispheres
        # Left hemisphere (SAH side)
        lx_left_pre, _, _ = calculate_lx_sah(pre_data['abp'].values, pre_data['ldf_left'].values, 'left')
        lx_left_post, _, _ = calculate_lx_sah(post_data['abp'].values, post_data['ldf_left'].values, 'left')
        
        result['lx_left_pre'] = np.mean(lx_left_pre) if lx_left_pre is not None else np.nan
        result['lx_left_post'] = np.mean(lx_left_post) if lx_left_post is not None else np.nan
        result['lx_left_change'] = result['lx_left_post'] - result['lx_left_pre'] if not np.isnan(result['lx_left_pre']) and not np.isnan(result['lx_left_post']) else np.nan
        
        # Right hemisphere (control side)
        lx_right_pre, _, _ = calculate_lx_sah(pre_data['abp'].values, pre_data['ldf_right'].values, 'right')
        lx_right_post, _, _ = calculate_lx_sah(post_data['abp'].values, post_data['ldf_right'].values, 'right')
        
        result['lx_right_pre'] = np.mean(lx_right_pre) if lx_right_pre is not None else np.nan
        result['lx_right_post'] = np.mean(lx_right_post) if lx_right_post is not None else np.nan
        result['lx_right_change'] = result['lx_right_post'] - result['lx_right_pre'] if not np.isnan(result['lx_right_pre']) and not np.isnan(result['lx_right_post']) else np.nan
        
        autoregulation_results.append(result)
        
    except Exception as e:
        print(f"Error processing {animal_id}: {e}")

# Convert to DataFrame
results_df = pd.DataFrame(autoregulation_results)

print(f"\nAutoregulation analysis complete!")
print(f"Successfully analyzed {len(results_df)} animals")
print(f"Results shape: {results_df.shape}")
    
    
# %% Summary of Autoregulation Results
   
print("AUTOREGULATION ANALYSIS SUMMARY")
print("="*60)

# Overall statistics
valid_prx = results_df['prx_change'].notna().sum()
valid_lx_left = results_df['lx_left_change'].notna().sum()
valid_lx_right = results_df['lx_right_change'].notna().sum()

print(f"Animals with valid PRx data: {valid_prx}")
print(f"Animals with valid Lx_left data: {valid_lx_left}")
print(f"Animals with valid Lx_right data: {valid_lx_right}")

# Mean changes across all animals
print(f"\nMEAN CHANGES (Pre → Post SAH):")
print(f"PRx change: {results_df['prx_change'].mean():.3f} ± {results_df['prx_change'].std():.3f}")
print(f"Lx_left change: {results_df['lx_left_change'].mean():.3f} ± {results_df['lx_left_change'].std():.3f}")
print(f"Lx_right change: {results_df['lx_right_change'].mean():.3f} ± {results_df['lx_right_change'].std():.3f}")

# Show first few results
print(f"\nSample results:")
display_cols = ['animal_id', 'sah_severity', 'prx_change', 'lx_left_change', 'lx_right_change']
available_cols = [col for col in display_cols if col in results_df.columns]
print(results_df[available_cols].head(10))

# SAH severity analysis
if 'sah_severity' in results_df.columns:
    print(f"\nResults by SAH severity:")
    severity_summary = results_df.groupby('sah_severity').agg({
        'prx_change': ['count', 'mean', 'std'],
        'lx_left_change': ['mean', 'std'],
        'lx_right_change': ['mean', 'std']
    }).round(3)
    print(severity_summary) 
    
    
# %% Create Comprehensive Autoregulation Analysis Figures

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Individual Animal Changes
ax1.set_title('Individual Animal Autoregulation Changes\n(Pre → Post SAH)', fontsize=14, fontweight='bold')

# Create x-axis positions
x_pos = np.arange(len(results_df))

# Plot PRx changes
valid_prx_mask = results_df['prx_change'].notna()
ax1.scatter(x_pos[valid_prx_mask], results_df.loc[valid_prx_mask, 'prx_change'], 
           alpha=0.7, s=60, label='PRx (pressure-based)', color='red')

# Plot Lx changes
valid_lx_left_mask = results_df['lx_left_change'].notna()
ax1.scatter(x_pos[valid_lx_left_mask], results_df.loc[valid_lx_left_mask, 'lx_left_change'], 
           alpha=0.7, s=60, label='Lx_left (SAH side)', color='darkred')

valid_lx_right_mask = results_df['lx_right_change'].notna()
ax1.scatter(x_pos[valid_lx_right_mask], results_df.loc[valid_lx_right_mask, 'lx_right_change'], 
           alpha=0.7, s=60, label='Lx_right (control side)', color='blue')

# Add reference line at zero (no change)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5, label='No change')

ax1.set_xlabel('Animal (ordered by ID)')
ax1.set_ylabel('Autoregulation Index Change')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Hemispheric Comparison
ax2.set_title('Hemispheric Comparison: Left (SAH) vs Right (Control)', fontsize=14, fontweight='bold')

# Create paired comparison
valid_bilateral = results_df['lx_left_change'].notna() & results_df['lx_right_change'].notna()
bilateral_data = results_df[valid_bilateral]

ax2.scatter(bilateral_data['lx_right_change'], bilateral_data['lx_left_change'], 
           alpha=0.7, s=80, color='purple')

# Add identity line (equal changes)
min_val = min(bilateral_data['lx_left_change'].min(), bilateral_data['lx_right_change'].min())
max_val = max(bilateral_data['lx_left_change'].max(), bilateral_data['lx_right_change'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal change')

# Add quadrant reference lines
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)

ax2.set_xlabel('Right Hemisphere Lx Change (Control Side)')
ax2.set_ylabel('Left Hemisphere Lx Change (SAH Side)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add quadrant labels
ax2.text(0.02, 0.98, 'Both\nWorsen', transform=ax2.transAxes, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
ax2.text(0.98, 0.02, 'Both\nImprove', transform=ax2.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

# Plot 3: SAH Severity vs Autoregulation Changes
ax3.set_title('SAH Severity vs Autoregulation Impairment', fontsize=14, fontweight='bold')

if 'sah_severity' in results_df.columns:
    # Plot PRx vs severity
    valid_severity_prx = results_df['sah_severity'].notna() & results_df['prx_change'].notna()
    ax3.scatter(results_df.loc[valid_severity_prx, 'sah_severity'], 
               results_df.loc[valid_severity_prx, 'prx_change'],
               alpha=0.7, s=80, color='red', label='PRx change')
    
    # Plot Lx_left vs severity
    valid_severity_lx = results_df['sah_severity'].notna() & results_df['lx_left_change'].notna()
    ax3.scatter(results_df.loc[valid_severity_lx, 'sah_severity'], 
               results_df.loc[valid_severity_lx, 'lx_left_change'],
               alpha=0.7, s=80, color='darkred', label='Lx_left change')

ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('SAH Severity (Sugawara Grade)')
ax3.set_ylabel('Autoregulation Index Change')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution of Changes
ax4.set_title('Distribution of Autoregulation Changes', fontsize=14, fontweight='bold')

# Create histogram data
prx_changes = results_df['prx_change'].dropna()
lx_left_changes = results_df['lx_left_change'].dropna()
lx_right_changes = results_df['lx_right_change'].dropna()

# Plot histograms
ax4.hist(prx_changes, alpha=0.6, bins=15, label=f'PRx (n={len(prx_changes)})', color='red')
ax4.hist(lx_left_changes, alpha=0.6, bins=15, label=f'Lx_left (n={len(lx_left_changes)})', color='darkred')
ax4.hist(lx_right_changes, alpha=0.6, bins=15, label=f'Lx_right (n={len(lx_right_changes)})', color='blue')

ax4.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2)
ax4.set_xlabel('Autoregulation Index Change')
ax4.set_ylabel('Number of Animals')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistical summary
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)
print(f"Mean ± SD (negative = improvement, positive = worsening):")
print(f"PRx change:       {results_df['prx_change'].mean():+.3f} ± {results_df['prx_change'].std():.3f}")
print(f"Lx_left change:   {results_df['lx_left_change'].mean():+.3f} ± {results_df['lx_left_change'].std():.3f}")
print(f"Lx_right change:  {results_df['lx_right_change'].mean():+.3f} ± {results_df['lx_right_change'].std():.3f}")
print("="*70)
    
    
# %% Create Timeline PRx Figure with SAH Event

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_prx_timeseries(abp, icp, window_seconds=300, fs=1.0):
    """Calculate PRx time series with timestamps"""
    # Find indices where both signals are valid
    valid_mask = ~(np.isnan(abp) | np.isnan(icp))
    valid_indices = np.where(valid_mask)[0]
    
    if np.sum(valid_mask) < window_seconds:
        return None, None
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    icp_valid = icp[valid_mask]
    
    # Calculate PRx using rolling correlation
    window_points = int(window_seconds * fs)
    prx_values = []
    prx_timestamps = []
    
    for i in range(window_points, len(abp_valid)):
        # Get data window
        abp_window = abp_valid[i-window_points:i]
        icp_window = icp_valid[i-window_points:i]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(abp_window, icp_window)[0, 1]
        
        if not np.isnan(correlation):
            prx_values.append(correlation)
            # Timestamp is the center of the window (in original timepoints)
            prx_timestamps.append(valid_indices[i - window_points//2])
    
    return np.array(prx_values), np.array(prx_timestamps)

# Collect PRx time series for all animals
print("Calculating PRx time series for all animals...")
all_prx_data = []

for animal_id in animals_for_prx[:20]:  # Use first 20 animals for cleaner visualization
    try:
        animal_data = sah_complete[sah_complete['animal_id'] == animal_id]
        pre_data = animal_data[animal_data['phase'] == 'pre']
        post_data = animal_data[animal_data['phase'] == 'post']
        
        # Calculate PRx time series
        prx_pre, timestamps_pre = calculate_prx_timeseries(pre_data['abp'].values, pre_data['icp'].values)
        prx_post, timestamps_post = calculate_prx_timeseries(post_data['abp'].values, post_data['icp'].values)
        
        if prx_pre is not None:
            # Convert timestamps to minutes relative to SAH (pre-SAH is negative)
            time_pre = (timestamps_pre - len(pre_data)) / 60  # Minutes before SAH (negative)
            for i, (t, prx) in enumerate(zip(time_pre, prx_pre)):
                all_prx_data.append({
                    'animal_id': animal_id,
                    'time_minutes': t,
                    'prx': prx,
                    'phase': 'pre'
                })
        
        if prx_post is not None:
            # Post-SAH time starts at minute 15 (after 15-min stabilization)
            time_post = 15 + (timestamps_post / 60)  # Minutes after SAH start
            for i, (t, prx) in enumerate(zip(time_post, prx_post)):
                all_prx_data.append({
                    'animal_id': animal_id,
                    'time_minutes': t,
                    'prx': prx,
                    'phase': 'post'
                })
                
    except Exception as e:
        print(f"Error processing {animal_id}: {e}")

# Convert to DataFrame
prx_timeseries_df = pd.DataFrame(all_prx_data)

print(f"Collected PRx data from {prx_timeseries_df['animal_id'].nunique()} animals")
print(f"Time range: {prx_timeseries_df['time_minutes'].min():.1f} to {prx_timeseries_df['time_minutes'].max():.1f} minutes")
    

# %% Create Timeline Figure

# Create time bins for averaging
time_bins_pre = np.arange(-30, 0, 2)  # 2-minute bins from -30 to 0
time_bins_post = np.arange(15, 60, 2)  # 2-minute bins from 15 to 60

# Calculate mean and SEM for each time bin
def calculate_binned_stats(df, time_bins):
    binned_stats = []
    
    for i in range(len(time_bins)-1):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        bin_center = (bin_start + bin_end) / 2
        
        # Get data in this time bin
        mask = (df['time_minutes'] >= bin_start) & (df['time_minutes'] < bin_end)
        bin_data = df[mask]['prx']
        
        if len(bin_data) > 0:
            binned_stats.append({
                'time': bin_center,
                'mean': np.mean(bin_data),
                'std': np.std(bin_data),
                'sem': np.std(bin_data) / np.sqrt(len(bin_data)),
                'n': len(bin_data)
            })
    
    return pd.DataFrame(binned_stats)

# Calculate binned statistics
pre_stats = calculate_binned_stats(prx_timeseries_df[prx_timeseries_df['phase'] == 'pre'], time_bins_pre)
post_stats = calculate_binned_stats(prx_timeseries_df[prx_timeseries_df['phase'] == 'post'], time_bins_post)

# Create the timeline figure
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot pre-SAH data
ax.plot(pre_stats['time'], pre_stats['mean'], 'b-', linewidth=3, label='Pre-SAH', alpha=0.8)
ax.fill_between(pre_stats['time'], 
                pre_stats['mean'] - pre_stats['sem'], 
                pre_stats['mean'] + pre_stats['sem'], 
                alpha=0.3, color='blue')

# Plot post-SAH data  
ax.plot(post_stats['time'], post_stats['mean'], 'r-', linewidth=3, label='Post-SAH', alpha=0.8)
ax.fill_between(post_stats['time'], 
                post_stats['mean'] - post_stats['sem'], 
                post_stats['mean'] + post_stats['sem'], 
                alpha=0.3, color='red')

# Mark key events
ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
ax.axvspan(0, 15, alpha=0.2, color='gray', label='Stabilization Period\n(15 min)')

# Add event annotations
ax.annotate('SAH\nInduction', xy=(0, ax.get_ylim()[1]), xytext=(0, ax.get_ylim()[1]*1.1),
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.annotate('Cushing Response\n& Stabilization', xy=(7.5, ax.get_ylim()[0]), xytext=(7.5, ax.get_ylim()[0]*1.2),
            ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('PRx (Pressure Reactivity Index)', fontsize=14, fontweight='bold')
ax.set_title('Temporal Evolution of Cerebral Autoregulation\nDuring Experimental SAH', 
             fontsize=16, fontweight='bold', pad=20)

# Add reference line for normal autoregulation
ax.axhline(0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Normal Autoregulation')

# Set x-axis ticks and labels
x_ticks = list(range(-30, 0, 5)) + [0] + list(range(15, 61, 5))
ax.set_xticks(x_ticks)
ax.set_xlim(-32, 62)

# Add grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='upper right', fontsize=12)

# Add text boxes with interpretation
ax.text(0.02, 0.98, 'Lower values = Better autoregulation\nHigher values = Impaired autoregulation', 
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
        fontsize=10)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n" + "="*60)
print("TEMPORAL PRx ANALYSIS SUMMARY")
print("="*60)
print(f"Pre-SAH baseline PRx:  {pre_stats['mean'].mean():.3f} ± {pre_stats['mean'].std():.3f}")
print(f"Post-SAH PRx:          {post_stats['mean'].mean():.3f} ± {post_stats['mean'].std():.3f}")
print(f"Overall change:        {post_stats['mean'].mean() - pre_stats['mean'].mean():+.3f}")
print(f"Animals analyzed:      {prx_timeseries_df['animal_id'].nunique()}")
print("="*60)
    
# %% Create Improved Timeline PRx Figure

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot pre-SAH data
ax.plot(pre_stats['time'], pre_stats['mean'], 'b-', linewidth=3, label='Pre-SAH', alpha=0.8)
ax.fill_between(pre_stats['time'], 
                pre_stats['mean'] - pre_stats['sem'], 
                pre_stats['mean'] + pre_stats['sem'], 
                alpha=0.3, color='blue')

# Plot post-SAH data  
ax.plot(post_stats['time'], post_stats['mean'], 'r-', linewidth=3, label='Post-SAH', alpha=0.8)
ax.fill_between(post_stats['time'], 
                post_stats['mean'] - post_stats['sem'], 
                post_stats['mean'] + post_stats['sem'], 
                alpha=0.3, color='red')

# Mark key events
ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')

# Much narrower stabilization period indicator
ax.axvspan(0, 2, alpha=0.4, color='gray', label='15 min Gap\n(Stabilization)')

# Add time discontinuity indicator
ax.text(7.5, ax.get_ylim()[0] + 0.02, '// 15 min gap //', 
        ha='center', va='bottom', fontsize=10, style='italic',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

# Add event annotation (positioned higher to avoid title overlap)
ax.annotate('SAH\nInduction', xy=(0, ax.get_ylim()[1]*0.9), xytext=(0, ax.get_ylim()[1]*0.95),
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('PRx (Pressure Reactivity Index)', fontsize=14, fontweight='bold')

# Title positioned much higher
ax.set_title('Temporal Evolution of Cerebral Autoregulation During Experimental SAH', 
             fontsize=16, fontweight='bold', pad=40)

# Add reference line for normal autoregulation
ax.axhline(0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Normal Autoregulation')

# Set x-axis ticks and labels
x_ticks = list(range(-30, 0, 5)) + [0] + list(range(15, 61, 5))
ax.set_xticks(x_ticks)
ax.set_xlim(-32, 62)

# Add grid
ax.grid(True, alpha=0.3)

# Legend moved to upper left to avoid overlap
ax.legend(loc='upper left', fontsize=12)

# Add interpretation text box (moved to avoid overlap)
ax.text(0.02, 0.15, 'Lower values = Better autoregulation\nHigher values = Impaired autoregulation', 
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
        fontsize=10)

# Add subtle time break indicators around the gap
ax.plot([12, 14], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k-', linewidth=2, alpha=0.5)
ax.plot([12.5, 13.5], [ax.get_ylim()[0] + 0.01, ax.get_ylim()[0] + 0.01], 'k-', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.show()

print(f"\n" + "="*60)
print("IMPROVED TEMPORAL PRx ANALYSIS")
print("="*60)
print(f"Pre-SAH baseline PRx:  {pre_stats['mean'].mean():.3f} ± {pre_stats['mean'].std():.3f}")
print(f"Post-SAH PRx:          {post_stats['mean'].mean():.3f} ± {post_stats['mean'].std():.3f}")
print(f"Overall change:        {post_stats['mean'].mean() - pre_stats['mean'].mean():+.3f}")
print(f"Animals analyzed:      {prx_timeseries_df['animal_id'].nunique()}")
print("="*60)


# %% Create Timeline Figure with Broken X-Axis

import matplotlib.pyplot as plt
import numpy as np

# Transform time data to compressed scale
def transform_time(time_minutes):
    """Transform time to compressed scale where gap 0-15 is compressed"""
    transformed = np.copy(time_minutes)
    
    # Pre-SAH times stay the same (negative values)
    pre_mask = time_minutes < 0
    transformed[pre_mask] = time_minutes[pre_mask]
    
    # Time 0 stays at 0
    zero_mask = time_minutes == 0
    transformed[zero_mask] = 0
    
    # Post-SAH times (15+) get shifted down by 13 units (compress 15-min gap to 2 units)
    post_mask = time_minutes >= 15
    transformed[post_mask] = time_minutes[post_mask] - 13  # 15 becomes 2, 60 becomes 47
    
    return transformed

# Transform the time data
pre_stats_transformed = pre_stats.copy()
pre_stats_transformed['time'] = transform_time(pre_stats['time'])

post_stats_transformed = post_stats.copy()
post_stats_transformed['time'] = transform_time(post_stats['time'])

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot pre-SAH data
ax.plot(pre_stats_transformed['time'], pre_stats_transformed['mean'], 'b-', linewidth=3, label='Pre-SAH', alpha=0.8)
ax.fill_between(pre_stats_transformed['time'], 
                pre_stats_transformed['mean'] - pre_stats_transformed['sem'], 
                pre_stats_transformed['mean'] + pre_stats_transformed['sem'], 
                alpha=0.3, color='blue')

# Plot post-SAH data  
ax.plot(post_stats_transformed['time'], post_stats_transformed['mean'], 'r-', linewidth=3, label='Post-SAH', alpha=0.8)
ax.fill_between(post_stats_transformed['time'], 
                post_stats_transformed['mean'] - post_stats_transformed['sem'], 
                post_stats_transformed['mean'] + post_stats_transformed['sem'], 
                alpha=0.3, color='red')

# Mark SAH induction
ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')

# Add break indicator in the gap
ax.axvspan(0, 2, alpha=0.3, color='lightgray')

# Add break symbols
break_x = 1  # Middle of the compressed gap
y_range = ax.get_ylim()
y_mid = (y_range[1] + y_range[0]) / 2

# Draw break symbol (zigzag lines)
ax.plot([0.8, 1.2], [y_mid - 0.02, y_mid + 0.02], 'k-', linewidth=2)
ax.plot([0.8, 1.2], [y_mid + 0.02, y_mid - 0.02], 'k-', linewidth=2)

# Add SAH annotation
ax.annotate('SAH\nInduction', xy=(0, y_range[1]*0.85), xytext=(0, y_range[1]*0.9),
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Create custom x-axis labels
# Pre-SAH labels (normal spacing)
pre_ticks = list(range(-30, 1, 5))  # -30, -25, -20, -15, -10, -5, 0
pre_labels = [str(x) for x in pre_ticks]

# Post-SAH labels (transformed spacing)
post_times_real = list(range(15, 61, 5))  # Real times: 15, 20, 25, ..., 60
post_times_transformed = [transform_time(np.array([t]))[0] for t in post_times_real]
post_labels = [str(t) for t in post_times_real]  # Keep original labels

# Combine ticks and labels
all_ticks = pre_ticks + post_times_transformed
all_labels = pre_labels + post_labels

ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels)

# Set axis limits
ax.set_xlim(-32, 49)  # Adjusted for compressed scale

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('PRx (Pressure Reactivity Index)', fontsize=14, fontweight='bold')
ax.set_title('Temporal Evolution of Cerebral Autoregulation During Experimental SAH', 
             fontsize=16, fontweight='bold', pad=40)

# Add reference line for normal autoregulation
ax.axhline(0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Normal Autoregulation')

# Add grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='upper left', fontsize=12)

# Add interpretation text
ax.text(0.02, 0.15, 'Lower values = Better autoregulation\nHigher values = Impaired autoregulation', 
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
        fontsize=10)

# Add note about time break
ax.text(1, y_range[0] - 0.05, '15 min gap\n(compressed)', 
        ha='center', va='top', fontsize=9, style='italic',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

print(f"\n" + "="*60)
print("COMPRESSED TIMELINE PRx ANALYSIS")
print("="*60)
print(f"X-axis compression: 15-minute gap compressed to 2 units")
print(f"Pre-SAH baseline PRx:  {pre_stats['mean'].mean():.3f} ± {pre_stats['mean'].std():.3f}")
print(f"Post-SAH PRx:          {post_stats['mean'].mean():.3f} ± {post_stats['mean'].std():.3f}")
print(f"Overall change:        {post_stats['mean'].mean() - pre_stats['mean'].mean():+.3f}")
print("="*60)


# %% Create Final Timeline Figure with Improved Layout

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot pre-SAH data
ax.plot(pre_stats_transformed['time'], pre_stats_transformed['mean'], 'b-', linewidth=3, label='Pre-SAH', alpha=0.8)
ax.fill_between(pre_stats_transformed['time'], 
                pre_stats_transformed['mean'] - pre_stats_transformed['sem'], 
                pre_stats_transformed['mean'] + pre_stats_transformed['sem'], 
                alpha=0.3, color='blue')

# Plot post-SAH data  
ax.plot(post_stats_transformed['time'], post_stats_transformed['mean'], 'r-', linewidth=3, label='Post-SAH', alpha=0.8)
ax.fill_between(post_stats_transformed['time'], 
                post_stats_transformed['mean'] - post_stats_transformed['sem'], 
                post_stats_transformed['mean'] + post_stats_transformed['sem'], 
                alpha=0.3, color='red')

# Mark SAH induction
ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')

# Add break indicator in the gap
ax.axvspan(0, 2, alpha=0.3, color='lightgray')

# Add break symbols
break_x = 1  # Middle of the compressed gap
y_range = ax.get_ylim()
y_mid = (y_range[1] + y_range[0]) / 2

# Draw break symbol (zigzag lines)
ax.plot([0.8, 1.2], [y_mid - 0.02, y_mid + 0.02], 'k-', linewidth=2)
ax.plot([0.8, 1.2], [y_mid + 0.02, y_mid - 0.02], 'k-', linewidth=2)

# SAH annotation positioned HIGHER (increased from 0.85 to 0.95)
ax.annotate('SAH\nInduction', xy=(0, y_range[1]*0.9), xytext=(0, y_range[1]*1.15),
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Create custom x-axis labels
pre_ticks = list(range(-30, 1, 5))  # -30, -25, -20, -15, -10, -5, 0
pre_labels = [str(x) for x in pre_ticks]

post_times_real = list(range(15, 61, 5))  # Real times: 15, 20, 25, ..., 60
post_times_transformed = [transform_time(np.array([t]))[0] for t in post_times_real]
post_labels = [str(t) for t in post_times_real]  # Keep original labels

all_ticks = pre_ticks + post_times_transformed
all_labels = pre_labels + post_labels

ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels)
ax.set_xlim(-32, 49)

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('PRx (Pressure Reactivity Index)', fontsize=14, fontweight='bold')

# New title as requested
ax.set_title('Pressure Reactivity Index Over Time Before and After SAH Induction', 
             fontsize=16, fontweight='bold', pad=70)

# Add reference line for normal autoregulation
ax.axhline(0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Normal Autoregulation')

# Add grid
ax.grid(True, alpha=0.3)

# Legend moved to lower left corner
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

# Add note about time break
ax.text(1, y_range[0] - 0.05, '15 min gap\n(compressed)', 
        ha='center', va='top', fontsize=9, style='italic',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

print(f"\n" + "="*60)
print("PRESSURE REACTIVITY INDEX TIMELINE ANALYSIS")
print("="*60)
print(f"Pre-SAH baseline PRx:  {pre_stats['mean'].mean():.3f} ± {pre_stats['mean'].std():.3f}")
print(f"Post-SAH PRx:          {post_stats['mean'].mean():.3f} ± {post_stats['mean'].std():.3f}")
print(f"Overall change:        {post_stats['mean'].mean() - pre_stats['mean'].mean():+.3f}")
print(f"Animals analyzed:      {prx_timeseries_df['animal_id'].nunique()}")
print("="*60)


# %% Create Timeline Figure with Standard Deviation

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot pre-SAH data with SD bands
ax.plot(pre_stats_transformed['time'], pre_stats_transformed['mean'], 'b-', linewidth=3, label='Pre-SAH', alpha=0.8)
ax.fill_between(pre_stats_transformed['time'], 
                pre_stats_transformed['mean'] - pre_stats_transformed['std'], 
                pre_stats_transformed['mean'] + pre_stats_transformed['std'], 
                alpha=0.3, color='blue')

# Plot post-SAH data with SD bands
ax.plot(post_stats_transformed['time'], post_stats_transformed['mean'], 'r-', linewidth=3, label='Post-SAH', alpha=0.8)
ax.fill_between(post_stats_transformed['time'], 
                post_stats_transformed['mean'] - post_stats_transformed['std'], 
                post_stats_transformed['mean'] + post_stats_transformed['std'], 
                alpha=0.3, color='red')

# Mark SAH induction
ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')

# Add break indicator and symbols
ax.axvspan(0, 2, alpha=0.3, color='lightgray')
y_range = ax.get_ylim()
y_mid = (y_range[1] + y_range[0]) / 2
ax.plot([0.8, 1.2], [y_mid - 0.02, y_mid + 0.02], 'k-', linewidth=2)
ax.plot([0.8, 1.2], [y_mid + 0.02, y_mid - 0.02], 'k-', linewidth=2)

# SAH annotation
ax.annotate('SAH\nInduction', xy=(0, y_range[1]*0.9), xytext=(0, y_range[1]*1.05),
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# X-axis setup
pre_ticks = list(range(-30, 1, 5))
pre_labels = [str(x) for x in pre_ticks]
post_times_real = list(range(15, 61, 5))
post_times_transformed = [transform_time(np.array([t]))[0] for t in post_times_real]
post_labels = [str(t) for t in post_times_real]
all_ticks = pre_ticks + post_times_transformed
all_labels = pre_labels + post_labels

ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels)
ax.set_xlim(-32, 49)

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('PRx (Pressure Reactivity Index)', fontsize=14, fontweight='bold')
ax.set_title('Pressure Reactivity Index Over Time Before and After SAH Induction', 
             fontsize=16, fontweight='bold', pad=70)

# Reference line and grid
ax.axhline(0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Normal Autoregulation')
ax.grid(True, alpha=0.3)

# Legend with note about error bands
ax.legend(loc='lower left', fontsize=12, framealpha=0.9, 
          title='Shaded areas = ±1 SD\n(individual animal variability)')

plt.tight_layout()
plt.show()

print(f"\nUsing Standard Deviation to show individual animal variability")









































