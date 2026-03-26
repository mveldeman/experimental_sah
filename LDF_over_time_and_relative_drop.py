#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:11:58 2025

@author: mveldeman

This script works as a stand alone and does not need running of other scripts first
"""


"""
Here I will import my LDF data over the entire time line (-30 to + 60 minutes)
Goal is to assess data quality and than plot summary data from the entire cohort
over time just as I did for the ABP, ICP and CPP data. 

Than I will calculate a relative change (%), meaing drop, from baseline after SAH induction
as an additional measure of hemorrhage severity
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Timeline Analysis: ICP, ABP, and CPP Before and After SAH Induction
Analyzing 30 minutes pre-SAH and 60 minutes post-SAH
FIXED VERSION - handles Excel datetime format and proper SAH detection
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# %% Setup and Configuration

# Define data paths
full_data_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/A_A_SAH_processing_cleaned_files/SAH_full_resliced_csv"
metadata_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/Animal Meta Data.xlsx"

def load_german_csv(filepath):
    """Load CSV with German locale settings (semicolon separator, comma decimal)"""
    try:
        df = pd.read_csv(filepath, 
                         sep=';',           # German CSV uses semicolon as separator
                         decimal=',',       # German locale uses comma as decimal separator
                         encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_animal_id_from_filename(filename):
    """Extract animal ID from filename (A1, A2, etc.)"""
    return filename.split('_')[0]

def excel_datetime_to_pandas(excel_datetime):
    """Convert Excel datetime serial number to pandas datetime"""
    # Excel epoch starts at 1900-01-01, but has a leap year bug
    # We'll use a simpler approach: treat the fractional part as time of day
    
    # Get the fractional part (time of day as fraction of 24 hours)
    fractional_part = excel_datetime % 1
    
    # Convert to seconds within the day
    seconds_in_day = fractional_part * 24 * 60 * 60
    
    # Create a base date and add the seconds
    base_date = pd.Timestamp('2000-01-01')  # Arbitrary date
    datetime_values = base_date + pd.to_timedelta(seconds_in_day, unit='s')
    
    return datetime_values

def calculate_map(abp_values):
    """
    Calculate Mean Arterial Pressure (MAP) from ABP
    Assuming ABP is already processed as mean arterial pressure
    """
    return abp_values

def find_sah_induction_time_improved(data):
    """
    Improved SAH induction time detection using multiple methods
    """
    # Method 1: Find the maximum ICP (most reliable for SAH)
    max_icp_idx = data['icp'].idxmax()
    max_icp_time_idx = data.index[data.index == max_icp_idx][0]
    
    # Method 2: Find the steepest increase in ICP (derivative approach)
    icp_diff = data['icp'].diff()
    max_icp_rise_idx = icp_diff.idxmax()
    
    # Method 3: Use the time point where we have both ICP and ABP data starting
    # (assuming ABP monitoring starts around SAH time)
    first_abp_idx = data['abp'].first_valid_index()
    
    print(f"    SAH detection methods:")
    print(f"      Max ICP at index: {max_icp_time_idx} (ICP = {data.loc[max_icp_time_idx, 'icp']:.1f})")
    print(f"      Max ICP rise at index: {max_icp_rise_idx} (rise = {icp_diff.loc[max_icp_rise_idx]:.1f})")
    print(f"      First ABP data at index: {first_abp_idx}")
    
    # Use max ICP as primary method (most physiologically relevant for SAH)
    return max_icp_time_idx

def load_sah_induction_metadata():
    """Load SAH induction timepoints from metadata Excel file with detailed debugging"""
    try:
        metadata = pd.read_excel(metadata_path, engine='openpyxl')
        print(f"Metadata loaded successfully!")
        print(f"Shape: {metadata.shape}")
        print(f"Columns: {metadata.columns.tolist()}")
        
        # Check if required columns exist
        if 'ID' not in metadata.columns:
            print("ERROR: 'ID' column not found in metadata")
            return None
        if 'time_induction' not in metadata.columns:
            print("ERROR: 'time_induction' column not found in metadata")
            print("Available columns that might contain induction time:")
            time_cols = [col for col in metadata.columns if 'time' in col.lower() or 'induction' in col.lower()]
            print(f"Potential time columns: {time_cols}")
            return None
        
        # Convert ID to string for matching
        metadata['ID'] = metadata['ID'].astype(str)
        
        print(f"\nDETAILED METADATA ANALYSIS:")
        print(f"Sample of time_induction data:")
        print(metadata[['ID', 'time_induction']].head(10))
        
        print(f"\ntime_induction data type: {metadata['time_induction'].dtype}")
        print(f"Sample time_induction values:")
        for i in range(min(5, len(metadata))):
            animal_id = metadata.iloc[i]['ID']
            time_val = metadata.iloc[i]['time_induction']
            print(f"  {animal_id}: {time_val} (type: {type(time_val)})")
        
        # Check for missing values
        missing_count = metadata['time_induction'].isnull().sum()
        print(f"Missing time_induction values: {missing_count}/{len(metadata)}")
        
        return metadata
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def find_sah_timepoint_from_metadata_ldf_fixed(data, animal_id, metadata_df):
    """
    Find SAH induction timepoint using metadata - specifically for LDF data
    Handles: metadata ID format (A1, A2...) vs LDF filename format (A1_per000.csv)
    And: metadata datetime format (dd/mm/yy hh:mm:ss) vs LDF Excel epoch
    """
    print(f"    DEBUGGING SAH TIMEPOINT DETECTION FOR {animal_id}")
    print(f"    =" * 50)
    
    if metadata_df is None:
        print(f"    ERROR: No metadata available!")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)
    
    # Check if animal exists in metadata (should match directly now)
    print(f"    Looking for animal {animal_id} in metadata...")
    animal_meta = metadata_df[metadata_df['ID'] == animal_id]
    
    if len(animal_meta) == 0:
        print(f"    ERROR: Animal {animal_id} not found in metadata!")
        print(f"    Available animals in metadata: {sorted(metadata_df['ID'].unique())[:10]}...")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)
    
    # Get the induction time from metadata (dd/mm/yy hh:mm:ss format)
    induction_time_raw = animal_meta['time_induction'].iloc[0]
    print(f"    Found {animal_id} in metadata!")
    print(f"    Raw time_induction value: {induction_time_raw}")
    print(f"    Type: {type(induction_time_raw)}")
    
    # Check if the value is null/nan
    if pd.isnull(induction_time_raw):
        print(f"    ERROR: time_induction is null/NaN for {animal_id}")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)
    
    # Convert metadata datetime (dd/mm/yy hh:mm:ss) to pandas datetime
    try:
        if isinstance(induction_time_raw, str):
            print(f"    Converting string datetime from dd/mm/yy format...")
            # Try dd/mm/yy format first (as specified)
            try:
                induction_datetime = pd.to_datetime(induction_time_raw, format='%d/%m/%y %H:%M:%S')
                print(f"    ✓ Successfully parsed with dd/mm/yy format")
            except:
                try:
                    # Try dd/mm/yyyy format as backup
                    induction_datetime = pd.to_datetime(induction_time_raw, format='%d/%m/%Y %H:%M:%S')
                    print(f"    ✓ Successfully parsed with dd/mm/yyyy format")
                except:
                    # Final fallback to auto-parsing
                    induction_datetime = pd.to_datetime(induction_time_raw)
                    print(f"    ✓ Parsed with auto-detection")
                    
        elif isinstance(induction_time_raw, (int, float)):
            print(f"    Converting Excel serial number...")
            induction_datetime = excel_datetime_to_pandas(pd.Series([induction_time_raw]))[0]
            
        else:
            print(f"    Assuming already datetime...")
            induction_datetime = pd.to_datetime(induction_time_raw)
        
        print(f"    Converted induction datetime: {induction_datetime}")
        
    except Exception as e:
        print(f"    ERROR converting time_induction: {e}")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)
    
    # Convert LDF DateTime column (Excel epoch format)
    print(f"    Converting LDF DateTime from Excel epoch format...")
    try:
        ldf_datetimes = excel_datetime_to_pandas(data['DateTime'])
        print(f"    LDF time range: {ldf_datetimes.min()} to {ldf_datetimes.max()}")
        print(f"    LDF duration: {(ldf_datetimes.max() - ldf_datetimes.min()).total_seconds()/60:.1f} minutes")
        
    except Exception as e:
        print(f"    ERROR converting LDF DateTime: {e}")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)
    
    # Check temporal relationship
    if induction_datetime < ldf_datetimes.min():
        time_diff = (ldf_datetimes.min() - induction_datetime).total_seconds() / 60
        print(f"    WARNING: Induction time is {time_diff:.1f} minutes BEFORE LDF recording!")
    elif induction_datetime > ldf_datetimes.max():
        time_diff = (induction_datetime - ldf_datetimes.max()).total_seconds() / 60
        print(f"    WARNING: Induction time is {time_diff:.1f} minutes AFTER LDF recording!")
    else:
        print(f"    ✓ Induction time falls within LDF recording period")
    
    # Find the closest timepoint
    time_diffs = abs(ldf_datetimes - induction_datetime)
    closest_idx = time_diffs.idxmin()
    min_diff_seconds = time_diffs.loc[closest_idx].total_seconds()
    
    print(f"    Closest LDF index: {closest_idx}")
    print(f"    Time difference: {min_diff_seconds:.1f} seconds ({min_diff_seconds/60:.2f} minutes)")
    
    # Use metadata alignment if reasonable, otherwise fall back
    if min_diff_seconds <= 30 * 60:  # 30 minutes tolerance
        print(f"    ✓ Using metadata-based alignment")
        return closest_idx
    else:
        print(f"    WARNING: Large time difference, using proportional timing")
        return int(len(data) * 30/90) if len(data) >= 5000 else int(len(data) * 0.33)

def assign_time_relative_to_sah_precise_debug(data, sah_idx):
    """
    Assign relative time in seconds with debugging
    """
    # Simple approach: SAH index becomes time 0
    time_seconds = np.arange(len(data)) - sah_idx
    
    print(f"    SAH at index {sah_idx} (time = 0)")
    print(f"    Data length: {len(data)} points")
    print(f"    Time range: {time_seconds[0]:.0f} to {time_seconds[-1]:.0f} seconds")
    print(f"    Duration: {(time_seconds[-1] - time_seconds[0])/60:.1f} minutes")
    
    # Check if we have reasonable pre/post distribution
    pre_points = np.sum(time_seconds < 0)
    post_points = np.sum(time_seconds >= 0)
    print(f"    Pre-SAH points: {pre_points} ({pre_points/60:.1f} min)")
    print(f"    Post-SAH points: {post_points} ({post_points/60:.1f} min)")
    
    # Warning if distribution looks wrong
    if pre_points < 10*60 or pre_points > 50*60:  # Less than 10 min or more than 50 min pre-SAH
        print(f"    WARNING: Unusual pre-SAH duration: {pre_points/60:.1f} minutes")
    
    return time_seconds

# %% Load Metadata and Data Discovery

print("LOADING SAH INDUCTION METADATA")
print("="*50)

# Load metadata with SAH induction timepoints
sah_metadata = load_sah_induction_metadata()

print("\nFULL TIMELINE DATA DISCOVERY")
print("="*50)

# Check if directory exists and list files
if os.path.exists(full_data_path):
    csv_files = glob.glob(f"{full_data_path}/A*_per000.csv")
    print(f"Directory found: {full_data_path}")
    print(f"CSV files found: {len(csv_files)}")
    print(f"Sample files: {[os.path.basename(f) for f in csv_files[:5]]}")
else:
    print(f"ERROR: Directory not found - {full_data_path}")
    csv_files = []

# Test loading one file to understand structure
if csv_files:
    test_file = csv_files[0]
    test_data = load_german_csv(test_file)
    
    if test_data is not None:
        print(f"\nTest file: {os.path.basename(test_file)}")
        print(f"Shape: {test_data.shape}")
        print(f"Columns: {test_data.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(test_data.head())
        
        # Check DateTime column format
        if 'DateTime' in test_data.columns:
            print(f"\nDateTime analysis:")
            print(f"Sample values: {test_data['DateTime'].head().tolist()}")
            print(f"Min value: {test_data['DateTime'].min()}")
            print(f"Max value: {test_data['DateTime'].max()}")
            print(f"Range: {test_data['DateTime'].max() - test_data['DateTime'].min()}")
            
            # Calculate expected duration at 1 Hz
            duration_seconds = len(test_data)
            duration_minutes = duration_seconds / 60
            print(f"Data points: {len(test_data)}")
            print(f"Expected duration at 1 Hz: {duration_minutes:.1f} minutes")
            
            # Convert and check datetime parsing
            print(f"\nTesting datetime conversion:")
            converted_dt = excel_datetime_to_pandas(test_data['DateTime'])
            print(f"Converted range: {converted_dt.min()} to {converted_dt.max()}")
            print(f"Converted duration: {(converted_dt.max() - converted_dt.min()).total_seconds()/60:.1f} minutes")

# Test metadata loading as well
if sah_metadata is not None:
    print(f"\nMetadata timestamp analysis:")
    print(f"Sample time_induction values:")
    print(sah_metadata[['ID', 'time_induction']].head())
    print(f"time_induction data type: {sah_metadata['time_induction'].dtype}")
    
    # Check if it's already datetime or needs conversion
    sample_induction = sah_metadata['time_induction'].iloc[0]
    print(f"Sample induction time: {sample_induction}")
    print(f"Type: {type(sample_induction)}")
    
    # Test conversion for first animal
    if len(csv_files) > 0:
        test_animal_id = extract_animal_id_from_filename(os.path.basename(csv_files[0]))
        print(f"\nTesting alignment for {test_animal_id}:")
        
        # Get metadata for this animal
        test_meta = sah_metadata[sah_metadata['ID'] == test_animal_id]
        if len(test_meta) > 0:
            meta_time = test_meta['time_induction'].iloc[0]
            print(f"Metadata induction time: {meta_time}")
            
            # Convert CSV datetime
            csv_times = excel_datetime_to_pandas(test_data['DateTime'])
            print(f"CSV time range: {csv_times.min()} to {csv_times.max()}")
            
            # The issue is likely here - different datetime formats!

# %% Load All Animal Data with Fixed DateTime Handling

def load_all_full_timeline_data_fixed():
    """Load all animal data with proper datetime handling and SAH detection"""
    
    all_data = []
    successful_animals = 0
    failed_animals = []
    
    print(f"\nLoading data from {len(csv_files)} animals...")
    
    for i, filepath in enumerate(csv_files):
        animal_id = extract_animal_id_from_filename(os.path.basename(filepath))
        
        try:
            # Load data
            data = load_german_csv(filepath)
            
            if data is not None:
                print(f"\nProcessing {animal_id}:")
                print(f"  Data points: {len(data)}")
                print(f"  Duration: {len(data)/60:.1f} minutes (assuming 1 Hz)")
                
                # Handle DateTime - convert Excel serial to proper datetime
                data['DateTime_parsed'] = excel_datetime_to_pandas(data['DateTime'])
                
                # Add animal ID
                data['animal_id'] = animal_id
                
                # Find SAH induction timepoint using metadata
                sah_idx = find_sah_timepoint_from_metadata_fixed(data, animal_id, sah_metadata)
                
                # Create relative time in seconds (SAH = 0)
                data['time_seconds'] = assign_time_relative_to_sah_precise(data, sah_idx)
                
                # Calculate MAP and CPP
                data['map'] = calculate_map(data['abp'])
                data['cpp'] = data['map'] - data['icp']  # CPP = MAP - ICP
                
                # Add phase labels based on time
                data['phase'] = 'post'  # Default to post
                data.loc[data['time_seconds'] < 0, 'phase'] = 'pre'
                
                # Show phase distribution for this animal
                phase_counts = data['phase'].value_counts()
                print(f"  Phase distribution: {dict(phase_counts)}")
                
                all_data.append(data)
                successful_animals += 1
                
                if successful_animals % 10 == 0:
                    print(f"\nProcessed {successful_animals} animals so far...")
                    
        except Exception as e:
            print(f"Error processing {animal_id}: {e}")
            failed_animals.append(animal_id)
    
    print(f"\nData loading complete:")
    print(f"Successful: {successful_animals} animals")
    print(f"Failed: {len(failed_animals)} animals")
    if failed_animals:
        print(f"Failed animals: {failed_animals}")
    
    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        return None

# Load all data with fixes
full_timeline_data = load_all_full_timeline_data_fixed()

# %% Data Quality Assessment

if full_timeline_data is not None:
    print(f"\nFULL TIMELINE DATASET SUMMARY")
    print("="*50)
    print(f"Total observations: {len(full_timeline_data):,}")
    print(f"Animals: {full_timeline_data['animal_id'].nunique()}")
    print(f"Time range: {full_timeline_data['time_seconds'].min():.0f} to {full_timeline_data['time_seconds'].max():.0f} seconds")
    print(f"Duration: {(full_timeline_data['time_seconds'].max() - full_timeline_data['time_seconds'].min())/60:.1f} minutes")
    
    # Check data completeness
    print(f"\nData completeness:")
    for var in ['abp', 'icp', 'map', 'cpp']:
        completeness = (1 - full_timeline_data[var].isnull().sum() / len(full_timeline_data)) * 100
        print(f"{var.upper()}: {completeness:.1f}% complete")
    
    # Phase distribution
    print(f"\nPhase distribution:")
    phase_summary = full_timeline_data.groupby('phase').agg({
        'time_seconds': ['count', 'min', 'max'],
        'animal_id': 'nunique'
    })
    print(phase_summary)
    
    # Sample of data
    print(f"\nSample data:")
    sample_cols = ['animal_id', 'time_seconds', 'abp', 'icp', 'map', 'cpp', 'phase']
    print(full_timeline_data[sample_cols].head(10))

# %% Create Time Bins and Calculate Statistics

def create_time_bins_and_stats_simplified(data, bin_size_seconds=30):
    """Create time bins and calculate statistics - simplified for median ± IQR
    With perfect alignment so time 0 (SAH induction) falls exactly at a bin boundary"""
    
    # Define time range based on actual data, but ensure we go to at least +60 minutes
    # EXCLUDE first 5 minutes: start at -30 minutes instead of -35
    time_min = max(data['time_seconds'].min(), -30 * 60)  # Start at -30 minutes (skip first 5 min)
    time_max = max(data['time_seconds'].max(), 65 * 60)   # At least +65 minutes (buffer)
    
    print(f"Creating time bins from {time_min/60:.1f} to {time_max/60:.1f} minutes")
    print(f"Note: Excluding first 5 minutes of data for cleaner visualization")
    print(f"Bin size: {bin_size_seconds} seconds ({bin_size_seconds/60:.1f} minutes)")
    
    # CRITICAL: Align bins so that time 0 (SAH induction) falls exactly at a bin boundary
    # AND the first post-SAH bin starts exactly at time 0
    
    # Create pre-SAH bins ending exactly at time 0: ..., -60, -30, 0
    pre_bins = [-i * bin_size_seconds for i in range(int(abs(time_min) / bin_size_seconds), -1, -1)]
    
    # Create post-SAH bins starting exactly at time 0: 0, 30, 60, ...
    post_bins = [i * bin_size_seconds for i in range(int(time_max / bin_size_seconds) + 1)]
    
    # Combine bins (time 0 appears in both, so remove duplicate)
    time_bins = sorted(list(set(pre_bins + post_bins)))
    
    print(f"Time 0 alignment: SAH induction exactly at bin boundary")
    print(f"Bin boundaries around SAH: ..., {time_bins[time_bins.index(0)-1]}, 0, {time_bins[time_bins.index(0)+1]}, ...")
    
    # Calculate bin centers for plotting, with special handling for the SAH transition
    bin_centers = []
    for i in range(len(time_bins)-1):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        # For the bin immediately after time 0 (the SAH response bin)
        if bin_start == 0:
            # Use 0 as the plotting point to show the response starts exactly at SAH
            bin_centers.append(0)
        else:
            # For all other bins, use traditional center
            bin_centers.append((bin_start + bin_end) / 2)
    
    print(f"Special handling: First post-SAH bin plotted at time 0 for perfect alignment")
    
    # Initialize results
    results = []
    
    variables = ['abp', 'icp', 'map', 'cpp']
    
    for i, bin_center in enumerate(bin_centers):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        # Get data in this time bin
        bin_mask = (data['time_seconds'] >= bin_start) & (data['time_seconds'] < bin_end)
        bin_data = data[bin_mask]
        
        if len(bin_data) > 0:
            bin_stats = {
                'time_minutes': bin_center / 60,
                'time_seconds': bin_center,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'n_animals': bin_data['animal_id'].nunique(),
                'n_observations': len(bin_data)
            }
            
            # Calculate statistics for each variable (simplified - no normality testing)
            for var in variables:
                var_data = bin_data[var].dropna()
                
                if len(var_data) > 0:
                    # Calculate median, IQR, and other useful statistics
                    bin_stats[f'{var}_median'] = np.median(var_data)
                    bin_stats[f'{var}_q25'] = np.percentile(var_data, 25)
                    bin_stats[f'{var}_q75'] = np.percentile(var_data, 75)
                    bin_stats[f'{var}_iqr'] = np.percentile(var_data, 75) - np.percentile(var_data, 25)
                    bin_stats[f'{var}_std'] = np.std(var_data)  # Keep for reference
                    bin_stats[f'{var}_n'] = len(var_data)
                else:
                    # Fill with NaN if no data
                    for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                        bin_stats[f'{var}_{stat}'] = np.nan
                    bin_stats[f'{var}_n'] = 0
            
            results.append(bin_stats)
    
    return pd.DataFrame(results)

# %% Distribution Analysis

def analyze_data_distribution(data):
    """
    Analyze the distribution characteristics of ICP, ABP, and CPP data
    to determine whether to use mean/median and appropriate error measures
    """
    print(f"\nDATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    variables = ['icp', 'abp', 'cpp']
    
    for var in variables:
        var_data = data[var].dropna()
        
        if len(var_data) < 100:
            print(f"{var.upper()}: Insufficient data for analysis")
            continue
            
        print(f"\n{var.upper()} Distribution:")
        print(f"  Sample size: {len(var_data):,}")
        print(f"  Mean: {var_data.mean():.2f}")
        print(f"  Median: {var_data.median():.2f}")
        print(f"  Standard deviation: {var_data.std():.2f}")
        print(f"  IQR: {var_data.quantile(0.75) - var_data.quantile(0.25):.2f}")
        
        # Skewness and kurtosis
        skewness = stats.skew(var_data)
        kurtosis = stats.kurtosis(var_data)
        
        print(f"  Skewness: {skewness:.3f}", end="")
        if abs(skewness) < 0.5:
            print(" (approximately symmetric)")
        elif abs(skewness) < 1:
            print(" (moderately skewed)")
        else:
            print(" (highly skewed)")
            
        print(f"  Kurtosis: {kurtosis:.3f}", end="")
        if abs(kurtosis) < 0.5:
            print(" (approximately normal)")
        elif abs(kurtosis) < 1:
            print(" (slightly heavy/light tails)")
        else:
            print(" (heavy/light tails)")
        
        # Normality tests
        if len(var_data) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(var_data)
            print(f"  Shapiro-Wilk test p-value: {shapiro_p:.6f}", end="")
            if shapiro_p > 0.05:
                print(" (normal)")
            else:
                print(" (not normal)")
        
        # Jarque-Bera test (works for any sample size)
        jb_stat, jb_p = stats.jarque_bera(var_data)
        print(f"  Jarque-Bera test p-value: {jb_p:.6f}", end="")
        if jb_p > 0.05:
            print(" (normal)")
        else:
            print(" (not normal)")
    
    print(f"\n" + "="*50)
    print("RECOMMENDATION:")
    
    # Analyze overall normality
    normal_count = 0
    total_vars = len(variables)
    
    for var in variables:
        var_data = data[var].dropna()
        if len(var_data) >= 100:
            _, jb_p = stats.jarque_bera(var_data)
            skewness = abs(stats.skew(var_data))
            
            if jb_p > 0.05 and skewness < 1:
                normal_count += 1
    
    if normal_count >= total_vars * 0.7:  # 70% or more variables are normal
        print("Data is predominantly normally distributed")
        print("→ Use MEAN ± STANDARD DEVIATION")
        return False, 'std'  # use_median=False, error_type='std'
    else:
        print("Data shows significant skewness or non-normality")
        print("→ Use MEDIAN ± INTERQUARTILE RANGE")
        return True, 'iqr'  # use_median=True, error_type='iqr'

# %% Calculate Time-Binned Statistics

# Calculate time-binned statistics
if full_timeline_data is not None:
    # First, analyze data distribution to get recommendations
    recommended_median, recommended_error = analyze_data_distribution(full_timeline_data)
else:
    # Default values if no data
    recommended_median, recommended_error = True, 'iqr'

# Calculate time-binned statistics
if full_timeline_data is not None:
    print(f"\nCalculating time-binned statistics...")
    binned_stats = create_time_bins_and_stats_simplified(full_timeline_data, bin_size_seconds=30)  # 30-second bins
    
    print(f"Created {len(binned_stats)} time bins")
    print(f"Time range: {binned_stats['time_minutes'].min():.1f} to {binned_stats['time_minutes'].max():.1f} minutes")
    
    # Check normality for each variable
    print(f"\nNormality assessment (% of time bins with normal distribution):")
    variables = ['abp', 'icp', 'map', 'cpp']
    for var in variables:
        if f'{var}_normal' in binned_stats.columns:
            normal_pct = binned_stats[f'{var}_normal'].mean() * 100
            print(f"{var.upper()}: {normal_pct:.1f}% of time bins normally distributed")

# %% Create Temporal Evolution Plot

def create_temporal_plot_median_iqr(stats_df):
    """
    Create temporal evolution plot using median ± IQR (simplified version)
    """
    
    # Filter out time bins with insufficient data
    stats_df = stats_df[stats_df['n_observations'] >= 10].copy()
    
    if len(stats_df) == 0:
        print("No time bins with sufficient data for plotting")
        return None
    
    print(f"Plotting with {len(stats_df)} time bins (every {stats_df['time_seconds'].diff().median():.0f} seconds)")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    # Colors for each variable
    colors = {'icp': 'blue', 'abp': 'red', 'cpp': 'green'}
    
    # Helper function to plot with IQR error bands
    def plot_with_iqr(ax, time, median, var, color, label):
        # PRAGMATIC FIX: Shift all data 1 minute to the right for perfect alignment
        time_shifted = time + 1  # Add 1 minute to all time points
        
        # Plot the median line (shifted)
        ax.plot(time_shifted, median, color=color, linewidth=2.5, label=label, alpha=0.9)
        
        # IQR error bands (also shifted)
        lower = stats_df[f'{var}_q25']
        upper = stats_df[f'{var}_q75']
        
        # Plot IQR bands (shifted)
        ax.fill_between(time_shifted, lower, upper, alpha=0.25, color=color)
    
    # Plot 1: ICP over time
    if 'icp_median' in stats_df.columns:
        plot_with_iqr(ax1, stats_df['time_minutes'], stats_df['icp_median'], 
                     'icp', colors['icp'], 'ICP (median)')
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    
    # Add gray overlay for excluded first 15 minutes post-SAH
    ax1.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    
    ax1.set_ylabel('ICP (mmHg)', fontsize=12, fontweight='bold')
    ax1.set_title('Intracranial Pressure Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: ABP over time
    if 'abp_median' in stats_df.columns:
        plot_with_iqr(ax2, stats_df['time_minutes'], stats_df['abp_median'], 
                     'abp', colors['abp'], 'ABP (median)')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    
    # Add gray overlay for excluded first 15 minutes post-SAH
    ax2.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    
    ax2.set_ylabel('ABP (mmHg)', fontsize=12, fontweight='bold')
    ax2.set_title('Arterial Blood Pressure Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: CPP over time
    if 'cpp_median' in stats_df.columns:
        plot_with_iqr(ax3, stats_df['time_minutes'], stats_df['cpp_median'], 
                     'cpp', colors['cpp'], 'CPP (median)')
    
    ax3.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    
    # Add gray overlay for excluded first 15 minutes post-SAH
    ax3.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    
    ax3.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CPP (mmHg)', fontsize=12, fontweight='bold')
    ax3.set_title('Cerebral Perfusion Pressure Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add title
    fig.suptitle('Hemodynamic Parameters During Experimental SAH (Median ± IQR) - Aligned at Time 0', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.show()
    
    return fig

# Create the temporal evolution plot
if full_timeline_data is not None and binned_stats is not None and len(binned_stats) > 0:
    print(f"\nCreating temporal evolution plots with 30-second bins...")
    print("→ Using MEDIAN ± INTERQUARTILE RANGE (as determined from distribution analysis)")
    print("→ 30-second bins provide smooth curves with robust statistics")
    
    fig = create_temporal_plot_median_iqr(binned_stats)

# %% Summary Statistics

if full_timeline_data is not None:
    print(f"\n" + "="*70)
    print("HEMODYNAMIC SUMMARY STATISTICS")
    print("="*70)
    
    # Pre vs Post comparison
    pre_data = full_timeline_data[full_timeline_data['phase'] == 'pre']
    post_data = full_timeline_data[full_timeline_data['phase'] == 'post']
    
    variables = ['icp', 'abp', 'cpp']
    
    print(f"{'Variable':<8} {'Pre-SAH':<20} {'Post-SAH':<20} {'Change':<15}")
    print("-" * 70)
    
    for var in variables:
        pre_mean = pre_data[var].mean()
        pre_std = pre_data[var].std()
        post_mean = post_data[var].mean()
        post_std = post_data[var].std()
        change = post_mean - pre_mean
        
        # Format with proper string formatting (fixed the syntax error)
        print(f"{var.upper():<8} {pre_mean:.1f} ± {pre_std:.1f} mmHg    {post_mean:.1f} ± {post_std:.1f} mmHg    {change:+.1f}")
    
    print("="*70)
    print(f"Total animals analyzed: {full_timeline_data['animal_id'].nunique()}")
    print(f"Total duration: {(full_timeline_data['time_seconds'].max() - full_timeline_data['time_seconds'].min())/60:.1f} minutes")
    print("="*70)

# %% SAH Severity Analysis using Metadata

def analyze_sah_severity():
    """Analyze SAH severity using metadata variables"""
    
    if sah_metadata is None:
        print("No metadata available for severity analysis")
        return None
    
    print("SAH SEVERITY ANALYSIS")
    print("="*50)
    
    # Check required columns
    required_cols = ['ID', 'icp_peak', 'time_icp_peak', 'time_induction', 'sugawara_grading']
    missing_cols = [col for col in required_cols if col not in sah_metadata.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {sah_metadata.columns.tolist()}")
        return None
    
    # Create analysis dataframe
    severity_data = sah_metadata.copy()
    
    # Calculate lag_icp_increase (time from induction to peak)
    # Convert datetime columns if they're not already datetime
    for col in ['time_icp_peak', 'time_induction']:
        if col in severity_data.columns:
            if severity_data[col].dtype == 'object':
                try:
                    severity_data[col] = pd.to_datetime(severity_data[col])
                except:
                    print(f"Could not convert {col} to datetime")
    
    # Calculate lag time in seconds
    if severity_data['time_icp_peak'].dtype.name.startswith('datetime') and \
       severity_data['time_induction'].dtype.name.startswith('datetime'):
        severity_data['lag_icp_increase'] = (severity_data['time_icp_peak'] - 
                                            severity_data['time_induction']).dt.total_seconds()
    else:
        # If not datetime, assume they're in the same units and subtract directly
        severity_data['lag_icp_increase'] = severity_data['time_icp_peak'] - severity_data['time_induction']
    
    # Remove any rows with missing critical data
    analysis_vars = ['icp_peak', 'lag_icp_increase', 'sugawara_grading']
    clean_data = severity_data.dropna(subset=analysis_vars)
    
    print(f"Animals with complete severity data: {len(clean_data)}/{len(severity_data)}")
    
    # Summary statistics
    print(f"\nSeverity Metrics Summary:")
    print(f"ICP Peak (mmHg): {clean_data['icp_peak'].mean():.1f} ± {clean_data['icp_peak'].std():.1f}")
    print(f"Lag to Peak (seconds): {clean_data['lag_icp_increase'].mean():.1f} ± {clean_data['lag_icp_increase'].std():.1f}")
    print(f"Sugawara Grading: {clean_data['sugawara_grading'].mean():.1f} ± {clean_data['sugawara_grading'].std():.1f}")
    
    # Distribution by Sugawara grade
    print(f"\nDistribution by Sugawara Grade:")
    grade_summary = clean_data.groupby('sugawara_grading').agg({
        'icp_peak': ['count', 'mean', 'std'],
        'lag_icp_increase': ['mean', 'std']
    }).round(2)
    print(grade_summary)
    
    return clean_data

def create_severity_summary_plot(severity_data):
    """Create summary visualization of SAH severity metrics"""
    
    if severity_data is None or len(severity_data) == 0:
        print("No data available for severity summary plot")
        return None
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: ICP Peak distribution
    ax1.hist(severity_data['icp_peak'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('ICP Peak Increase (mmHg)', fontweight='bold')
    ax1.set_ylabel('Number of Animals', fontweight='bold')
    ax1.set_title('Distribution of ICP Peak Increases', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_peak = severity_data['icp_peak'].mean()
    std_peak = severity_data['icp_peak'].std()
    ax1.axvline(mean_peak, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_peak:.1f} mmHg')
    ax1.legend()
    
    # Plot 2: Lag time distribution
    ax2.hist(severity_data['lag_icp_increase'], bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Time to ICP Peak (seconds)', fontweight='bold')
    ax2.set_ylabel('Number of Animals', fontweight='bold')
    ax2.set_title('Distribution of Time to ICP Peak', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_lag = severity_data['lag_icp_increase'].mean()
    ax2.axvline(mean_lag, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lag:.1f} s')
    ax2.legend()
    
    # Plot 3: Sugawara grading distribution
    grade_counts = severity_data['sugawara_grading'].value_counts().sort_index()
    ax3.bar(grade_counts.index, grade_counts.values, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Sugawara Grade', fontweight='bold')
    ax3.set_ylabel('Number of Animals', fontweight='bold')
    ax3.set_title('Distribution of SAH Severity (Sugawara Grading)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation matrix heatmap
    corr_vars = ['icp_peak', 'lag_icp_increase', 'sugawara_grading']
    corr_matrix = severity_data[corr_vars].corr()
    
    # Create heatmap
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add correlation values as text
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax4.set_xticks(range(len(corr_vars)))
    ax4.set_yticks(range(len(corr_vars)))
    ax4.set_xticklabels(['ICP Peak', 'Lag Time', 'Sugawara Grade'])
    ax4.set_yticklabels(['ICP Peak', 'Lag Time', 'Sugawara Grade'])
    ax4.set_title('Correlation Matrix of Severity Metrics', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.suptitle('SAH Severity Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    return fig

def create_icp_sugawara_correlation_plot(severity_data):
    """Create correlation plot between ICP peak and Sugawara grading"""
    
    if severity_data is None or len(severity_data) == 0:
        print("No data available for correlation plot")
        return None
    
    # Remove any remaining NaN values for this analysis
    plot_data = severity_data[['icp_peak', 'sugawara_grading']].dropna()
    
    if len(plot_data) < 5:
        print("Insufficient data for correlation analysis")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(plot_data['icp_peak'], plot_data['sugawara_grading'], 
              alpha=0.7, s=80, color='darkblue', edgecolors='black', linewidth=1)
    
    # Calculate linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(plot_data['icp_peak'], 
                                                           plot_data['sugawara_grading'])
    
    # Create regression line
    x_line = np.linspace(plot_data['icp_peak'].min(), plot_data['icp_peak'].max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot regression line
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Linear fit (R² = {r_value**2:.3f})')
    
    # Calculate confidence intervals for the regression line
    from scipy import stats as scipy_stats
    
    # Standard error of prediction
    y_pred = slope * plot_data['icp_peak'] + intercept
    residuals = plot_data['sugawara_grading'] - y_pred
    mse = np.sum(residuals**2) / (len(plot_data) - 2)  # Mean squared error
    
    # Calculate standard error for prediction
    x_mean = plot_data['icp_peak'].mean()
    sxx = np.sum((plot_data['icp_peak'] - x_mean)**2)
    
    # 95% confidence interval
    t_val = scipy_stats.t.ppf(0.975, len(plot_data) - 2)  # 95% confidence
    
    # Calculate confidence bands
    se_line = np.sqrt(mse * (1/len(plot_data) + (x_line - x_mean)**2 / sxx))
    ci_upper = y_line + t_val * se_line
    ci_lower = y_line - t_val * se_line
    
    # Plot confidence bands
    ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red', label='95% Confidence Interval')
    
    # Labels and formatting
    ax.set_xlabel('ICP Peak Increase (mmHg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sugawara Grading', fontsize=12, fontweight='bold')
    ax.set_title('Correlation between ICP Peak and SAH Severity\n(Sugawara Grading)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text box
    stats_text = f'n = {len(plot_data)}\nR² = {r_value**2:.3f}\np = {p_value:.3f}\nSlope = {slope:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

# %% LDF Data Analysis

def load_ldf_data():
    """Load LDF data from resliced timeline CSV files (90 minutes: 30 pre + 60 post SAH)"""
    
    # Define LDF data path - CORRECTED to resliced data
    ldf_data_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/A_A_SAH_processing_cleaned_files/SAH_full_with_LDF_resliced_csv"
    
    print("LDF RESLICED DATA LOADING")
    print("="*50)
    print("Loading pre-sliced 90-minute data (30 min pre-SAH + 60 min post-SAH)")
    
    # Check if directory exists and list files
    if not os.path.exists(ldf_data_path):
        print(f"ERROR: LDF resliced directory not found - {ldf_data_path}")
        return None
    
    ldf_csv_files = glob.glob(f"{ldf_data_path}/A*_per000.csv")
    print(f"LDF resliced directory found: {ldf_data_path}")
    print(f"LDF CSV files found: {len(ldf_csv_files)}")
    print(f"Sample LDF files: {[os.path.basename(f) for f in ldf_csv_files[:5]]}")
    
    # Test loading one LDF file to understand structure
    if ldf_csv_files:
        test_file = ldf_csv_files[0]
        test_ldf = load_german_csv(test_file)
        
        if test_ldf is not None:
            print(f"\nTest LDF resliced file: {os.path.basename(test_file)}")
            print(f"Shape: {test_ldf.shape}")
            print(f"Expected: ~5400 points (90 min * 60 sec/min)")
            print(f"Columns: {test_ldf.columns.tolist()}")
            print(f"\nFirst few rows:")
            print(test_ldf.head())
            
            # Check DateTime and LDF columns
            if 'DateTime' in test_ldf.columns:
                print(f"\nDateTime analysis:")
                print(f"Sample values: {test_ldf['DateTime'].head().tolist()}")
                print(f"Min value: {test_ldf['DateTime'].min()}")
                print(f"Max value: {test_ldf['DateTime'].max()}")
                
                # Calculate duration
                duration_minutes = len(test_ldf) / 60
                print(f"Actual duration at 1 Hz: {duration_minutes:.1f} minutes")
                print(f"Expected ~90 minutes for resliced data")
            
            # Check LDF columns
            for ldf_col in ['ldf_left', 'ldf_right']:
                if ldf_col in test_ldf.columns:
                    completeness = (1 - test_ldf[ldf_col].isnull().sum() / len(test_ldf)) * 100
                    print(f"{ldf_col}: {completeness:.1f}% complete")
                else:
                    print(f"WARNING: {ldf_col} column not found")
    
    return ldf_csv_files

def calculate_map_from_raw_abp(abp_values):
    """
    Calculate Mean Arterial Pressure (MAP) from raw ABP waveform data
    
    Since the ABP data is 2.5s moving averaged raw ABP (not MAP), we need to 
    calculate the actual MAP. For continuous ABP waveforms, MAP is typically
    calculated as the time-weighted average of the pressure waveform.
    
    For 1Hz sampled data that's already been smoothed with 2.5s moving average,
    we can use the values directly as they represent the mean pressure over
    short time windows.
    """
    # Since the data is already 2.5s moving averaged raw ABP at 1Hz,
    # these values are effectively short-term MAP estimates
    # We can use them directly or apply additional smoothing if needed
    
    return abp_values  # The 2.5s moving average is already a reasonable MAP estimate

def load_all_ldf_resliced_data(ldf_files):
    """Load all LDF resliced data (already time-aligned: 30 min pre + 60 min post SAH)"""
    
    def extract_animal_id_from_ldf_filename(filename):
        """Extract animal ID from LDF filename (A1_per000.csv -> A1)"""
        # Remove .csv extension first
        base_name = filename.replace('.csv', '')
        # Split by underscore and take the first part (A1_per000 -> A1)
        animal_id = base_name.split('_')[0]
        return animal_id
    
    if not ldf_files:
        print("No LDF resliced files available")
        return None
    
    all_ldf_data = []
    successful_animals = 0
    failed_animals = []
    
    print(f"\nLoading resliced LDF data from {len(ldf_files)} animals...")
    print("Note: Data is already time-sliced (30 min pre + 60 min post SAH)")
    print("SAH induction occurs at 30 minutes (1800 seconds) into each recording")
    
    for i, filepath in enumerate(ldf_files):
        # Extract animal ID from LDF filename (A1_per000.csv -> A1)
        animal_id = extract_animal_id_from_ldf_filename(os.path.basename(filepath))
        
        try:
            print(f"\nProcessing resliced LDF for {animal_id} (from file: {os.path.basename(filepath)}):")
            
            # Load LDF data
            ldf_data = load_german_csv(filepath)
            
            if ldf_data is not None:
                print(f"  Data points: {len(ldf_data)}")
                print(f"  Duration: {len(ldf_data)/60:.1f} minutes")
                print(f"  Expected: ~90 minutes (30 pre + 60 post SAH)")
                
                # Add animal ID
                ldf_data['animal_id'] = animal_id
                
                # For resliced data, SAH induction is at 30 minutes (1800 seconds)
                # Create time relative to SAH (SAH = time 0)
                sah_timepoint = 30 * 60  # 30 minutes * 60 seconds = 1800 seconds
                ldf_data['time_seconds'] = np.arange(len(ldf_data)) - sah_timepoint
                
                print(f"  SAH induction at index {sah_timepoint} (30 minutes)")
                print(f"  Time range: {ldf_data['time_seconds'].min():.0f} to {ldf_data['time_seconds'].max():.0f} seconds")
                print(f"  Pre-SAH duration: {abs(ldf_data['time_seconds'].min())/60:.1f} minutes")
                print(f"  Post-SAH duration: {ldf_data['time_seconds'].max()/60:.1f} minutes")
                
                # Calculate MAP from ABP if available (2.5s moving averaged raw ABP)
                if 'abp' in ldf_data.columns:
                    ldf_data['map'] = calculate_map_from_raw_abp(ldf_data['abp'])
                    print(f"  ✓ Calculated MAP from 2.5s averaged ABP data")
                else:
                    ldf_data['map'] = np.nan
                    print(f"  No ABP data available for MAP calculation")
                
                # Calculate CPP (if both MAP and ICP available)
                if 'icp' in ldf_data.columns and 'map' in ldf_data.columns:
                    ldf_data['cpp'] = ldf_data['map'] - ldf_data['icp']
                    print(f"  ✓ Calculated CPP from MAP and ICP")
                else:
                    ldf_data['cpp'] = np.nan
                    print(f"  Cannot calculate CPP - missing ICP or MAP data")
                
                # Add phase labels based on time
                ldf_data['phase'] = 'post'  # Default to post
                ldf_data.loc[ldf_data['time_seconds'] < 0, 'phase'] = 'pre'
                
                # Show phase distribution for this animal
                phase_counts = ldf_data['phase'].value_counts()
                print(f"  Phase distribution: {dict(phase_counts)}")
                
                # Check data quality for all variables
                data_vars = ['abp', 'icp', 'map', 'cpp', 'ldf_left', 'ldf_right']
                for var in data_vars:
                    if var in ldf_data.columns:
                        completeness = (1 - ldf_data[var].isnull().sum() / len(ldf_data)) * 100
                        print(f"  {var}: {completeness:.1f}% complete")
                
                all_ldf_data.append(ldf_data)
                successful_animals += 1
                
                if successful_animals % 10 == 0:
                    print(f"\nProcessed {successful_animals} resliced LDF files so far...")
                    
        except Exception as e:
            print(f"Error processing resliced LDF for {animal_id}: {e}")
            failed_animals.append(animal_id)
    
    print(f"\nResliced LDF data loading complete:")
    print(f"Successful: {successful_animals} animals")
    print(f"Failed: {len(failed_animals)} animals")
    if failed_animals:
        print(f"Failed animals: {failed_animals}")
    
    # Combine all LDF data
    if all_ldf_data:
        combined_ldf_data = pd.concat(all_ldf_data, ignore_index=True)
        return combined_ldf_data
    else:
        return None

def create_ldf_summary_stats(ldf_data, bin_size_seconds=30):
    """Create time-binned statistics for LDF data (same approach as ICP/ABP/CPP)"""
    
    if ldf_data is None:
        return None
    
    # Use the same time range as before: -30 to +65 minutes, excluding first 5 minutes
    time_min = max(ldf_data['time_seconds'].min(), -30 * 60)  # Start at -30 minutes
    time_max = max(ldf_data['time_seconds'].max(), 65 * 60)   # At least +65 minutes
    
    print(f"Creating LDF time bins from {time_min/60:.1f} to {time_max/60:.1f} minutes")
    print(f"Bin size: {bin_size_seconds} seconds ({bin_size_seconds/60:.1f} minutes)")
    
    # Create time-0-aligned bins (same as before)
    pre_bins = [-i * bin_size_seconds for i in range(int(abs(time_min) / bin_size_seconds), -1, -1)]
    post_bins = [i * bin_size_seconds for i in range(int(time_max / bin_size_seconds) + 1)]
    time_bins = sorted(list(set(pre_bins + post_bins)))
    
    # Calculate bin centers with special handling for SAH transition
    bin_centers = []
    for i in range(len(time_bins)-1):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        if bin_start == 0:
            bin_centers.append(0)  # Perfect alignment at SAH
        else:
            bin_centers.append((bin_start + bin_end) / 2)
    
    # Calculate statistics for each bin
    results = []
    ldf_variables = ['ldf_left', 'ldf_right']
    
    for i, bin_center in enumerate(bin_centers):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        # Get data in this time bin
        bin_mask = (ldf_data['time_seconds'] >= bin_start) & (ldf_data['time_seconds'] < bin_end)
        bin_data = ldf_data[bin_mask]
        
        if len(bin_data) > 0:
            bin_stats = {
                'time_minutes': bin_center / 60,
                'time_seconds': bin_center,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'n_animals': bin_data['animal_id'].nunique(),
                'n_observations': len(bin_data)
            }
            
            # Calculate statistics for LDF variables
            for var in ldf_variables:
                if var in bin_data.columns:
                    var_data = bin_data[var].dropna()
                    
                    if len(var_data) > 0:
                        # Calculate median, IQR, and other statistics
                        bin_stats[f'{var}_median'] = np.median(var_data)
                        bin_stats[f'{var}_q25'] = np.percentile(var_data, 25)
                        bin_stats[f'{var}_q75'] = np.percentile(var_data, 75)
                        bin_stats[f'{var}_iqr'] = np.percentile(var_data, 75) - np.percentile(var_data, 25)
                        bin_stats[f'{var}_std'] = np.std(var_data)
                        bin_stats[f'{var}_n'] = len(var_data)
                    else:
                        # Fill with NaN if no data
                        for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                            bin_stats[f'{var}_{stat}'] = np.nan
                        bin_stats[f'{var}_n'] = 0
                else:
                    # Fill with NaN if column doesn't exist
                    for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                        bin_stats[f'{var}_{stat}'] = np.nan
                    bin_stats[f'{var}_n'] = 0
            
            results.append(bin_stats)
    
    return pd.DataFrame(results)

def create_ldf_temporal_plot(ldf_stats):
    """Create temporal evolution plot for LDF data"""
    
    if ldf_stats is None or len(ldf_stats) == 0:
        print("No LDF statistics available for plotting")
        return None
    
    # Filter out time bins with insufficient data
    ldf_stats = ldf_stats[ldf_stats['n_observations'] >= 10].copy()
    
    print(f"Plotting LDF with {len(ldf_stats)} time bins")
    
    # Create figure with subplots for left and right LDF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Colors for LDF sides
    colors = {'ldf_left': 'red', 'ldf_right': 'blue'}  # Left=red (SAH side), Right=blue (control)
    
    # Helper function to plot LDF with IQR error bands
    def plot_ldf_with_iqr(ax, time, median, var, color, label):
        # Shift time by 1 minute for alignment (same as before)
        time_shifted = time + 1
        
        # Plot the median line
        ax.plot(time_shifted, median, color=color, linewidth=2.5, label=label, alpha=0.9)
        
        # IQR error bands
        lower = ldf_stats[f'{var}_q25']
        upper = ldf_stats[f'{var}_q75']
        
        # Plot IQR bands
        ax.fill_between(time_shifted, lower, upper, alpha=0.25, color=color)
    
    # Plot 1: Left LDF (SAH side)
    if 'ldf_left_median' in ldf_stats.columns:
        plot_ldf_with_iqr(ax1, ldf_stats['time_minutes'], ldf_stats['ldf_left_median'], 
                          'ldf_left', colors['ldf_left'], 'LDF Left (SAH side)')
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax1.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax1.set_ylabel('LDF Left (perfusion units)', fontsize=12, fontweight='bold')
    ax1.set_title('Laser Doppler Flowmetry - Left Hemisphere (SAH Side)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Right LDF (control side)
    if 'ldf_right_median' in ldf_stats.columns:
        plot_ldf_with_iqr(ax2, ldf_stats['time_minutes'], ldf_stats['ldf_right_median'], 
                          'ldf_right', colors['ldf_right'], 'LDF Right (control side)')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax2.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LDF Right (perfusion units)', fontsize=12, fontweight='bold')
    ax2.set_title('Laser Doppler Flowmetry - Right Hemisphere (Control Side)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add main title
    fig.suptitle('Bilateral Laser Doppler Flowmetry During Experimental SAH (Median ± IQR)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.show()
    
    return fig

# Run LDF analysis
print("\n" + "="*70)
print("LDF DATA ANALYSIS")
print("="*70)

# Load LDF data
ldf_files = load_ldf_data()

if ldf_files:
    # Load and process all resliced LDF data (no metadata alignment needed!)
    full_ldf_data = load_all_ldf_resliced_data(ldf_files)
    
    if full_ldf_data is not None:
        print(f"\nLDF DATASET SUMMARY")
        print("="*50)
        print(f"Total observations: {len(full_ldf_data):,}")
        print(f"Animals: {full_ldf_data['animal_id'].nunique()}")
        print(f"Time range: {full_ldf_data['time_seconds'].min():.0f} to {full_ldf_data['time_seconds'].max():.0f} seconds")
        
        # Check data completeness
        print(f"\nLDF data completeness:")
        for var in ['ldf_left', 'ldf_right']:
            if var in full_ldf_data.columns:
                completeness = (1 - full_ldf_data[var].isnull().sum() / len(full_ldf_data)) * 100
                print(f"{var.upper()}: {completeness:.1f}% complete")
        
        # Phase distribution
        print(f"\nPhase distribution:")
        phase_summary = full_ldf_data.groupby('phase').agg({
            'time_seconds': ['count', 'min', 'max'],
            'animal_id': 'nunique'
        })
        print(phase_summary)
        
        # Create time-binned statistics
        print(f"\nCalculating LDF time-binned statistics...")
        ldf_binned_stats = create_ldf_summary_stats(full_ldf_data, bin_size_seconds=30)
        
        if ldf_binned_stats is not None:
            print(f"Created {len(ldf_binned_stats)} LDF time bins")
            print(f"Time range: {ldf_binned_stats['time_minutes'].min():.1f} to {ldf_binned_stats['time_minutes'].max():.1f} minutes")
            
            # Create LDF temporal plots
            print(f"\nCreating LDF temporal evolution plots...")
            ldf_fig = create_ldf_temporal_plot(ldf_binned_stats)
            
            print(f"\n✅ LDF temporal analysis complete!")
            print(f"Generated bilateral LDF plots showing SAH side vs control side")
        else:
            print("Failed to create LDF statistics")
    else:
        print("Failed to load LDF timeline data")
else:
    print("No LDF files found")

print("\n" + "="*70)



# %% Corrected code with adjustment for differences in baseline levels
"""
I made a mistake in my previous LDF over time plot. I did not adjust for differences in 
baseline values. I will redo the entire code here whilst calculating ldf_left_adj
and ldf_right_adj which will be the absolute values - the mean calculate from 3
5 minutes or 300s of data before SAH induction. 
"""

def normalize_ldf_to_baseline(ldf_data, baseline_window_seconds=300):
    """
    Normalize LDF data to baseline values calculated from a pre-SAH time window.
    
    Parameters:
    -----------
    ldf_data : pd.DataFrame
        DataFrame containing LDF data with 'time_seconds', 'animal_id', 'ldf_left', 'ldf_right'
    baseline_window_seconds : int
        Duration of baseline window in seconds (default: 300 = 5 minutes)
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns 'ldf_left_adj' and 'ldf_right_adj'
    """
    
    print(f"NORMALIZING LDF DATA TO BASELINE")
    print("="*50)
    print(f"Baseline window: {baseline_window_seconds} seconds ({baseline_window_seconds/60:.1f} minutes)")
    print(f"Baseline period: -{baseline_window_seconds} to 0 seconds relative to SAH")
    
    # Create a copy to avoid modifying original data
    normalized_data = ldf_data.copy()
    
    # Initialize adjustment columns
    normalized_data['ldf_left_adj'] = np.nan
    normalized_data['ldf_right_adj'] = np.nan
    normalized_data['ldf_left_baseline'] = np.nan
    normalized_data['ldf_right_baseline'] = np.nan
    
    # Track processing statistics
    successful_animals = 0
    failed_animals = []
    baseline_stats = []
    
    # Process each animal separately
    unique_animals = normalized_data['animal_id'].unique()
    print(f"\nProcessing {len(unique_animals)} animals...")
    
    for animal_id in unique_animals:
        try:
            print(f"\nProcessing {animal_id}:")
            
            # Get data for this animal
            animal_mask = normalized_data['animal_id'] == animal_id
            animal_data = normalized_data[animal_mask].copy()
            
            # Define baseline period: from -baseline_window_seconds to 0 seconds
            baseline_start = -baseline_window_seconds
            baseline_end = 0
            
            # Get baseline data
            baseline_mask = ((animal_data['time_seconds'] >= baseline_start) & 
                           (animal_data['time_seconds'] < baseline_end))
            baseline_data = animal_data[baseline_mask]
            
            print(f"  Total data points: {len(animal_data)}")
            print(f"  Baseline data points: {len(baseline_data)}")
            print(f"  Baseline time range: {baseline_data['time_seconds'].min():.0f} to {baseline_data['time_seconds'].max():.0f} seconds")
            
            if len(baseline_data) < 60:  # Less than 1 minute of baseline data
                print(f"  WARNING: Insufficient baseline data ({len(baseline_data)} points)")
                failed_animals.append(animal_id)
                continue
            
            # Calculate baseline values for each hemisphere
            baseline_stats_animal = {'animal_id': animal_id}
            
            for side in ['left', 'right']:
                ldf_col = f'ldf_{side}'
                adj_col = f'ldf_{side}_adj'
                baseline_col = f'ldf_{side}_baseline'
                
                if ldf_col in baseline_data.columns:
                    # Get baseline LDF data (remove NaN values)
                    baseline_ldf = baseline_data[ldf_col].dropna()
                    
                    if len(baseline_ldf) >= 30:  # At least 30 seconds of data
                        # Calculate baseline mean
                        baseline_mean = baseline_ldf.mean()
                        baseline_std = baseline_ldf.std()
                        baseline_stats_animal[f'{side}_baseline_mean'] = baseline_mean
                        baseline_stats_animal[f'{side}_baseline_std'] = baseline_std
                        baseline_stats_animal[f'{side}_baseline_n'] = len(baseline_ldf)
                        
                        print(f"  {side.upper()} baseline: {baseline_mean:.2f} ± {baseline_std:.2f} PU (n={len(baseline_ldf)})")
                        
                        # Apply normalization to all data for this animal and side
                        animal_ldf_data = animal_data[ldf_col]
                        normalized_values = animal_ldf_data - baseline_mean
                        
                        # Store normalized values back to main dataframe
                        normalized_data.loc[animal_mask, adj_col] = normalized_values
                        normalized_data.loc[animal_mask, baseline_col] = baseline_mean
                        
                        print(f"  ✓ Normalized {side} LDF data (baseline subtracted)")
                        
                    else:
                        print(f"  WARNING: Insufficient {side} baseline data ({len(baseline_ldf)} points)")
                        baseline_stats_animal[f'{side}_baseline_mean'] = np.nan
                        baseline_stats_animal[f'{side}_baseline_std'] = np.nan
                        baseline_stats_animal[f'{side}_baseline_n'] = 0
                else:
                    print(f"  WARNING: {ldf_col} column not found")
                    baseline_stats_animal[f'{side}_baseline_mean'] = np.nan
                    baseline_stats_animal[f'{side}_baseline_std'] = np.nan
                    baseline_stats_animal[f'{side}_baseline_n'] = 0
            
            baseline_stats.append(baseline_stats_animal)
            successful_animals += 1
            
        except Exception as e:
            print(f"  ERROR processing {animal_id}: {e}")
            failed_animals.append(animal_id)
    
    print(f"\nBASELINE NORMALIZATION COMPLETE")
    print(f"Successful: {successful_animals} animals")
    print(f"Failed: {len(failed_animals)} animals")
    if failed_animals:
        print(f"Failed animals: {failed_animals}")
    
    # Create baseline statistics summary
    baseline_df = pd.DataFrame(baseline_stats)
    
    if len(baseline_df) > 0:
        print(f"\nBASELINE STATISTICS SUMMARY")
        print("-" * 40)
        
        for side in ['left', 'right']:
            mean_col = f'{side}_baseline_mean'
            if mean_col in baseline_df.columns:
                baseline_means = baseline_df[mean_col].dropna()
                if len(baseline_means) > 0:
                    print(f"{side.upper()} hemisphere baseline values:")
                    print(f"  Range: {baseline_means.min():.2f} to {baseline_means.max():.2f} PU")
                    print(f"  Mean across animals: {baseline_means.mean():.2f} ± {baseline_means.std():.2f} PU")
                    print(f"  Animals with data: {len(baseline_means)}")
                else:
                    print(f"{side.upper()} hemisphere: No baseline data available")
    
    # Check normalization results
    print(f"\nNORMALIZATION RESULTS CHECK")
    print("-" * 40)
    
    for side in ['left', 'right']:
        adj_col = f'ldf_{side}_adj'
        if adj_col in normalized_data.columns:
            adj_data = normalized_data[adj_col].dropna()
            if len(adj_data) > 0:
                print(f"{side.upper()} adjusted LDF:")
                print(f"  Range: {adj_data.min():.2f} to {adj_data.max():.2f} PU change")
                print(f"  Mean: {adj_data.mean():.2f} ± {adj_data.std():.2f} PU change")
                print(f"  Data points: {len(adj_data):,}")
                
                # Check baseline period (should be centered around 0)
                baseline_adj = normalized_data[(normalized_data['time_seconds'] >= -300) & 
                                             (normalized_data['time_seconds'] < 0)][adj_col].dropna()
                if len(baseline_adj) > 0:
                    print(f"  Baseline period mean: {baseline_adj.mean():.3f} PU change (should be ~0)")
            else:
                print(f"{side.upper()} adjusted LDF: No data available")
    
    return normalized_data, baseline_df

def create_ldf_summary_stats_normalized(ldf_data, bin_size_seconds=30):
    """Create time-binned statistics for normalized LDF data"""
    
    if ldf_data is None:
        return None
    
    # Use the same time range as before: -30 to +65 minutes
    time_min = max(ldf_data['time_seconds'].min(), -30 * 60)
    time_max = max(ldf_data['time_seconds'].max(), 65 * 60)
    
    print(f"Creating normalized LDF time bins from {time_min/60:.1f} to {time_max/60:.1f} minutes")
    
    # Create time-0-aligned bins (same as before)
    pre_bins = [-i * bin_size_seconds for i in range(int(abs(time_min) / bin_size_seconds), -1, -1)]
    post_bins = [i * bin_size_seconds for i in range(int(time_max / bin_size_seconds) + 1)]
    time_bins = sorted(list(set(pre_bins + post_bins)))
    
    # Calculate bin centers with special handling for SAH transition
    bin_centers = []
    for i in range(len(time_bins)-1):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        if bin_start == 0:
            bin_centers.append(0)  # Perfect alignment at SAH
        else:
            bin_centers.append((bin_start + bin_end) / 2)
    
    # Calculate statistics for each bin
    results = []
    ldf_variables = ['ldf_left_adj', 'ldf_right_adj']  # Use normalized variables
    
    for i, bin_center in enumerate(bin_centers):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        # Get data in this time bin
        bin_mask = (ldf_data['time_seconds'] >= bin_start) & (ldf_data['time_seconds'] < bin_end)
        bin_data = ldf_data[bin_mask]
        
        if len(bin_data) > 0:
            bin_stats = {
                'time_minutes': bin_center / 60,
                'time_seconds': bin_center,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'n_animals': bin_data['animal_id'].nunique(),
                'n_observations': len(bin_data)
            }
            
            # Calculate statistics for normalized LDF variables
            for var in ldf_variables:
                if var in bin_data.columns:
                    var_data = bin_data[var].dropna()
                    
                    if len(var_data) > 0:
                        # Calculate median, IQR, and other statistics
                        bin_stats[f'{var}_median'] = np.median(var_data)
                        bin_stats[f'{var}_q25'] = np.percentile(var_data, 25)
                        bin_stats[f'{var}_q75'] = np.percentile(var_data, 75)
                        bin_stats[f'{var}_iqr'] = np.percentile(var_data, 75) - np.percentile(var_data, 25)
                        bin_stats[f'{var}_std'] = np.std(var_data)
                        bin_stats[f'{var}_n'] = len(var_data)
                    else:
                        # Fill with NaN if no data
                        for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                            bin_stats[f'{var}_{stat}'] = np.nan
                        bin_stats[f'{var}_n'] = 0
                else:
                    # Fill with NaN if column doesn't exist
                    for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                        bin_stats[f'{var}_{stat}'] = np.nan
                    bin_stats[f'{var}_n'] = 0
            
            results.append(bin_stats)
    
    return pd.DataFrame(results)

def create_ldf_temporal_plot_normalized(ldf_stats):
    """Create temporal evolution plot for normalized LDF data"""
    
    if ldf_stats is None or len(ldf_stats) == 0:
        print("No normalized LDF statistics available for plotting")
        return None
    
    # Filter out time bins with insufficient data
    ldf_stats = ldf_stats[ldf_stats['n_observations'] >= 10].copy()
    
    print(f"Plotting normalized LDF with {len(ldf_stats)} time bins")
    
    # Create figure with subplots for left and right LDF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Colors for LDF sides
    colors = {'ldf_left_adj': 'red', 'ldf_right_adj': 'blue'}  # Left=red (SAH side), Right=blue (control)
    
    # Helper function to plot normalized LDF with IQR error bands
    def plot_normalized_ldf_with_iqr(ax, time, median, var, color, label):
        # Shift time by 1 minute for alignment (same as before)
        time_shifted = time + 1
        
        # Plot the median line
        ax.plot(time_shifted, median, color=color, linewidth=2.5, label=label, alpha=0.9)
        
        # IQR error bands
        lower = ldf_stats[f'{var}_q25']
        upper = ldf_stats[f'{var}_q75']
        
        # Plot IQR bands
        ax.fill_between(time_shifted, lower, upper, alpha=0.25, color=color)
    
    # Plot 1: Left LDF (SAH side) - normalized
    if 'ldf_left_adj_median' in ldf_stats.columns:
        plot_normalized_ldf_with_iqr(ax1, ldf_stats['time_minutes'], ldf_stats['ldf_left_adj_median'], 
                                   'ldf_left_adj', colors['ldf_left_adj'], 'LDF Left (SAH side)')
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Baseline (0)')
    ax1.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax1.set_ylabel('LDF Left (PU change from baseline)', fontsize=12, fontweight='bold')
    ax1.set_title('Laser Doppler Flowmetry - Left Hemisphere (SAH Side)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Right LDF (control side) - normalized
    if 'ldf_right_adj_median' in ldf_stats.columns:
        plot_normalized_ldf_with_iqr(ax2, ldf_stats['time_minutes'], ldf_stats['ldf_right_adj_median'], 
                                   'ldf_right_adj', colors['ldf_right_adj'], 'LDF Right (control side)')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Baseline (0)')
    ax2.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LDF Right (PU change from baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Laser Doppler Flowmetry - Right Hemisphere (Control Side)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add main title
  #  fig.suptitle('Bilateral Laser Doppler Flowmetry During Experimental SAH (Median ± IQR)', 
  #               fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    
    return fig

# Apply the normalization to your existing LDF data
if 'full_ldf_data' in locals() and full_ldf_data is not None:
    print("\n" + "="*70)
    print("APPLYING LDF BASELINE NORMALIZATION")
    print("="*70)
    
    # Normalize LDF data to baseline (5-minute window before SAH)
    normalized_ldf_data, baseline_summary = normalize_ldf_to_baseline(full_ldf_data, baseline_window_seconds=300)
    
    if normalized_ldf_data is not None:
        # Create time-binned statistics for normalized data
        print(f"\nCalculating normalized LDF time-binned statistics...")
        normalized_ldf_stats = create_ldf_summary_stats_normalized(normalized_ldf_data, bin_size_seconds=30)
        
        if normalized_ldf_stats is not None:
            print(f"Created {len(normalized_ldf_stats)} normalized LDF time bins")
            
            # Create normalized LDF temporal plots
            print(f"\nCreating normalized LDF temporal evolution plots...")
            normalized_ldf_fig = create_ldf_temporal_plot_normalized(normalized_ldf_stats)
            
            print(f"\n✅ LDF baseline normalization and plotting complete!")
            print(f"Generated baseline-normalized bilateral LDF plots")
            print(f"Y-axis now shows 'PU change from baseline' for proper cross-animal comparison")
        else:
            print("Failed to create normalized LDF statistics")
    else:
        print("Failed to normalize LDF data")
else:
    print("No LDF data available for normalization")
    print("Please run the LDF data loading section first")
    
    
# %% Now from this data I will calculate the drop in LDF per hemisphere, which 
# I want to use as a severity marker

def calculate_ldf_drops(normalized_ldf_data):
    """
    Calculate LDF drops per animal per hemisphere.
    
    LDF drop = difference between baseline mean and lowest point after SAH induction
    (expressed as positive values)
    
    Parameters:
    -----------
    normalized_ldf_data : pd.DataFrame
        DataFrame with normalized LDF data containing 'ldf_left_baseline', 'ldf_right_baseline',
        'ldf_left', 'ldf_right', 'time_seconds', 'animal_id'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: animal_id, ldf_left_drop, ldf_right_drop
    """
    
    print("CALCULATING LDF DROPS PER ANIMAL")
    print("="*50)
    print("LDF Drop = Baseline Mean - Minimum Post-SAH Value")
    print("(Expressed as positive values)")
    
    # Initialize results list
    ldf_drops = []
    
    # Get unique animals
    unique_animals = normalized_ldf_data['animal_id'].unique()
    print(f"\nProcessing {len(unique_animals)} animals...")
    
    for animal_id in unique_animals:
        try:
            print(f"\nProcessing {animal_id}:")
            
            # Get data for this animal
            animal_data = normalized_ldf_data[normalized_ldf_data['animal_id'] == animal_id].copy()
            
            # Get post-SAH data (time >= 0)
            post_sah_data = animal_data[animal_data['time_seconds'] >= 0].copy()
            
            if len(post_sah_data) == 0:
                print(f"  WARNING: No post-SAH data found for {animal_id}")
                continue
            
            # Initialize result dictionary for this animal
            animal_result = {'animal_id': animal_id}
            
            # Process each hemisphere
            for side in ['left', 'right']:
                ldf_col = f'ldf_{side}'
                baseline_col = f'ldf_{side}_baseline'
                drop_col = f'ldf_{side}_drop'
                
                # Get baseline value (should be consistent across all rows for this animal)
                baseline_values = animal_data[baseline_col].dropna()
                
                if len(baseline_values) == 0:
                    print(f"  WARNING: No baseline data for {side} hemisphere")
                    animal_result[drop_col] = np.nan
                    continue
                
                baseline_mean = baseline_values.iloc[0]  # Should be the same for all rows
                
                # Get post-SAH LDF values for this hemisphere
                post_sah_ldf = post_sah_data[ldf_col].dropna()
                
                if len(post_sah_ldf) == 0:
                    print(f"  WARNING: No post-SAH {side} LDF data")
                    animal_result[drop_col] = np.nan
                    continue
                
                # Find minimum post-SAH value
                min_post_sah = post_sah_ldf.min()
                min_time_idx = post_sah_ldf.idxmin()
                min_time_seconds = post_sah_data.loc[min_time_idx, 'time_seconds']
                
                # Calculate drop (baseline - minimum, expressed as positive)
                ldf_drop = baseline_mean - min_post_sah
                
                # Ensure it's positive (in case minimum is somehow above baseline)
                ldf_drop = max(0, ldf_drop)
                
                animal_result[drop_col] = ldf_drop
                
                print(f"  {side.upper()} hemisphere:")
                print(f"    Baseline mean: {baseline_mean:.2f} PU")
                print(f"    Post-SAH minimum: {min_post_sah:.2f} PU (at {min_time_seconds/60:.1f} min)")
                print(f"    LDF drop: {ldf_drop:.2f} PU")
            
            ldf_drops.append(animal_result)
            
        except Exception as e:
            print(f"  ERROR processing {animal_id}: {e}")
            # Add animal with NaN values
            ldf_drops.append({
                'animal_id': animal_id,
                'ldf_left_drop': np.nan,
                'ldf_right_drop': np.nan
            })
    
    # Convert to DataFrame
    ldf_drops_df = pd.DataFrame(ldf_drops)
    
    print(f"\nLDF DROP CALCULATION COMPLETE")
    print(f"Animals processed: {len(ldf_drops_df)}")
    
    # Summary statistics
    if len(ldf_drops_df) > 0:
        print(f"\nLDF DROP SUMMARY STATISTICS:")
        print("-" * 40)
        
        for side in ['left', 'right']:
            drop_col = f'ldf_{side}_drop'
            if drop_col in ldf_drops_df.columns:
                drop_values = ldf_drops_df[drop_col].dropna()
                
                if len(drop_values) > 0:
                    print(f"{side.upper()} hemisphere LDF drops:")
                    print(f"  n = {len(drop_values)} animals")
                    print(f"  Mean: {drop_values.mean():.2f} ± {drop_values.std():.2f} PU")
                    print(f"  Range: {drop_values.min():.2f} to {drop_values.max():.2f} PU")
                    print(f"  Median: {drop_values.median():.2f} PU")
                else:
                    print(f"{side.upper()} hemisphere: No valid data")
        
        # Check for missing data
        missing_left = ldf_drops_df['ldf_left_drop'].isnull().sum()
        missing_right = ldf_drops_df['ldf_right_drop'].isnull().sum()
        
        if missing_left > 0 or missing_right > 0:
            print(f"\nMissing data:")
            print(f"  Left hemisphere: {missing_left} animals")
            print(f"  Right hemisphere: {missing_right} animals")
    
    return ldf_drops_df

def export_ldf_drops_csv(ldf_drops_df, output_path):
    """
    Export LDF drops to CSV file with only animal_id, ldf_left_drop, ldf_right_drop
    
    Parameters:
    -----------
    ldf_drops_df : pd.DataFrame
        DataFrame with LDF drop data
    output_path : str
        Full path to output CSV file
    """
    
    if ldf_drops_df is None or len(ldf_drops_df) == 0:
        print("No LDF drop data to export")
        return False
    
    try:
        # Ensure output directory exists
        import os
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Select only required columns and ensure proper order
        export_columns = ['animal_id', 'ldf_left_drop', 'ldf_right_drop']
        
        # Check if all required columns exist
        missing_cols = [col for col in export_columns if col not in ldf_drops_df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns for export: {missing_cols}")
            return False
        
        # Create export dataframe with only required columns
        export_df = ldf_drops_df[export_columns].copy()
        
        # Sort by animal_id for consistent output
        export_df = export_df.sort_values('animal_id')
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        
        print(f"\nLDF DROPS EXPORTED SUCCESSFULLY")
        print(f"File: {output_path}")
        print(f"Animals: {len(export_df)}")
        print(f"Columns: {list(export_df.columns)}")
        
        # Show preview of exported data
        print(f"\nPreview of exported data:")
        print(export_df.head(10))
        
        # Check for any missing values in export
        for col in ['ldf_left_drop', 'ldf_right_drop']:
            missing_count = export_df[col].isnull().sum()
            if missing_count > 0:
                print(f"WARNING: {missing_count} missing values in {col}")
        
        return True
        
    except Exception as e:
        print(f"ERROR exporting LDF drops: {e}")
        return False

def add_ldf_drops_to_main_dataframe(normalized_ldf_data, ldf_drops_df):
    """
    Add LDF drop values to the main dataframe
    
    Parameters:
    -----------
    normalized_ldf_data : pd.DataFrame
        Main LDF dataframe
    ldf_drops_df : pd.DataFrame
        LDF drops per animal
        
    Returns:
    --------
    pd.DataFrame
        Main dataframe with added ldf_left_drop and ldf_right_drop columns
    """
    
    print("ADDING LDF DROPS TO MAIN DATAFRAME")
    print("="*40)
    
    # Create a copy to avoid modifying original
    enhanced_data = normalized_ldf_data.copy()
    
    # Merge LDF drops with main dataframe
    enhanced_data = enhanced_data.merge(
        ldf_drops_df[['animal_id', 'ldf_left_drop', 'ldf_right_drop']], 
        on='animal_id', 
        how='left'
    )
    
    # Check merge results
    animals_with_drops = enhanced_data['ldf_left_drop'].notna().any()
    
    if animals_with_drops:
        print(f"✓ Successfully added LDF drops to main dataframe")
        print(f"  New columns: ldf_left_drop, ldf_right_drop")
        print(f"  Animals with drop data: {enhanced_data.groupby('animal_id')['ldf_left_drop'].first().notna().sum()}")
    else:
        print("WARNING: No LDF drop data was successfully merged")
    
    return enhanced_data

# Execute the LDF drop calculation and export
if 'normalized_ldf_data' in locals() and normalized_ldf_data is not None:
    print("\n" + "="*70)
    print("LDF DROP ANALYSIS AND EXPORT")
    print("="*70)
    
    # Calculate LDF drops per animal per hemisphere
    ldf_drops_df = calculate_ldf_drops(normalized_ldf_data)
    
    if ldf_drops_df is not None and len(ldf_drops_df) > 0:
        
        # Add LDF drops to main dataframe
        enhanced_ldf_data = add_ldf_drops_to_main_dataframe(normalized_ldf_data, ldf_drops_df)
        
        # Export to CSV
        output_csv_path = "/Users/mveldeman/Desktop/Experimental SAH/Statistics/experimental_sah_analysis/data/ldf_drops.csv"
        
        export_success = export_ldf_drops_csv(ldf_drops_df, output_csv_path)
        
        if export_success:
            print(f"\n✅ LDF DROP ANALYSIS COMPLETE!")
            print(f"✅ CSV file exported: ldf_drops.csv")
            print(f"✅ Main dataframe enhanced with drop values")
            
            # Update the main variable for further use
            normalized_ldf_data = enhanced_ldf_data
            
        else:
            print("❌ Failed to export CSV file")
    else:
        print("❌ Failed to calculate LDF drops")
        
else:
    print("❌ No normalized LDF data available")
    print("Please run the LDF data loading and normalization sections first")
    
    
    
    
    # %% Corrected code with adjustment for differences in baseline levels
"""
I made a mistake in my previous LDF over time plot. I did not adjust for differences in 
baseline values. I will redo the entire code here whilst calculating ldf_left_adj
and ldf_right_adj which will be the absolute values - the mean calculate from 3
5 minutes or 300s of data before SAH induction. 
"""

def normalize_ldf_to_baseline(ldf_data, baseline_window_seconds=300):
    """
    Normalize LDF data to baseline values calculated from a pre-SAH time window.
    
    Parameters:
    -----------
    ldf_data : pd.DataFrame
        DataFrame containing LDF data with 'time_seconds', 'animal_id', 'ldf_left', 'ldf_right'
    baseline_window_seconds : int
        Duration of baseline window in seconds (default: 300 = 5 minutes)
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns 'ldf_left_adj' and 'ldf_right_adj'
    """
    
    print(f"NORMALIZING LDF DATA TO BASELINE")
    print("="*50)
    print(f"Baseline window: {baseline_window_seconds} seconds ({baseline_window_seconds/60:.1f} minutes)")
    print(f"Baseline period: -{baseline_window_seconds} to 0 seconds relative to SAH")
    
    # Create a copy to avoid modifying original data
    normalized_data = ldf_data.copy()
    
    # Initialize adjustment columns
    normalized_data['ldf_left_adj'] = np.nan
    normalized_data['ldf_right_adj'] = np.nan
    normalized_data['ldf_left_baseline'] = np.nan
    normalized_data['ldf_right_baseline'] = np.nan
    
    # Track processing statistics
    successful_animals = 0
    failed_animals = []
    baseline_stats = []
    
    # Process each animal separately
    unique_animals = normalized_data['animal_id'].unique()
    print(f"\nProcessing {len(unique_animals)} animals...")
    
    for animal_id in unique_animals:
        try:
            print(f"\nProcessing {animal_id}:")
            
            # Get data for this animal
            animal_mask = normalized_data['animal_id'] == animal_id
            animal_data = normalized_data[animal_mask].copy()
            
            # Define baseline period: from -baseline_window_seconds to 0 seconds
            baseline_start = -baseline_window_seconds
            baseline_end = 0
            
            # Get baseline data
            baseline_mask = ((animal_data['time_seconds'] >= baseline_start) & 
                           (animal_data['time_seconds'] < baseline_end))
            baseline_data = animal_data[baseline_mask]
            
            print(f"  Total data points: {len(animal_data)}")
            print(f"  Baseline data points: {len(baseline_data)}")
            print(f"  Baseline time range: {baseline_data['time_seconds'].min():.0f} to {baseline_data['time_seconds'].max():.0f} seconds")
            
            if len(baseline_data) < 60:  # Less than 1 minute of baseline data
                print(f"  WARNING: Insufficient baseline data ({len(baseline_data)} points)")
                failed_animals.append(animal_id)
                continue
            
            # Calculate baseline values for each hemisphere
            baseline_stats_animal = {'animal_id': animal_id}
            
            for side in ['left', 'right']:
                ldf_col = f'ldf_{side}'
                adj_col = f'ldf_{side}_adj'
                baseline_col = f'ldf_{side}_baseline'
                
                if ldf_col in baseline_data.columns:
                    # Get baseline LDF data (remove NaN values)
                    baseline_ldf = baseline_data[ldf_col].dropna()
                    
                    if len(baseline_ldf) >= 30:  # At least 30 seconds of data
                        # Calculate baseline mean
                        baseline_mean = baseline_ldf.mean()
                        baseline_std = baseline_ldf.std()
                        baseline_stats_animal[f'{side}_baseline_mean'] = baseline_mean
                        baseline_stats_animal[f'{side}_baseline_std'] = baseline_std
                        baseline_stats_animal[f'{side}_baseline_n'] = len(baseline_ldf)
                        
                        print(f"  {side.upper()} baseline: {baseline_mean:.2f} ± {baseline_std:.2f} PU (n={len(baseline_ldf)})")
                        
                        # Apply normalization to all data for this animal and side
                        animal_ldf_data = animal_data[ldf_col]
                        normalized_values = animal_ldf_data - baseline_mean
                        
                        # Store normalized values back to main dataframe
                        normalized_data.loc[animal_mask, adj_col] = normalized_values
                        normalized_data.loc[animal_mask, baseline_col] = baseline_mean
                        
                        print(f"  ✓ Normalized {side} LDF data (baseline subtracted)")
                        
                    else:
                        print(f"  WARNING: Insufficient {side} baseline data ({len(baseline_ldf)} points)")
                        baseline_stats_animal[f'{side}_baseline_mean'] = np.nan
                        baseline_stats_animal[f'{side}_baseline_std'] = np.nan
                        baseline_stats_animal[f'{side}_baseline_n'] = 0
                else:
                    print(f"  WARNING: {ldf_col} column not found")
                    baseline_stats_animal[f'{side}_baseline_mean'] = np.nan
                    baseline_stats_animal[f'{side}_baseline_std'] = np.nan
                    baseline_stats_animal[f'{side}_baseline_n'] = 0
            
            baseline_stats.append(baseline_stats_animal)
            successful_animals += 1
            
        except Exception as e:
            print(f"  ERROR processing {animal_id}: {e}")
            failed_animals.append(animal_id)
    
    print(f"\nBASELINE NORMALIZATION COMPLETE")
    print(f"Successful: {successful_animals} animals")
    print(f"Failed: {len(failed_animals)} animals")
    if failed_animals:
        print(f"Failed animals: {failed_animals}")
    
    # Create baseline statistics summary
    baseline_df = pd.DataFrame(baseline_stats)
    
    if len(baseline_df) > 0:
        print(f"\nBASELINE STATISTICS SUMMARY")
        print("-" * 40)
        
        for side in ['left', 'right']:
            mean_col = f'{side}_baseline_mean'
            if mean_col in baseline_df.columns:
                baseline_means = baseline_df[mean_col].dropna()
                if len(baseline_means) > 0:
                    print(f"{side.upper()} hemisphere baseline values:")
                    print(f"  Range: {baseline_means.min():.2f} to {baseline_means.max():.2f} PU")
                    print(f"  Mean across animals: {baseline_means.mean():.2f} ± {baseline_means.std():.2f} PU")
                    print(f"  Animals with data: {len(baseline_means)}")
                else:
                    print(f"{side.upper()} hemisphere: No baseline data available")
    
    # Check normalization results
    print(f"\nNORMALIZATION RESULTS CHECK")
    print("-" * 40)
    
    for side in ['left', 'right']:
        adj_col = f'ldf_{side}_adj'
        if adj_col in normalized_data.columns:
            adj_data = normalized_data[adj_col].dropna()
            if len(adj_data) > 0:
                print(f"{side.upper()} adjusted LDF:")
                print(f"  Range: {adj_data.min():.2f} to {adj_data.max():.2f} PU change")
                print(f"  Mean: {adj_data.mean():.2f} ± {adj_data.std():.2f} PU change")
                print(f"  Data points: {len(adj_data):,}")
                
                # Check baseline period (should be centered around 0)
                baseline_adj = normalized_data[(normalized_data['time_seconds'] >= -300) & 
                                             (normalized_data['time_seconds'] < 0)][adj_col].dropna()
                if len(baseline_adj) > 0:
                    print(f"  Baseline period mean: {baseline_adj.mean():.3f} PU change (should be ~0)")
            else:
                print(f"{side.upper()} adjusted LDF: No data available")
    
    return normalized_data, baseline_df


def create_ldf_summary_stats_normalized(ldf_data, bin_size_seconds=30):
    """Create time-binned statistics for normalized LDF data"""
    
    if ldf_data is None:
        return None
    
    # Use the same time range as before: -30 to +65 minutes
    time_min = max(ldf_data['time_seconds'].min(), -30 * 60)
    time_max = max(ldf_data['time_seconds'].max(), 65 * 60)
    
    print(f"Creating normalized LDF time bins from {time_min/60:.1f} to {time_max/60:.1f} minutes")
    
    # Create time-0-aligned bins (same as before)
    pre_bins = [-i * bin_size_seconds for i in range(int(abs(time_min) / bin_size_seconds), -1, -1)]
    post_bins = [i * bin_size_seconds for i in range(int(time_max / bin_size_seconds) + 1)]
    time_bins = sorted(list(set(pre_bins + post_bins)))
    
    # Calculate bin centers with special handling for SAH transition
    bin_centers = []
    for i in range(len(time_bins)-1):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        if bin_start == 0:
            bin_centers.append(0)
        else:
            bin_centers.append((bin_start + bin_end) / 2)
    
    # Calculate statistics for each bin
    results = []
    ldf_variables = ['ldf_left_adj', 'ldf_right_adj']
    
    for i, bin_center in enumerate(bin_centers):
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        
        bin_mask = (ldf_data['time_seconds'] >= bin_start) & (ldf_data['time_seconds'] < bin_end)
        bin_data = ldf_data[bin_mask]
        
        if len(bin_data) > 0:
            bin_stats = {
                'time_minutes': bin_center / 60,
                'time_seconds': bin_center,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'n_animals': bin_data['animal_id'].nunique(),
                'n_observations': len(bin_data)
            }
            
            for var in ldf_variables:
                if var in bin_data.columns:
                    var_data = bin_data[var].dropna()
                    
                    if len(var_data) > 0:
                        bin_stats[f'{var}_median'] = np.median(var_data)
                        bin_stats[f'{var}_q25'] = np.percentile(var_data, 25)
                        bin_stats[f'{var}_q75'] = np.percentile(var_data, 75)
                        bin_stats[f'{var}_iqr'] = np.percentile(var_data, 75) - np.percentile(var_data, 25)
                        bin_stats[f'{var}_std'] = np.std(var_data)
                        bin_stats[f'{var}_n'] = len(var_data)
                    else:
                        for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                            bin_stats[f'{var}_{stat}'] = np.nan
                        bin_stats[f'{var}_n'] = 0
                else:
                    for stat in ['median', 'q25', 'q75', 'iqr', 'std']:
                        bin_stats[f'{var}_{stat}'] = np.nan
                    bin_stats[f'{var}_n'] = 0
            
            results.append(bin_stats)
    
    return pd.DataFrame(results)


def create_ldf_temporal_plot_normalized(ldf_stats):
    """Create temporal evolution plot for normalized LDF data"""
    
    if ldf_stats is None or len(ldf_stats) == 0:
        print("No normalized LDF statistics available for plotting")
        return None
    
    # Filter out time bins with insufficient data
    ldf_stats = ldf_stats[ldf_stats['n_observations'] >= 10].copy()
    
    print(f"Plotting normalized LDF with {len(ldf_stats)} time bins")
    
    # Create figure with subplots for left and right LDF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Colors for LDF sides
    colors = {'ldf_left_adj': 'red', 'ldf_right_adj': 'blue'}
    
    # Helper function to plot normalized LDF with IQR error bands
    def plot_normalized_ldf_with_iqr(ax, time, median, var, color, label):
        time_shifted = time + 1
        ax.plot(time_shifted, median, color=color, linewidth=2.5, label=label, alpha=0.9)
        lower = ldf_stats[f'{var}_q25']
        upper = ldf_stats[f'{var}_q75']
        ax.fill_between(time_shifted, lower, upper, alpha=0.25, color=color)
    
    # Plot 1: Left LDF (SAH side) - normalized
    if 'ldf_left_adj_median' in ldf_stats.columns:
        plot_normalized_ldf_with_iqr(ax1, ldf_stats['time_minutes'], ldf_stats['ldf_left_adj_median'], 
                                     'ldf_left_adj', colors['ldf_left_adj'], 'LDF Left (SAH side)')
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Baseline (0)')
    ax1.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax1.set_ylabel('LDF Left', fontsize=18, fontweight='bold')
    ax1.set_title('Laser Doppler Flowmetry - Left Hemisphere (SAH Side)', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop={'size': 14})
    ax1.tick_params(axis='both', labelsize=14)
    
    # Plot 2: Right LDF (control side) - normalized
    if 'ldf_right_adj_median' in ldf_stats.columns:
        plot_normalized_ldf_with_iqr(ax2, ldf_stats['time_minutes'], ldf_stats['ldf_right_adj_median'], 
                                     'ldf_right_adj', colors['ldf_right_adj'], 'LDF Right (control side)')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAH Induction')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Baseline (0)')
    ax2.axvspan(0, 15, alpha=0.3, color='gray', label='Excluded Period (0-15 min)')
    ax2.set_xlabel('Time (minutes)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('LDF Right', fontsize=18, fontweight='bold')
    ax2.set_title('Laser Doppler Flowmetry - Right Hemisphere (Control Side)', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop={'size': 14})
    ax2.tick_params(axis='both', labelsize=14)
    
    # Add main title
  #  fig.suptitle('Bilateral Laser Doppler Flowmetry During Experimental SAH (Median ± IQR)', 
  #               fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    
    return fig


# Apply the normalization to your existing LDF data
if 'full_ldf_data' in locals() and full_ldf_data is not None:
    print("\n" + "="*70)
    print("APPLYING LDF BASELINE NORMALIZATION")
    print("="*70)
    
    # Normalize LDF data to baseline (5-minute window before SAH)
    normalized_ldf_data, baseline_summary = normalize_ldf_to_baseline(full_ldf_data, baseline_window_seconds=300)
    
    if normalized_ldf_data is not None:
        # Create time-binned statistics for normalized data
        print(f"\nCalculating normalized LDF time-binned statistics...")
        normalized_ldf_stats = create_ldf_summary_stats_normalized(normalized_ldf_data, bin_size_seconds=30)
        
        if normalized_ldf_stats is not None:
            print(f"Created {len(normalized_ldf_stats)} normalized LDF time bins")
            
            # Create normalized LDF temporal plots
            print(f"\nCreating normalized LDF temporal evolution plots...")
            normalized_ldf_fig = create_ldf_temporal_plot_normalized(normalized_ldf_stats)
            
            print(f"\n✅ LDF baseline normalization and plotting complete!")
            print(f"Generated baseline-normalized bilateral LDF plots")
            print(f"Y-axis now shows 'PU change from baseline' for proper cross-animal comparison")
        else:
            print("Failed to create normalized LDF statistics")
    else:
        print("Failed to normalize LDF data")
else:
    print("No LDF data available for normalization")
    print("Please run the LDF data loading section first")
    
