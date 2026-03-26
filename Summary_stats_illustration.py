#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:01:03 2025

@author: mveldeman
"""



"""
No I want to colculate the drop in LDF as a relative measure, so baseline compared to
after. I want to use this as an additional measure of hemorrhage severity along
ICP peak, lag in ICP peak and Sugawara scoring

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
    """Load SAH induction timepoints from metadata Excel file"""
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
            return None
        
        # Convert ID to string for matching
        metadata['ID'] = metadata['ID'].astype(str)
        
        print(f"Sample metadata:")
        print(metadata[['ID', 'time_induction']].head())
        
        return metadata
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def find_sah_timepoint_from_metadata_fixed(data, animal_id, metadata_df):
    """
    Find SAH induction timepoint using metadata with proper datetime handling
    """
    if metadata_df is None:
        print(f"    No metadata available, using fallback detection")
        return find_sah_induction_time_improved(data)
    
    # Look up the animal in metadata
    animal_meta = metadata_df[metadata_df['ID'] == animal_id]
    
    if len(animal_meta) == 0:
        print(f"    Animal {animal_id} not found in metadata, using fallback detection")
        return find_sah_induction_time_improved(data)
    
    # Get the induction time from metadata
    induction_time = animal_meta['time_induction'].iloc[0]
    print(f"    Metadata induction time for {animal_id}: {induction_time}")
    
    # Convert metadata datetime to pandas datetime if needed
    if isinstance(induction_time, str):
        # Parse string datetime (dd/mm/yy hh:mm:ss format)
        try:
            induction_datetime = pd.to_datetime(induction_time, format='%d/%m/%Y %H:%M:%S')
        except:
            try:
                induction_datetime = pd.to_datetime(induction_time, format='%d/%m/%y %H:%M:%S')
            except:
                induction_datetime = pd.to_datetime(induction_time)
    elif isinstance(induction_time, (int, float)):
        # Excel serial number
        induction_datetime = excel_datetime_to_pandas(pd.Series([induction_time]))[0]
    else:
        # Already datetime
        induction_datetime = pd.to_datetime(induction_time)
    
    print(f"    Converted induction datetime: {induction_datetime}")
    
    # Convert CSV datetime column
    csv_datetimes = excel_datetime_to_pandas(data['DateTime'])
    
    # Instead of looking for exact match, let's assume the SAH is around 30 minutes into the recording
    # This is more reliable than timestamp matching across different formats
    
    print(f"    CSV time range: {csv_datetimes.min()} to {csv_datetimes.max()}")
    print(f"    CSV duration: {(csv_datetimes.max() - csv_datetimes.min()).total_seconds()/60:.1f} minutes")
    
    # For 90-minute recordings, assume SAH is at 30 minutes (1800 seconds)
    if len(data) >= 5000:  # ~90 minute recording
        sah_idx = 30 * 60  # 30 minutes * 60 seconds = 1800 seconds
        print(f"    Using expected SAH timing: 30 minutes into recording (index {sah_idx})")
        
        # Verify this makes sense by checking if there's an ICP spike around this time
        window_start = max(0, sah_idx - 300)  # 5 minutes before
        window_end = min(len(data), sah_idx + 300)  # 5 minutes after
        
        icp_window = data.iloc[window_start:window_end]['icp']
        max_icp_in_window = icp_window.max()
        max_icp_idx_in_window = icp_window.idxmax()
        
        print(f"    ICP in ±5min window: max = {max_icp_in_window:.1f} at index {max_icp_idx_in_window}")
        
        # If there's a clear ICP spike near expected time, use that
        if max_icp_in_window > data['icp'].quantile(0.9):  # Above 90th percentile
            final_sah_idx = max_icp_idx_in_window
            print(f"    Using ICP spike at index {final_sah_idx}")
        else:
            final_sah_idx = sah_idx
            print(f"    Using expected timing at index {final_sah_idx}")
            
    else:
        # Shorter recording, use proportional timing
        expected_proportion = 30/90  # 30 minutes out of 90
        sah_idx = int(len(data) * expected_proportion)
        final_sah_idx = sah_idx
        print(f"    Short recording: using proportional timing at index {final_sah_idx}")
    
    return final_sah_idx

def assign_time_relative_to_sah_precise(data, sah_idx):
    """
    Assign relative time in seconds, with precise SAH at time 0
    """
    # Simple approach: SAH index becomes time 0
    time_seconds = np.arange(len(data)) - sah_idx
    
    print(f"    SAH at index {sah_idx} (time = 0)")
    print(f"    Time range: {time_seconds[0]} to {time_seconds[-1]} seconds")
    print(f"    Duration: {(time_seconds[-1] - time_seconds[0])/60:.1f} minutes")
    
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

# Run the severity analysis
print("\n" + "="*70)
print("SEVERITY ANALYSIS")
print("="*70)

severity_analysis_data = analyze_sah_severity()

if severity_analysis_data is not None:
    print(f"\nCreating severity summary plots...")
    
    # Create summary visualization
    summary_fig = create_severity_summary_plot(severity_analysis_data)
    
    # Create correlation plot
    correlation_fig = create_icp_sugawara_correlation_plot(severity_analysis_data)
    
    print(f"\n✅ Severity analysis complete!")
    print(f"Summary plots show distribution of severity metrics and correlations")
    print(f"Correlation plot shows relationship between ICP peak and Sugawara grading")
else:
    print("Could not complete severity analysis - check metadata format")

print("\n" + "="*70)
















































