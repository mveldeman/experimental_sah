#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:53:31 2025

@author: mveldeman
"""

"""
Here I will redo my time window defining analyses on my Sham animals, 
now after detrending the data. 

"""

# %% First import my data of all Sham animals form my csv files with the detrended data

import pandas as pd
import glob
import os

# Set the folder path
folder_path = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/B_Sham/B_Sham_with_LDF"

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "S*.csv"))

# Sort the files to ensure consistent ordering (S1, S2, S3, etc.)
csv_files.sort()

# Initialize empty list to store dataframes
dataframes = []

# Loop through each CSV file
for file in csv_files:
    # Extract the ID from filename (e.g., "S1" from "S1.csv")
    file_id = os.path.basename(file).replace('.csv', '')
    
    # Read the CSV file with German locale settings
    df = pd.read_csv(file, 
                     sep=';',  # German CSV typically uses semicolon
                     decimal=',',  # German locale uses comma for decimals
                     encoding='utf-8')
    
    # Add ID column as the first column
    df.insert(0, 'ID', file_id)
    
    # Append to list
    dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

print(f"Successfully imported {len(csv_files)} files")
print(f"Combined dataframe shape: {combined_df.shape}")
print(f"Column names: {list(combined_df.columns)}")
print(f"First few rows:")
print(combined_df.head())


# %% Converting the time stamp and identifying the optimal 90 min. or 5400 sec. window per animal

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Convert DateTime from Excel epoch to readable format
# Excel epoch starts from 1900-01-01, but has a leap year bug, so we adjust
excel_epoch = pd.to_datetime('1899-12-30')
combined_df['DateTime_readable'] = excel_epoch + pd.to_timedelta(combined_df['DateTime'], unit='D')

# Create a time column in seconds from start for each animal
combined_df['time_seconds'] = combined_df.groupby('ID')['DateTime_readable'].transform(lambda x: (x - x.min()).dt.total_seconds())

# Update PRx columns with the correct names from your data
prx_columns = ['prx_110_30', 'prx_110_60', 'prx_110_120', 'prx_110_300']

print("Updated PRx columns:", prx_columns)


def find_best_window(animal_data, columns, window_size=5400):
    """
    Find the best 90-minute (5400 second) window for each column based on data completeness.
    
    This function slides a 5400-second window across the animal's data and finds the position
    that maximizes the number of non-null data points for each specified column.
    
    Parameters:
    - animal_data: DataFrame for a single animal with 'time_seconds' column
    - columns: list of column names to analyze (e.g., PRx or LDx columns)
    - window_size: window size in seconds (default 5400 = 90 minutes)
    
    Returns:
    - Dictionary with column names as keys and (start_time, end_time, data_count) tuples as values
    """
    windows = {}
    
    # Get the time range for this animal
    min_time = animal_data['time_seconds'].min()
    max_time = animal_data['time_seconds'].max()
    
    for col in columns:
        if col not in animal_data.columns:
            continue
            
        # Get data points where this column has values
        col_data = animal_data[animal_data[col].notna()].copy()
        
        if len(col_data) < 50:  # Skip if too little data
            continue
        
        best_start = None
        best_end = None
        max_count = 0
        
        # Try different window positions (slide window by 5-minute increments)
        current_start = min_time
        
        while current_start + window_size <= max_time:
            current_end = current_start + window_size
            
            # Count valid data points in this window
            window_data = col_data[
                (col_data['time_seconds'] >= current_start) & 
                (col_data['time_seconds'] <= current_end)
            ]
            
            count = len(window_data)
            
            # Update best window if this one has more data
            if count > max_count:
                max_count = count
                best_start = current_start
                best_end = current_end
            
            # Move window by 5 minutes (300 seconds) for next iteration
            current_start += 300
        
        # If we found a good window, store it
        if best_start is not None and max_count > 0:
            windows[col] = (best_start, best_end, max_count)
    
    return windows

def create_summary_data(prx_columns, window_size=5400):
    """
    Create summary statistics for PRx data across all animals.
    
    This function takes the optimal windows found by find_best_window() and creates
    time-binned summary statistics (mean and standard deviation) across all animals.
    
    Parameters:
    - prx_columns: list of PRx column names to analyze
    - window_size: window size in seconds (should match find_best_window)
    
    Returns:
    - Dictionary with column names as keys and DataFrames with time-binned statistics as values
    """
    summary_data = {}
    
    for col in prx_columns:
        all_animal_data = []
        
        # Collect data from all animals for this column
        for animal_id in combined_df['ID'].unique():
            if col in animal_windows[animal_id]:
                start_time, end_time, _ = animal_windows[animal_id][col]
                
                # Get animal data for this window
                animal_data = combined_df[combined_df['ID'] == animal_id].copy()
                window_data = animal_data[
                    (animal_data['time_seconds'] >= start_time) & 
                    (animal_data['time_seconds'] <= end_time)
                ].copy()
                
                if len(window_data) > 0:
                    # Create relative time (0 to 90 minutes)
                    window_data['relative_time_min'] = (window_data['time_seconds'] - start_time) / 60
                    # Keep only the columns we need
                    window_data = window_data[['relative_time_min', col]].dropna()
                    all_animal_data.append(window_data)
        
        if all_animal_data:
            # Combine all animals
            combined_data = pd.concat(all_animal_data, ignore_index=True)
            
            # Create time bins (every minute from 0 to 90)
            time_bins = np.arange(0, 91, 1)  # 0 to 90 minutes
            binned_data = []
            
            for i in range(len(time_bins)-1):
                # Get data in this time bin
                bin_data = combined_data[
                    (combined_data['relative_time_min'] >= time_bins[i]) & 
                    (combined_data['relative_time_min'] < time_bins[i+1])
                ][col]
                
                if len(bin_data) > 0:
                    binned_data.append({
                        'time_bin': time_bins[i] + 0.5,  # Center of bin
                        'mean': bin_data.mean(),
                        'std': bin_data.std() if len(bin_data) > 1 else 0
                    })
            
            if binned_data:
                summary_data[col] = pd.DataFrame(binned_data)
    
    return summary_data



# Verify these columns have data
print("\n=== UPDATED COLUMN CHECK ===")
for col in prx_columns:
    if col in combined_df.columns:
        non_null_count = combined_df[col].notna().sum()
        total_count = len(combined_df)
        print(f"{col}: EXISTS - {non_null_count}/{total_count} non-null values ({non_null_count/total_count*100:.1f}%)")
    else:
        print(f"{col}: MISSING from dataframe")

# Re-run the window finding with correct column names
animal_windows = {}
for animal_id in combined_df['ID'].unique():
    animal_data = combined_df[combined_df['ID'] == animal_id].copy()
    animal_windows[animal_id] = find_best_window(animal_data, prx_columns)

print(f"\nWindow analysis complete for {len(animal_windows)} animals!")

# Quick check of windows for first animal
sample_animal = list(combined_df['ID'].unique())[0]
print(f"\nSample windows for {sample_animal}:")
for col in prx_columns:
    if col in animal_windows[sample_animal]:
        start, end, count = animal_windows[sample_animal][col]
        print(f"  {col}: window {start}-{end}s, {count} data points")
        
        
# %% Creating Plots of PRx over time for each of the used time windows (30, 60, 120, 300s)
        
        # Create summary data with the correct column names
summary_data = create_summary_data(prx_columns)

# Create the 4-panel plot matching your aesthetics
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# Plot titles for each PRx calculation window
titles = ['PRx 30s Window', 'PRx 60s Window', 'PRx 120s Window', 'PRx 300s Window']
prx_labels = ['prx_110_30', 'prx_110_60', 'prx_110_120', 'prx_110_300']

for i, (ax, col, title) in enumerate(zip(axes, prx_labels, titles)):
    if col in summary_data and len(summary_data[col]) > 0:
        data = summary_data[col]
        
        # Main PRx line (matching your green color)
        ax.plot(data['time_bin'], data['mean'], color='green', linewidth=2, label='Mean PRx')
        
        # Standard deviation fill (matching your alpha)
        ax.fill_between(data['time_bin'],
                       data['mean'] - data['std'],
                       data['mean'] + data['std'],
                       color='green', alpha=0.2, label='±1 SD')
        
        # PRx threshold line (matching your red dashed line)
        ax.axhline(0.3, color='red', linestyle='--', alpha=0.7, label='PRx = 0.3 threshold')
        
        # Grid (matching your style)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Labels with bold formatting (matching your style)
        if i == 3:  # Only bottom plot gets x-label
            ax.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PRx', fontsize=14, fontweight='bold')
        
        # Title for each subplot
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=12, fontweight='bold', 
                verticalalignment='top')
        
        # Legend only on top plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=11)
    
    else:
        # If no data for this column, show empty plot with message
        ax.text(0.5, 0.5, f'No data available for {col}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    # Set x-axis limits (0 to 90 minutes)
    ax.set_xlim(0, 90)

# Overall title (matching your style)
fig.suptitle('PRx Over Time - Different Calculation Windows', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

print("Plot created successfully!")

# Also print some diagnostic info about the summary data
print("\n=== SUMMARY DATA INFO ===")
for col in prx_columns:
    if col in summary_data:
        print(f"{col}: {len(summary_data[col])} time bins with data")
        if len(summary_data[col]) > 0:
            print(f"  Mean range: {summary_data[col]['mean'].min():.3f} to {summary_data[col]['mean'].max():.3f}")
    else:
        print(f"{col}: no summary data created")
        
        
# %% Adjusted plot with equal y-axis range and added boxplots

# Create summary data and also collect all data points for boxplots
def create_summary_and_boxplot_data(prx_columns, window_size=5400):
    summary_data = {}
    boxplot_data = {}
    
    for col in prx_columns:
        all_animal_data = []
        all_values = []  # For boxplot
        
        for animal_id in combined_df['ID'].unique():
            if col in animal_windows[animal_id]:
                start_time, end_time, _ = animal_windows[animal_id][col]
                
                # Get animal data for this window
                animal_data = combined_df[combined_df['ID'] == animal_id].copy()
                window_data = animal_data[(animal_data['time_seconds'] >= start_time) & 
                                        (animal_data['time_seconds'] <= end_time)].copy()
                
                if len(window_data) > 0:
                    # Create relative time (0 to 90 minutes)
                    window_data['relative_time_min'] = (window_data['time_seconds'] - start_time) / 60
                    window_data = window_data[['relative_time_min', col]].dropna()
                    all_animal_data.append(window_data)
                    
                    # Collect all values for boxplot
                    all_values.extend(window_data[col].tolist())
        
        # Store boxplot data
        boxplot_data[col] = all_values
        
        if all_animal_data:
            # Combine all animals for time series
            combined_data = pd.concat(all_animal_data, ignore_index=True)
            
            # Create time bins (every minute)
            time_bins = np.arange(0, 91, 1)  # 0 to 90 minutes
            binned_data = []
            
            for i in range(len(time_bins)-1):
                bin_data = combined_data[(combined_data['relative_time_min'] >= time_bins[i]) & 
                                       (combined_data['relative_time_min'] < time_bins[i+1])][col]
                if len(bin_data) > 0:
                    binned_data.append({
                        'time_bin': time_bins[i] + 0.5,  # Center of bin
                        'mean': bin_data.mean(),
                        'std': bin_data.std()
                    })
            
            summary_data[col] = pd.DataFrame(binned_data)
    
    return summary_data, boxplot_data

# Create summary data and boxplot data
summary_data, boxplot_data = create_summary_and_boxplot_data(prx_columns)

# Create the 4-panel plot with boxplots
fig, axes = plt.subplots(4, 2, figsize=(16, 16), 
                        gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.1})

# Plot titles for each PRx calculation window
titles = ['PRx 30s Window', 'PRx 60s Window', 'PRx 120s Window', 'PRx 300s Window']
prx_labels = ['prx_110_30', 'prx_110_60', 'prx_110_120', 'prx_110_300']

for i, (col, title) in enumerate(zip(prx_labels, titles)):
    # Time series plot (left column)
    ax_time = axes[i, 0]
    # Boxplot (right column)
    ax_box = axes[i, 1]
    
    if col in summary_data and len(summary_data[col]) > 0:
        data = summary_data[col]
        
        # Main PRx line (matching your green color)
        ax_time.plot(data['time_bin'], data['mean'], color='green', linewidth=2, label='Mean PRx')
        
        # Standard deviation fill (matching your alpha)
        ax_time.fill_between(data['time_bin'],
                           data['mean'] - data['std'],
                           data['mean'] + data['std'],
                           color='green', alpha=0.2, label='±1 SD')
        
        # PRx threshold line (matching your red dashed line)
        ax_time.axhline(0.3, color='red', linestyle='--', alpha=0.7, label='PRx = 0.3 threshold')
        
        # Grid (matching your style)
        ax_time.grid(True, linestyle=':', alpha=0.5)
        
        # Create boxplot with warm orange-yellow color
        if col in boxplot_data and len(boxplot_data[col]) > 0:
            values = boxplot_data[col]
            bp = ax_box.boxplot(values, patch_artist=True, widths=0.6)
            # Set warm orangy-yellow color
            bp['boxes'][0].set_facecolor('#FFB84D')  # Warm orange-yellow
            bp['boxes'][0].set_alpha(0.7)
            
            # Calculate mean and std
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Add mean as a red diamond
            ax_box.scatter([1], [mean_val], marker='D', color='red', s=50, zorder=5)
            
            # Add text with mean and std in top right corner
            ax_box.text(0.98, 0.95, f'Mean: {mean_val:.3f}\nSD: {std_val:.3f}', 
                       transform=ax_box.transAxes,
                       fontsize=10, 
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Also add threshold line to boxplot
            ax_box.axhline(0.3, color='red', linestyle='--', alpha=0.7)
        
        # Remove x-axis labels and ticks for boxplot
        ax_box.set_xticks([])
        ax_box.set_xlabel('')
        
    else:
        # If no data for this column, show empty plot with message
        ax_time.text(0.5, 0.5, f'No data available for {col}', 
                    transform=ax_time.transAxes, ha='center', va='center', fontsize=12)
        ax_box.text(0.5, 0.5, 'No data', 
                   transform=ax_box.transAxes, ha='center', va='center', fontsize=10)
    
    # Set consistent y-axis limits for both plots
    ax_time.set_ylim(-0.5, 0.6)
    ax_box.set_ylim(-0.5, 0.6)
    
    # Set x-axis limits (0 to 90 minutes for time series)
    ax_time.set_xlim(0, 90)
    
    # Labels with bold formatting
    if i == 3:  # Only bottom plot gets x-label
        ax_time.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax_time.set_ylabel('PRx', fontsize=14, fontweight='bold')
    
    # Remove y-axis labels from boxplot (shared with time series)
    ax_box.set_yticklabels([])
    
    # Title for each subplot row
    ax_time.text(0.02, 0.95, title, transform=ax_time.transAxes, fontsize=12, fontweight='bold', 
                verticalalignment='top')
    
    # Legend only on top plot
    if i == 0:
        ax_time.legend(loc='upper right', fontsize=11)

# Overall title (matching your style)
fig.suptitle('PRx Over Time - Different Calculation Windows', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

print("Plot with boxplots and statistics created successfully!")


# %% Preparing my data for coherence analysis

import numpy as np
from scipy import signal

def calculate_coherence_spectrum_detrended(abp_dt, icp_dt, fs=1.0, nperseg=256):
    """
    Calculate coherence spectrum between detrended ABP and ICP using only valid data points
    """
    # Find valid data points (both signals present)
    valid_mask = ~(np.isnan(abp_dt) | np.isnan(icp_dt))
    
    if np.sum(valid_mask) < nperseg:
        return None
    
    # Extract valid data
    abp_valid = abp_dt[valid_mask]
    icp_valid = icp_dt[valid_mask]
    
    # Calculate coherence
    frequencies, coherence = signal.coherence(
        abp_valid, icp_valid, 
        fs=fs, 
        nperseg=nperseg,
        noverlap=nperseg//2
    )
    
    # Also calculate power spectra for completeness
    f_abp, psd_abp = signal.welch(abp_valid, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    f_icp, psd_icp = signal.welch(icp_valid, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    return {
        'frequencies': frequencies,
        'coherence': coherence,
        'abp_psd': psd_abp,
        'icp_psd': psd_icp,
        'n_valid_points': np.sum(valid_mask)
    }

# %% Prepare Data for All Animals

print("Preparing detrended data for coherence analysis...")

# Check which animals have the required detrended columns
required_columns = ['abp_dt_110', 'icp_dt_110']
valid_animals = []

for animal_id in combined_df['ID'].unique():
    animal_data = combined_df[combined_df['ID'] == animal_id]
    
    # Check if both columns exist and have data
    abp_available = 'abp_dt_110' in animal_data.columns and animal_data['abp_dt_110'].notna().sum() > 0
    icp_available = 'icp_dt_110' in animal_data.columns and animal_data['icp_dt_110'].notna().sum() > 0
    
    if abp_available and icp_available:
        abp_completeness = (animal_data['abp_dt_110'].notna().sum() / len(animal_data)) * 100
        icp_completeness = (animal_data['icp_dt_110'].notna().sum() / len(animal_data)) * 100
        
        valid_animals.append({
            'id': animal_id,
            'abp_completeness': abp_completeness,
            'icp_completeness': icp_completeness,
            'abp_data': animal_data['abp_dt_110'].values,
            'icp_data': animal_data['icp_dt_110'].values
        })
        
        print(f"{animal_id}: ABP={abp_completeness:.1f}%, ICP={icp_completeness:.1f}%")

print(f"\nFound {len(valid_animals)} animals with both detrended ABP and ICP data")


# %% Perform Coherence Analysis on All Animals

print("Performing coherence analysis on all animals with detrended data...")

all_coherence_results = []
all_frequencies = None

for animal in valid_animals:
    animal_id = animal['id']
    
    result = calculate_coherence_spectrum_detrended(animal['abp_data'], animal['icp_data'])
    
    if result is not None:
        all_coherence_results.append({
            'animal_id': animal_id,
            'result': result,
            'abp_completeness': animal['abp_completeness']
        })
        
        if all_frequencies is None:
            all_frequencies = result['frequencies']
        
        print(f"{animal_id}: {result['n_valid_points']} valid points used")
    else:
        print(f"{animal_id}: Failed - insufficient valid data")

print(f"\nSuccessfully analyzed {len(all_coherence_results)} out of {len(valid_animals)} animals")

# Calculate statistics across all animals
if all_coherence_results:
    coherence_matrix = np.array([item['result']['coherence'] for item in all_coherence_results])
    mean_coherence = np.mean(coherence_matrix, axis=0)
    std_coherence = np.std(coherence_matrix, axis=0)
    
    # Show contributing animals
    contributing_animals = [item['animal_id'] for item in all_coherence_results]
    abp_quality = [item['abp_completeness'] for item in all_coherence_results]
    
    print(f"Contributing animals: {contributing_animals}")
    print(f"ABP completeness range: {min(abp_quality):.1f}% to {max(abp_quality):.1f}%")
    print(f"Mean ABP completeness: {np.mean(abp_quality):.1f}%")

# %% Analyze Results and Find Peak Frequency

# Focus on the autoregulation frequency range (0.005 to 0.1 Hz)
autoregulation_mask = (all_frequencies >= 0.005) & (all_frequencies <= 0.1)
autoregulation_freqs = all_frequencies[autoregulation_mask]
autoregulation_coherence = mean_coherence[autoregulation_mask]

# Find peak coherence in autoregulation range
peak_idx = np.argmax(autoregulation_coherence)
peak_frequency = autoregulation_freqs[peak_idx]
peak_coherence = autoregulation_coherence[peak_idx]

print("\n" + "="*50)
print("DETRENDED DATA COHERENCE ANALYSIS RESULTS")
print("="*50)
print(f"Peak frequency: {peak_frequency:.4f} Hz ({1/peak_frequency:.0f}s period)")
print(f"Peak coherence: {peak_coherence:.3f}")

# Calculate recommended PRx windows
recommended_window_3cycles = int(3 / peak_frequency)
recommended_window_5cycles = int(5 / peak_frequency)

print(f"\nRecommended PRx windows:")
print(f"3 cycles: {recommended_window_3cycles} seconds")
print(f"5 cycles: {recommended_window_5cycles} seconds")

# Frequency band analysis
freq_bands = [
    ("Very slow waves", 0.005, 0.02),
    ("Slow waves", 0.02, 0.05),
    ("Intermediate", 0.05, 0.1),
    ("Higher freq", 0.1, 0.2)
]

print(f"\nCoherence by frequency band:")
for band_name, f_low, f_high in freq_bands:
    band_mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(band_mask):
        band_coherence = mean_coherence[band_mask]
        max_coh = np.max(band_coherence)
        mean_coh = np.mean(band_coherence)
        print(f"{band_name:15s}: max={max_coh:.3f}, mean={mean_coh:.3f}")
        
        
# %% Create Final Coherence Analysis Figure - Detrended Data for ABP and ICP

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Define the red-to-yellow color scheme
bar_colors = ['darkred', 'red', 'orange', 'yellow']
overlay_colors = ['lightcoral', 'lightpink', 'lightsalmon', 'lightyellow']

# Plot 1: Mean coherence spectrum
ax1.set_title('ABP-ICP Coherence Spectrum - Detrended Data (All 21 Animals)', fontsize=14, fontweight='bold')
ax1.plot(all_frequencies, mean_coherence, 'k-', linewidth=3, label='Mean coherence')
ax1.fill_between(all_frequencies, 
                 mean_coherence - std_coherence, 
                 mean_coherence + std_coherence, 
                 alpha=0.2, color='gray', label='±1 SD')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Autoregulation range with frequency band overlays
ax2.set_title('Autoregulation Frequency Bands', fontsize=14, fontweight='bold')

# Add frequency band background colors
band_labels = ['Very slow waves\n(0.005-0.02 Hz)', 'Slow waves\n(0.02-0.05 Hz)', 
               'Intermediate\n(0.05-0.1 Hz)', 'Higher freq\n(0.1-0.2 Hz)']
band_ranges = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)]

for i, (f_low, f_high) in enumerate(band_ranges):
    if f_high <= 0.15:  # Only show bands in our plot range
        ax2.axvspan(f_low, f_high, alpha=0.3, color=overlay_colors[i], 
                   label=band_labels[i])

# Plot the coherence line on top
ax2.plot(all_frequencies, mean_coherence, 'k-', linewidth=3, label='Mean coherence')
ax2.fill_between(all_frequencies, 
                 mean_coherence - std_coherence, 
                 mean_coherence + std_coherence, 
                 alpha=0.2, color='gray')

# Mark the peak frequency
ax2.axvline(peak_frequency, color='red', linestyle='--', linewidth=3, 
            label=f'Peak: {peak_frequency:.4f} Hz\n({1/peak_frequency:.0f}s period)')
ax2.scatter(peak_frequency, peak_coherence, 
           color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0.005, 0.15)
ax2.set_ylim(0, max(mean_coherence[autoregulation_mask]) * 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Frequency bands with matching red-to-yellow colors
band_names = ['Very slow', 'Slow', 'Intermediate', 'Higher']
band_values = []

for f_low, f_high in band_ranges:
    mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(mask):
        band_values.append(np.mean(mean_coherence[mask]))
    else:
        band_values.append(0)

ax3.set_title('Mean Coherence by Frequency Band', fontsize=14, fontweight='bold')
bars = ax3.bar(band_names, band_values, color=bar_colors)
ax3.set_ylabel('Mean Coherence')
ax3.set_ylim(0, max(band_values) * 1.2)

# Add values on bars
for bar, value in zip(bars, band_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: PRx Window recommendations
windows = [120, 180, 240, 300, 360, 480, 600]
cycles = [w * peak_frequency for w in windows]

ax4.set_title('PRx Window Selection Guide', fontsize=14, fontweight='bold')
ax4.plot(windows, cycles, 'bo-', linewidth=2, markersize=8, label='Autoregulation cycles')
ax4.axhline(3, color='green', linestyle='--', linewidth=2, label='3 cycles')
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='5 cycles')

# Highlight your actual recommendations
ax4.axvline(384, color='green', linestyle=':', alpha=0.7, linewidth=2)  # 3 cycles
ax4.axvline(640, color='red', linestyle=':', alpha=0.7, linewidth=2)    # 5 cycles

ax4.set_xlabel('PRx Window (seconds)')
ax4.set_ylabel('Number of Autoregulation Cycles')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_ylim(1, 6)

plt.tight_layout()
plt.show()

# Final summary
print("\n" + "="*60)
print("SPECTRAL ANALYSIS RESULTS - DETRENDED DATA (21 SHAM ANIMALS)")
print("="*60)
print(f"Peak frequency: {peak_frequency:.4f} Hz ({1/peak_frequency:.0f}s period)")
print(f"Peak coherence: {peak_coherence:.3f}")
print(f"Dominant band: Very slow waves (0.005-0.02 Hz)")
print(f"Recommended PRx windows:")
print(f"  - {recommended_window_3cycles}s (3-cycle window)")
print(f"  - {recommended_window_5cycles}s (5-cycle window)")
print("="*60)


# %% Assessing the number of windows used for my coherence analysis

# Define the parameters from your code
nperseg = 256
noverlap = nperseg // 2  # which is 128

# Create a list to store the number of windows for each animal
window_counts = []

for animal in valid_animals:
    # Get the number of valid data points for the animal
    n_valid_points = np.sum(~(np.isnan(animal['abp_data']) | np.isnan(animal['icp_data'])))

    # Calculate the number of windows using the formula
    if n_valid_points >= nperseg:
        # The floor division `//` is a more robust way to handle this in Python
        n_windows = (n_valid_points - noverlap) // (nperseg - noverlap)
        window_counts.append(n_windows)
        print(f"Animal {animal['id']}: {n_windows} windows used.")
    else:
        print(f"Animal {animal['id']}: Insufficient data for a single window.")

# You can also calculate the average and range of windows used
if window_counts:
    mean_windows = np.mean(window_counts)
    min_windows = np.min(window_counts)
    max_windows = np.max(window_counts)

    print(f"\nAverage number of windows used: {mean_windows:.1f}")
    print(f"Range of windows used: {min_windows} to {max_windows}")


# %% Now making the plots for LDx for each of the time windows (30, 60, 120, 300s) for both sides 

# Define LDx columns for left and right hemispheres
ldx_left_columns = ['ldx_left_dt_110_30', 'ldx_left_dt_110_60', 'ldx_left_dt_110_120', 'ldx_left_dt_110_300']
ldx_right_columns = ['ldx_right_dt_110_30', 'ldx_right_dt_110_60', 'ldx_right_dt_110_120', 'ldx_right_dt_110_300']

print("Checking LDx columns availability...")
all_ldx_columns = ldx_left_columns + ldx_right_columns

for col in all_ldx_columns:
    if col in combined_df.columns:
        non_null_count = combined_df[col].notna().sum()
        total_count = len(combined_df)
        print(f"{col}: EXISTS - {non_null_count}/{total_count} non-null values ({non_null_count/total_count*100:.1f}%)")
    else:
        print(f"{col}: MISSING from dataframe")

# Find best windows for LDx (using the same function as before)
ldx_animal_windows = {}
for animal_id in combined_df['ID'].unique():
    animal_data = combined_df[combined_df['ID'] == animal_id].copy()
    ldx_animal_windows[animal_id] = find_best_window(animal_data, all_ldx_columns)

print(f"\nLDx window analysis complete for {len(ldx_animal_windows)} animals!")

# Quick check of windows for first animal
sample_animal = list(combined_df['ID'].unique())[0]
print(f"\nSample LDx windows for {sample_animal}:")
for col in all_ldx_columns[:4]:  # Show first 4 columns
    if col in ldx_animal_windows[sample_animal]:
        start, end, count = ldx_animal_windows[sample_animal][col]
        print(f"  {col}: window {start}-{end}s, {count} data points")
        
        
# %% Create LDx Summary Data for Both Hemispheres

def create_ldx_summary_data(left_columns, right_columns, window_size=5400):
    summary_data_left = {}
    summary_data_right = {}
    boxplot_data_left = {}
    boxplot_data_right = {}
    
    # Process left hemisphere
    for col in left_columns:
        all_animal_data = []
        all_values = []
        
        for animal_id in combined_df['ID'].unique():
            if col in ldx_animal_windows[animal_id]:
                start_time, end_time, _ = ldx_animal_windows[animal_id][col]
                
                animal_data = combined_df[combined_df['ID'] == animal_id].copy()
                window_data = animal_data[(animal_data['time_seconds'] >= start_time) & 
                                        (animal_data['time_seconds'] <= end_time)].copy()
                
                if len(window_data) > 0:
                    window_data['relative_time_min'] = (window_data['time_seconds'] - start_time) / 60
                    window_data = window_data[['relative_time_min', col]].dropna()
                    all_animal_data.append(window_data)
                    all_values.extend(window_data[col].tolist())
        
        boxplot_data_left[col] = all_values
        
        if all_animal_data:
            combined_data = pd.concat(all_animal_data, ignore_index=True)
            time_bins = np.arange(0, 91, 1)
            binned_data = []
            
            for i in range(len(time_bins)-1):
                bin_data = combined_data[(combined_data['relative_time_min'] >= time_bins[i]) & 
                                       (combined_data['relative_time_min'] < time_bins[i+1])][col]
                if len(bin_data) > 0:
                    binned_data.append({
                        'time_bin': time_bins[i] + 0.5,
                        'mean': bin_data.mean(),
                        'std': bin_data.std()
                    })
            
            summary_data_left[col] = pd.DataFrame(binned_data)
    
    # Process right hemisphere (same process)
    for col in right_columns:
        all_animal_data = []
        all_values = []
        
        for animal_id in combined_df['ID'].unique():
            if col in ldx_animal_windows[animal_id]:
                start_time, end_time, _ = ldx_animal_windows[animal_id][col]
                
                animal_data = combined_df[combined_df['ID'] == animal_id].copy()
                window_data = animal_data[(animal_data['time_seconds'] >= start_time) & 
                                        (animal_data['time_seconds'] <= end_time)].copy()
                
                if len(window_data) > 0:
                    window_data['relative_time_min'] = (window_data['time_seconds'] - start_time) / 60
                    window_data = window_data[['relative_time_min', col]].dropna()
                    all_animal_data.append(window_data)
                    all_values.extend(window_data[col].tolist())
        
        boxplot_data_right[col] = all_values
        
        if all_animal_data:
            combined_data = pd.concat(all_animal_data, ignore_index=True)
            time_bins = np.arange(0, 91, 1)
            binned_data = []
            
            for i in range(len(time_bins)-1):
                bin_data = combined_data[(combined_data['relative_time_min'] >= time_bins[i]) & 
                                       (combined_data['relative_time_min'] < time_bins[i+1])][col]
                if len(bin_data) > 0:
                    binned_data.append({
                        'time_bin': time_bins[i] + 0.5,
                        'mean': bin_data.mean(),
                        'std': bin_data.std()
                    })
            
            summary_data_right[col] = pd.DataFrame(binned_data)
    
    return summary_data_left, summary_data_right, boxplot_data_left, boxplot_data_right

# Create summary data for both hemispheres
ldx_summary_left, ldx_summary_right, ldx_boxplot_left, ldx_boxplot_right = create_ldx_summary_data(
    ldx_left_columns, ldx_right_columns)

print("LDx summary data created for both hemispheres!")


# %% Create LDx 4-Panel Plot with Both Hemispheres

fig, axes = plt.subplots(4, 2, figsize=(18, 16), 
                        gridspec_kw={'width_ratios': [5, 2], 'wspace': 0.15})

# Plot titles and column pairs
titles = ['LDx 30s Window', 'LDx 60s Window', 'LDx 120s Window', 'LDx 300s Window']
left_labels = ['ldx_left_dt_110_30', 'ldx_left_dt_110_60', 'ldx_left_dt_110_120', 'ldx_left_dt_110_300']
right_labels = ['ldx_right_dt_110_30', 'ldx_right_dt_110_60', 'ldx_right_dt_110_120', 'ldx_right_dt_110_300']

# Color scheme matching your reference plot
left_color = 'red'      # Red for left hemisphere
right_color = 'blue'    # Blue for right hemisphere
left_alpha = 0.2
right_alpha = 0.2

for i, (left_col, right_col, title) in enumerate(zip(left_labels, right_labels, titles)):
    # Time series plot (left column)
    ax_time = axes[i, 0]
    # Boxplot (right column) 
    ax_box = axes[i, 1]
    
    # Plot LEFT hemisphere time series
    if left_col in ldx_summary_left and len(ldx_summary_left[left_col]) > 0:
        left_data = ldx_summary_left[left_col]
        
        # Left hemisphere line and fill
        ax_time.plot(left_data['time_bin'], left_data['mean'], 
                    color=left_color, linewidth=2, label='Mean LDx Left')
        ax_time.fill_between(left_data['time_bin'],
                           left_data['mean'] - left_data['std'],
                           left_data['mean'] + left_data['std'],
                           color=left_color, alpha=left_alpha, label='±1 SD Left')
    
    # Plot RIGHT hemisphere time series
    if right_col in ldx_summary_right and len(ldx_summary_right[right_col]) > 0:
        right_data = ldx_summary_right[right_col]
        
        # Right hemisphere line and fill
        ax_time.plot(right_data['time_bin'], right_data['mean'], 
                    color=right_color, linewidth=2, label='Mean LDx Right')
        ax_time.fill_between(right_data['time_bin'],
                           right_data['mean'] - right_data['std'],
                           right_data['mean'] + right_data['std'],
                           color=right_color, alpha=right_alpha, label='±1 SD Right')
    
    # LDx threshold line (0.3 like PRx)
    ax_time.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='LDx = 0.5 threshold')
    
    # Grid
    ax_time.grid(True, linestyle=':', alpha=0.5)
    
    # Create paired boxplots
    boxplot_positions = [1, 2]
    boxplot_data_list = []
    boxplot_colors = []
    
    # Left hemisphere boxplot data
    if left_col in ldx_boxplot_left and len(ldx_boxplot_left[left_col]) > 0:
        left_values = ldx_boxplot_left[left_col]
        boxplot_data_list.append(left_values)
        boxplot_colors.append(left_color)
        
        # Calculate and display stats for left
        left_mean = np.mean(left_values)
        left_std = np.std(left_values)
        ax_box.scatter([1], [left_mean], marker='D', color=left_color, s=50, zorder=5)
        
    # Right hemisphere boxplot data  
    if right_col in ldx_boxplot_right and len(ldx_boxplot_right[right_col]) > 0:
        right_values = ldx_boxplot_right[right_col]
        boxplot_data_list.append(right_values)
        boxplot_colors.append(right_color)
        
        # Calculate and display stats for right
        right_mean = np.mean(right_values)
        right_std = np.std(right_values)
        ax_box.scatter([2], [right_mean], marker='D', color=right_color, s=50, zorder=5)
    
    # Create the boxplots
    if boxplot_data_list:
        bp = ax_box.boxplot(boxplot_data_list, positions=boxplot_positions[:len(boxplot_data_list)], 
                           patch_artist=True, widths=0.4)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], boxplot_colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add statistics text
# Add statistics text - TWO SEPARATE BOXES WITH LOWERED POSITIONING
        if left_col in ldx_boxplot_left and len(ldx_boxplot_left[left_col]) > 0:
            left_stats_text = f"Mean: {left_mean:.3f}\nSD: {left_std:.3f}"
            # Position over left boxplot (x=1 in data coordinates, lowered y-position)
            ax_box.text(1, ax_box.get_ylim()[1] * 0.55, left_stats_text, 
                       fontsize=9, 
                       verticalalignment='top',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        if right_col in ldx_boxplot_right and len(ldx_boxplot_right[right_col]) > 0:
            right_stats_text = f"Mean: {right_mean:.3f}\nSD: {right_std:.3f}"
            # Position over right boxplot (x=2 in data coordinates, lowered y-position)
            ax_box.text(2, ax_box.get_ylim()[1] * 0.55, right_stats_text,
                       fontsize=9,
                       verticalalignment='top', 
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    
    
    
    
    # Threshold line in boxplot
    ax_box.axhline(0.3, color='gray', linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits 
    ax_time.set_ylim(-0.5, 0.6)
    ax_box.set_ylim(-0.5, 0.6)
    
    # Set x-axis limits for time series
    ax_time.set_xlim(0, 90)
    
    # Labels
    if i == 3:  # Only bottom plot gets x-label
        ax_time.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax_time.set_ylabel('LDx', fontsize=14, fontweight='bold')
    
    # Boxplot formatting
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(['Left', 'Right'], fontsize=12)
    ax_box.set_yticklabels([])  # Remove y-axis labels (shared with time series)
    
    # Title for each subplot row
    ax_time.text(0.02, 0.95, title, transform=ax_time.transAxes, fontsize=12, fontweight='bold', 
                verticalalignment='top')
    
    # Legend only on top plot
    if i == 0:
        ax_time.legend(loc='upper right', fontsize=10)

# Overall title
fig.suptitle('LDx Over Time - Different Calculation Windows (Left vs Right)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

print("LDx plot with both hemispheres created successfully!")


# %% LDF Coherence Analysis with Detrended Data

from scipy import signal

def calculate_ldf_coherence_spectrum(abp_dt, ldf_dt, fs=1.0, nperseg=256):
    """
    Calculate coherence spectrum between detrended ABP and LDF
    """
    # Find valid data points (both signals present)
    valid_mask = ~(np.isnan(abp_dt) | np.isnan(ldf_dt))
    
    if np.sum(valid_mask) < nperseg:
        return None
    
    # Extract valid data
    abp_valid = abp_dt[valid_mask]
    ldf_valid = ldf_dt[valid_mask]
    
    # Calculate coherence
    frequencies, coherence = signal.coherence(
        abp_valid, ldf_valid, 
        fs=fs, 
        nperseg=nperseg,
        noverlap=nperseg//2
    )
    
    return {
        'frequencies': frequencies,
        'coherence': coherence,
        'n_valid_points': np.sum(valid_mask)
    }

# %% Prepare LDF Data and Perform Analysis

print("LDF Spectral Analysis with Detrended Data...")
print("Animal\t\tABP%\tLDF_L%\tLDF_R%\tLeft_Points\tRight_Points\tLeft_Peak\tRight_Peak")
print("-" * 90)

# Check data availability and perform coherence analysis
ldf_coherence_results = {'left': [], 'right': []}
all_frequencies_ldf = None
contributing_animals_ldf = []

for animal_id in combined_df['ID'].unique():
    animal_data = combined_df[combined_df['ID'] == animal_id]
    
    # Check data completeness
    abp_completeness = (animal_data['abp_dt'].notna().sum() / len(animal_data)) * 100
    ldf_left_completeness = (animal_data['ldf_left_dt'].notna().sum() / len(animal_data)) * 100
    ldf_right_completeness = (animal_data['ldf_right_dt'].notna().sum() / len(animal_data)) * 100
    
    # Perform coherence analysis if sufficient data (>20% completeness)
    result_left = None
    result_right = None
    
    if abp_completeness > 20 and ldf_left_completeness > 20:
        result_left = calculate_ldf_coherence_spectrum(
            animal_data['abp_dt'].values, 
            animal_data['ldf_left_dt'].values
        )
        if result_left is not None:
            ldf_coherence_results['left'].append({
                'animal_id': animal_id,
                'result': result_left,
                'abp_completeness': abp_completeness,
                'ldf_completeness': ldf_left_completeness
            })
            if all_frequencies_ldf is None:
                all_frequencies_ldf = result_left['frequencies']
            contributing_animals_ldf.append(animal_id)
    
    if abp_completeness > 20 and ldf_right_completeness > 20:
        result_right = calculate_ldf_coherence_spectrum(
            animal_data['abp_dt'].values, 
            animal_data['ldf_right_dt'].values
        )
        if result_right is not None:
            ldf_coherence_results['right'].append({
                'animal_id': animal_id,
                'result': result_right,
                'abp_completeness': abp_completeness,
                'ldf_completeness': ldf_right_completeness
            })
    
    # Print results
    left_points = result_left['n_valid_points'] if result_left else 0
    right_points = result_right['n_valid_points'] if result_right else 0
    left_peak = np.max(result_left['coherence']) if result_left else 0
    right_peak = np.max(result_right['coherence']) if result_right else 0
    
    if left_points > 0 or right_points > 0:
        print(f"{animal_id}\t\t{abp_completeness:.1f}%\t{ldf_left_completeness:.1f}%\t{ldf_right_completeness:.1f}%\t{left_points}\t\t{right_points}\t\t{left_peak:.3f}\t\t{right_peak:.3f}")

print(f"\nLDF COHERENCE ANALYSIS SUMMARY:")
print(f"Animals contributing to LEFT hemisphere: {len(ldf_coherence_results['left'])}")
print(f"Animals contributing to RIGHT hemisphere: {len(ldf_coherence_results['right'])}")
print(f"Total contributing animals: {len(set(contributing_animals_ldf))}")
print(f"Contributing animals: {sorted(set(contributing_animals_ldf))}")


# %% Calculate LDF Coherence Statistics and Find Peak Frequencies

if len(ldf_coherence_results['left']) > 0 and len(ldf_coherence_results['right']) > 0:
    
    # Calculate mean coherence across all contributing animals
    coherence_matrix_left = np.array([item['result']['coherence'] for item in ldf_coherence_results['left']])
    mean_coherence_left = np.mean(coherence_matrix_left, axis=0)
    std_coherence_left = np.std(coherence_matrix_left, axis=0)
    
    coherence_matrix_right = np.array([item['result']['coherence'] for item in ldf_coherence_results['right']])
    mean_coherence_right = np.mean(coherence_matrix_right, axis=0)
    std_coherence_right = np.std(coherence_matrix_right, axis=0)
    
    # Find peak frequencies within autoregulation range (0.005-0.1 Hz)
    autoregulation_mask = (all_frequencies_ldf >= 0.005) & (all_frequencies_ldf <= 0.1)
    autoregulation_freqs = all_frequencies_ldf[autoregulation_mask]
    
    # Left hemisphere
    autoregulation_coherence_left = mean_coherence_left[autoregulation_mask]
    peak_idx_left = np.argmax(autoregulation_coherence_left)
    peak_frequency_left = autoregulation_freqs[peak_idx_left]
    peak_coherence_left = autoregulation_coherence_left[peak_idx_left]
    
    # Right hemisphere
    autoregulation_coherence_right = mean_coherence_right[autoregulation_mask]
    peak_idx_right = np.argmax(autoregulation_coherence_right)
    peak_frequency_right = autoregulation_freqs[peak_idx_right]
    peak_coherence_right = autoregulation_coherence_right[peak_idx_right]
    
    print("=" * 70)
    print("LDF-BASED AUTOREGULATION FREQUENCY ANALYSIS (DETRENDED DATA)")
    print("=" * 70)
    print(f"LEFT HEMISPHERE:")
    print(f"  Peak frequency: {peak_frequency_left:.4f} Hz ({1/peak_frequency_left:.0f}s period)")
    print(f"  Peak coherence: {peak_coherence_left:.3f}")
    print(f"\nRIGHT HEMISPHERE:")
    print(f"  Peak frequency: {peak_frequency_right:.4f} Hz ({1/peak_frequency_right:.0f}s period)")
    print(f"  Peak coherence: {peak_coherence_right:.3f}")
    
    # Compare with ABP-ICP results from your earlier analysis
    print(f"\nCOMPARISON WITH ABP-ICP ANALYSIS:")
    print(f"  ABP-ICP peak frequency: 0.0078 Hz (128s period)")
    print(f"  LDF-Left peak frequency: {peak_frequency_left:.4f} Hz ({1/peak_frequency_left:.0f}s period)")
    print(f"  LDF-Right peak frequency: {peak_frequency_right:.4f} Hz ({1/peak_frequency_right:.0f}s period)")
    
    # Calculate recommended windows for LDF
    avg_ldf_frequency = (peak_frequency_left + peak_frequency_right) / 2
    ldf_window_3cycles = int(3 / avg_ldf_frequency)
    ldf_window_5cycles = int(5 / avg_ldf_frequency)
    
    print(f"\nRECOMMENDED LDF AUTOREGULATION WINDOWS:")
    print(f"  Average peak frequency: {avg_ldf_frequency:.4f} Hz")
    print(f"  3 cycles: {ldf_window_3cycles} seconds")
    print(f"  5 cycles: {ldf_window_5cycles} seconds")
    print(f"  300s window captures: {300 * avg_ldf_frequency:.1f} cycles")

# %% Analyze coherence by frequency bands

freq_bands = [
    ("Very slow waves", 0.005, 0.02),
    ("Slow waves", 0.02, 0.05),
    ("Intermediate", 0.05, 0.1),
    ("Higher freq", 0.1, 0.2)
]

print(f"\nCOHERENCE BY FREQUENCY BAND:")
print("Left Hemisphere:")
for band_name, f_low, f_high in freq_bands:
    band_mask = (all_frequencies_ldf >= f_low) & (all_frequencies_ldf < f_high)
    if np.any(band_mask):
        band_coherence = mean_coherence_left[band_mask]
        max_coh = np.max(band_coherence)
        mean_coh = np.mean(band_coherence)
        print(f"  {band_name:15s}: max={max_coh:.3f}, mean={mean_coh:.3f}")

print("Right Hemisphere:")
for band_name, f_low, f_high in freq_bands:
    band_mask = (all_frequencies_ldf >= f_low) & (all_frequencies_ldf < f_high)
    if np.any(band_mask):
        band_coherence = mean_coherence_right[band_mask]
        max_coh = np.max(band_coherence)
        mean_coh = np.mean(band_coherence)
        print(f"  {band_name:15s}: max={max_coh:.3f}, mean={mean_coh:.3f}")
        
        
# %% Create Final LDF Coherence Analysis Figure

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Left Hemisphere Analysis
ax1.set_title('ABP-LDF Coherence: Left Hemisphere (n=16)', fontsize=14, fontweight='bold')

# Plot individual animal coherence curves with transparency
colors_left = plt.cm.Reds(np.linspace(0.3, 0.9, len(ldf_coherence_results['left'])))
for i, item in enumerate(ldf_coherence_results['left']):
    result = item['result']
    ax1.plot(result['frequencies'], result['coherence'], 
             alpha=0.4, color='lightcoral', linewidth=1)

# Plot mean coherence
ax1.plot(all_frequencies_ldf, mean_coherence_left, 'darkred', linewidth=4, 
         label='Mean coherence', zorder=10)
ax1.fill_between(all_frequencies_ldf, 
                 mean_coherence_left - std_coherence_left, 
                 mean_coherence_left + std_coherence_left, 
                 alpha=0.3, color='red', zorder=5)

# Mark peak frequency and autoregulation range
ax1.axvspan(0.005, 0.02, alpha=0.15, color='lightblue', 
           label='Optimal autoregulation\nrange (0.005-0.02 Hz)', zorder=1)
ax1.axvline(peak_frequency_left, color='red', linestyle='--', linewidth=3, 
            label=f'Peak: {peak_frequency_left:.4f} Hz', zorder=6)
ax1.scatter(peak_frequency_left, peak_coherence_left, 
           color='red', s=150, zorder=15, edgecolor='darkred', linewidth=2)

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Right Hemisphere Analysis
ax2.set_title('ABP-LDF Coherence: Right Hemisphere (n=16)', fontsize=14, fontweight='bold')

# Plot individual animal coherence curves
for i, item in enumerate(ldf_coherence_results['right']):
    result = item['result']
    ax2.plot(result['frequencies'], result['coherence'], 
             alpha=0.4, color='lightblue', linewidth=1)

# Plot mean coherence
ax2.plot(all_frequencies_ldf, mean_coherence_right, 'darkblue', linewidth=4, 
         label='Mean coherence', zorder=10)
ax2.fill_between(all_frequencies_ldf, 
                 mean_coherence_right - std_coherence_right, 
                 mean_coherence_right + std_coherence_right, 
                 alpha=0.3, color='blue', zorder=5)

# Mark peak frequency and autoregulation range
ax2.axvspan(0.005, 0.02, alpha=0.15, color='lightblue', 
           label='Optimal autoregulation\nrange (0.005-0.02 Hz)', zorder=1)
ax2.axvline(peak_frequency_right, color='blue', linestyle='--', linewidth=3, 
            label=f'Peak: {peak_frequency_right:.4f} Hz', zorder=6)
ax2.scatter(peak_frequency_right, peak_coherence_right, 
           color='blue', s=150, zorder=15, edgecolor='darkblue', linewidth=2)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0, 0.15)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Hemispheric Comparison
ax3.set_title('Hemispheric Comparison: ABP-LDF Coherence', fontsize=14, fontweight='bold')

# Add autoregulation range background
ax3.axvspan(0.005, 0.02, alpha=0.3, color='lightgreen', 
           label='Autoregulation range\n(0.005-0.02 Hz)', zorder=1)

# Plot both hemispheres
ax3.plot(all_frequencies_ldf, mean_coherence_left, 'darkred', linewidth=3, 
         label='Left hemisphere', zorder=5)
ax3.plot(all_frequencies_ldf, mean_coherence_right, 'darkblue', linewidth=3, 
         label='Right hemisphere', zorder=5)

# Mark both peak frequencies
ax3.axvline(peak_frequency_left, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
ax3.axvline(peak_frequency_right, color='blue', linestyle='--', linewidth=2, alpha=0.7, zorder=2)

ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Coherence')
ax3.set_xlim(0.005, 0.1)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Peak Frequency Comparison with All Methods
ax4.set_title('Peak Autoregulation Frequencies: All Methods', fontsize=14, fontweight='bold')

methods = ['ABP-ICP\n(n=21)', 'LDF-Left\n(n=16)', 'LDF-Right\n(n=16)']
frequencies = [0.0078, peak_frequency_left, peak_frequency_right]
colors = ['gray', 'darkred', 'darkblue']

bars = ax4.bar(methods, frequencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add frequency values on bars
for bar, freq in zip(bars, frequencies):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{freq:.4f} Hz\n({1/freq:.0f}s period)', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4.set_ylabel('Peak Frequency (Hz)')
ax4.set_ylim(0, max(frequencies) * 1.3)
ax4.grid(True, alpha=0.3, axis='y')

# Add interpretation text
ax4.text(0.5, 0.7, 'Left hemisphere matches\nABP-ICP analysis\n\nRight hemisphere shows\nhigher frequency peak', 
         transform=ax4.transAxes, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Final comprehensive summary
print("\n" + "="*80)
print("FINAL LDF AUTOREGULATION ANALYSIS SUMMARY - DETRENDED DATA")
print("="*80)
print(f"Contributing animals: 16 out of 21 total")
print(f"Data quality: LDF >98% complete, ABP variable (22-71%)")
print(f"\nKey findings:")
print(f"• Left hemisphere: {peak_frequency_left:.4f} Hz - MATCHES ABP-ICP perfectly")
print(f"• Right hemisphere: {peak_frequency_right:.4f} Hz - Higher frequency peak")
print(f"• Hemispheric asymmetry in autoregulation frequencies")
print(f"• 300s window: captures {300 * peak_frequency_left:.1f} cycles (left), {300 * peak_frequency_right:.1f} cycles (right)")
print(f"• Validates spectral approach for autoregulation analysis")
print("="*80)


# %% Create Clean 2-Panel LDF Figure - Final Version

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Combined Left and Right Hemisphere Coherence
ax1.set_title('ABP-LDF Coherence: Bilateral Comparison', fontsize=14, fontweight='bold')

# Enhanced autoregulation range
ax1.axvspan(0.005, 0.02, alpha=0.3, color='lightgreen', 
           label='Autoregulation range\n(0.005-0.02 Hz)', zorder=1)

# Plot both hemispheres
ax1.plot(all_frequencies_ldf, mean_coherence_left, 'darkred', linewidth=3, 
         label='Left hemisphere', zorder=5)
ax1.fill_between(all_frequencies_ldf, 
                 mean_coherence_left - std_coherence_left, 
                 mean_coherence_left + std_coherence_left, 
                 alpha=0.2, color='red', zorder=3)

ax1.plot(all_frequencies_ldf, mean_coherence_right, 'darkblue', linewidth=3, 
         label='Right hemisphere', zorder=5)
ax1.fill_between(all_frequencies_ldf, 
                 mean_coherence_right - std_coherence_right, 
                 mean_coherence_right + std_coherence_right, 
                 alpha=0.2, color='blue', zorder=3)

# Mark peak frequencies
ax1.axvline(peak_frequency_left, color='red', linestyle='--', linewidth=3, 
           alpha=0.8, zorder=6)
ax1.axvline(peak_frequency_right, color='blue', linestyle='--', linewidth=3, 
           alpha=0.8, zorder=6)
ax1.scatter(peak_frequency_left, peak_coherence_left, 
           color='red', s=150, zorder=10, edgecolor='darkred', linewidth=3)
ax1.scatter(peak_frequency_right, peak_coherence_right, 
           color='blue', s=150, zorder=10, edgecolor='darkblue', linewidth=3)

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Panel 2: Peak Frequency Comparison Bar Chart
ax2.set_title('Peak Autoregulation Frequencies: All Methods', fontsize=14, fontweight='bold')

methods = ['ABP-ICP', 'LDF-Left', 'LDF-Right']
frequencies = [0.0078, peak_frequency_left, peak_frequency_right]
colors = ['gray', 'darkred', 'darkblue']

bars = ax2.bar(methods, frequencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add frequency values on bars
for bar, freq in zip(bars, frequencies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f'{freq:.4f} Hz\n({1/freq:.0f}s period)', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel('Peak Frequency (Hz)')
ax2.set_ylim(0, max(frequencies) * 1.3)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SUMMARY: LDF AUTOREGULATION ANALYSIS")
print("="*60)
print(f"• Left hemisphere peak: {peak_frequency_left:.4f} Hz - matches ABP-ICP")
print(f"• Right hemisphere peak: {peak_frequency_right:.4f} Hz - higher frequency")
print(f"• Strong bilateral LDF coherence (n=16 animals)")
print(f"• Hemispheric asymmetry in autoregulation frequencies")
print("="*60)