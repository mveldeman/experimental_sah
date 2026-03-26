# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:07:14 2025

@author: mveldeman
"""

"""
This is the first analysis of experimental SAH data focussing on PRx calculation
befroe and after SAH induction. The Sham animal group will be used as the normal
baseline to calculate appropriate time windows for PRx calculations. 
"""

"""
This first section will focus on determining the optimal window for PRx 
calculation in my experimental setting using the data from 21 sham operated 
rats. 
"""
# %% Setup and File Loading

import numpy as np
import pandas as pd
import os

# Set your data path
data_dir = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/B_Sham/AB_Sham_processing_cleaned_converted/B_Sham_resliced_as_csv"

# Get CSV files with the correct pattern
csv_files = []
for filename in os.listdir(data_dir):
    if filename.startswith('S') and filename.endswith('_per000.csv'):
        csv_files.append(filename)

csv_files.sort()
print(f"Found {len(csv_files)} sham files")

# %% Data Loading Function

def load_animal_data_german(filepath):
    """Load CSV data for one animal with German locale settings"""
    df = pd.read_csv(filepath, 
                     sep=';',           
                     decimal=',',       
                     encoding='utf-8')
    
    # Extract the main signals we need for PRx calculation
    abp = df['abp'].values  
    icp = df['icp'].values
    datetime = df['DateTime'].values
    
    # Check data quality
    abp_completeness = np.sum(~np.isnan(abp)) / len(abp) * 100
    icp_completeness = np.sum(~np.isnan(icp)) / len(icp) * 100
    
    return {
        'abp': abp,
        'icp': icp, 
        'datetime': datetime,
        'abp_completeness': abp_completeness,
        'icp_completeness': icp_completeness,
        'n_points': len(abp)
    }

# %% Data Quality Assessment

print("Complete data quality assessment:")
print("Animal\t\tABP%\tICP%\tPoints\tDuration(min)")
print("-" * 55)

all_animal_data = {}

for csv_file in csv_files:
    filepath = os.path.join(data_dir, csv_file)
    data = load_animal_data_german(filepath)
    
    animal_id = csv_file.replace('_per000.csv', '')
    duration_min = data['n_points'] / 60
    
    all_animal_data[animal_id] = data
    
    print(f"{animal_id}\t\t{data['abp_completeness']:.1f}%\t{data['icp_completeness']:.1f}%\t{data['n_points']}\t{duration_min:.1f}")

# Summary statistics
abp_completeness = [data['abp_completeness'] for data in all_animal_data.values()]
print(f"\nABP Data Quality Summary:")
print(f"Mean: {np.mean(abp_completeness):.1f}%")
print(f"Median: {np.median(abp_completeness):.1f}%")
print(f"Min: {np.min(abp_completeness):.1f}%")
print(f"Max: {np.max(abp_completeness):.1f}%")

# Identify animals with good ABP data (>70%) for initial spectral analysis
good_abp_animals = [animal_id for animal_id, data in all_animal_data.items() 
                    if data['abp_completeness'] > 70]
print(f"\nAnimals with >70% ABP data: {len(good_abp_animals)}")
print(good_abp_animals)

# %%

# %% PRx Calculation Function
def calculate_prx_valid_data(abp, icp, window_seconds=120, fs=1.0):
    """
    Calculate PRx using only time points where both ABP and ICP are valid
    
    Parameters:
    - abp: arterial blood pressure array
    - icp: intracranial pressure array  
    - window_seconds: correlation window length in seconds
    - fs: sampling frequency (1 Hz for your data)
    
    Returns:
    - prx_values: array of PRx values
    - valid_indices: indices where PRx could be calculated
    - n_valid_points: number of valid data points used
    """
    
    # Find indices where both signals are valid (not NaN)
    valid_mask = ~(np.isnan(abp) | np.isnan(icp))
    valid_indices = np.where(valid_mask)[0]
    
    if np.sum(valid_mask) < window_seconds:
        print(f"Warning: Only {np.sum(valid_mask)} valid points, need at least {window_seconds}")
        return None, None, 0
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    icp_valid = icp[valid_mask]
    
    # Calculate PRx using rolling correlation
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
            prx_indices.append(valid_indices[i])  # Map back to original time series
    
    return np.array(prx_values), np.array(prx_indices), np.sum(valid_mask)

# %% Test PRx Calculation on One Animal

# Test with first animal that has good ABP data
test_animal = good_abp_animals[0] if good_abp_animals else list(all_animal_data.keys())[0]
test_data = all_animal_data[test_animal]

print(f"Testing PRx calculation on animal {test_animal}")
print(f"ABP completeness: {test_data['abp_completeness']:.1f}%")

# Calculate PRx with different window sizes to see effect
window_sizes = [60, 120, 180, 240]  # Test different windows

for window in window_sizes:
    prx_vals, prx_idx, n_valid = calculate_prx_valid_data(
        test_data['abp'], 
        test_data['icp'], 
        window_seconds=window
    )
    
    if prx_vals is not None:
        print(f"Window {window}s: {len(prx_vals)} PRx values, "
              f"mean PRx = {np.mean(prx_vals):.3f} ± {np.std(prx_vals):.3f}")
    else:
        print(f"Window {window}s: Failed - insufficient data")
        
        
# %% Restart from here afte Kernel reboot     

import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt

# Set your data path
data_dir = "/Volumes/LaCie/A_A_A_Animal Data Labchart/A_Animals to use/B_Sham/AB_Sham_processing_cleaned_converted/B_Sham_resliced_as_csv"

# Get CSV files with the correct pattern
csv_files = []
for filename in os.listdir(data_dir):
    if filename.startswith('S') and filename.endswith('_per000.csv'):
        csv_files.append(filename)

csv_files.sort()
print(f"Found {len(csv_files)} sham files")

# Data loading function
def load_animal_data_german(filepath):
    """Load CSV data for one animal with German locale settings"""
    df = pd.read_csv(filepath, 
                     sep=';',           
                     decimal=',',       
                     encoding='utf-8')
    
    # Extract the main signals we need for PRx calculation
    abp = df['abp'].values  
    icp = df['icp'].values
    datetime = df['DateTime'].values
    
    # Check data quality
    abp_completeness = np.sum(~np.isnan(abp)) / len(abp) * 100
    icp_completeness = np.sum(~np.isnan(icp)) / len(icp) * 100
    
    return {
        'abp': abp,
        'icp': icp, 
        'datetime': datetime,
        'abp_completeness': abp_completeness,
        'icp_completeness': icp_completeness,
        'n_points': len(abp)
    }

# PRx calculation function
def calculate_prx_valid_data(abp, icp, window_seconds=120, fs=1.0):
    """Calculate PRx using only time points where both ABP and ICP are valid"""
    
    # Find indices where both signals are valid (not NaN)
    valid_mask = ~(np.isnan(abp) | np.isnan(icp))
    valid_indices = np.where(valid_mask)[0]
    
    if np.sum(valid_mask) < window_seconds:
        print(f"Warning: Only {np.sum(valid_mask)} valid points, need at least {window_seconds}")
        return None, None, 0
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    icp_valid = icp[valid_mask]
    
    # Calculate PRx using rolling correlation
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
            prx_indices.append(valid_indices[i])  # Map back to original time series
    
    return np.array(prx_values), np.array(prx_indices), np.sum(valid_mask)

# Load all animal data
print("\nLoading all animal data...")
all_animal_data = {}

for csv_file in csv_files:
    filepath = os.path.join(data_dir, csv_file)
    data = load_animal_data_german(filepath)
    animal_id = csv_file.replace('_per000.csv', '')
    all_animal_data[animal_id] = data

# Identify animals with good ABP data (>70%) for spectral analysis
good_abp_animals = [animal_id for animal_id, data in all_animal_data.items() 
                    if data['abp_completeness'] > 70]

print(f"Loaded {len(all_animal_data)} animals")
print(f"Animals with >70% ABP data: {len(good_abp_animals)}")
print("Setup complete! Ready for spectral analysis.")


# %% Spectral Analysis Functions

def calculate_coherence_spectrum(abp, icp, fs=1.0, nperseg=256):
    """
    Calculate coherence spectrum between ABP and ICP using only valid data points
    """
    # Find valid data points (both signals present)
    valid_mask = ~(np.isnan(abp) | np.isnan(icp))
    
    if np.sum(valid_mask) < nperseg:
        return None
    
    # Extract valid data
    abp_valid = abp[valid_mask]
    icp_valid = icp[valid_mask]
    
    # Calculate coherence
    frequencies, coherence = signal.coherence(
        abp_valid, icp_valid, 
        fs=fs, 
        nperseg=nperseg,
        noverlap=nperseg//2  # Changed from 'overlap' to 'noverlap'
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

# %% Multi-Animal Spectral Analysis

print("Performing spectral analysis on animals with good ABP data...")

all_coherence_results = []
all_frequencies = None

for animal_id in good_abp_animals:
    data = all_animal_data[animal_id]
    
    result = calculate_coherence_spectrum(data['abp'], data['icp'])
    
    if result is not None:
        all_coherence_results.append(result)
        if all_frequencies is None:
            all_frequencies = result['frequencies']
        
        print(f"{animal_id}: {result['n_valid_points']} valid points used")

print(f"\nSuccessfully analyzed {len(all_coherence_results)} animals")

# Calculate mean coherence across all animals
if all_coherence_results:
    coherence_matrix = np.array([result['coherence'] for result in all_coherence_results])
    mean_coherence = np.mean(coherence_matrix, axis=0)
    std_coherence = np.std(coherence_matrix, axis=0)
    
    print(f"Coherence analysis complete for {len(all_coherence_results)} animals")
    print(f"Frequency range: {all_frequencies[0]:.4f} to {all_frequencies[-1]:.4f} Hz")
    

    
# %% Analyze Coherence Results
    
# Focus on the autoregulation frequency range (0.005 to 0.1 Hz)
autoregulation_mask = (all_frequencies >= 0.005) & (all_frequencies <= 0.1)
autoregulation_freqs = all_frequencies[autoregulation_mask]
autoregulation_coherence = mean_coherence[autoregulation_mask]

# Find peak coherence in autoregulation range
peak_idx = np.argmax(autoregulation_coherence)
peak_frequency = autoregulation_freqs[peak_idx]
peak_coherence = autoregulation_coherence[peak_idx]

print("Autoregulation Frequency Analysis:")
print(f"Peak coherence frequency: {peak_frequency:.4f} Hz")
print(f"Peak coherence value: {peak_coherence:.3f}")
print(f"Period of peak frequency: {1/peak_frequency:.1f} seconds")

# Calculate recommended PRx window (3-5 cycles of dominant frequency)
recommended_window_3cycles = int(3 / peak_frequency)
recommended_window_5cycles = int(5 / peak_frequency)

print(f"\nRecommended PRx windows:")
print(f"3 cycles: {recommended_window_3cycles} seconds")
print(f"5 cycles: {recommended_window_5cycles} seconds")

# Show coherence statistics in different frequency bands
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
    
# %% Create Spectral Analysis Figure
    
plt.figure(figsize=(12, 8))

# Create subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Individual animal coherence curves
ax1.set_title('ABP-ICP Coherence by Animal', fontsize=14, fontweight='bold')
colors = plt.cm.Set1(np.linspace(0, 1, len(all_coherence_results)))

for i, result in enumerate(all_coherence_results):
    ax1.plot(result['frequencies'], result['coherence'], 
             alpha=0.3, color=colors[i], linewidth=1)

# Plot mean coherence
ax1.plot(all_frequencies, mean_coherence, 'k-', linewidth=3, label='Mean coherence')
ax1.fill_between(all_frequencies, 
                 mean_coherence - std_coherence, 
                 mean_coherence + std_coherence, 
                 alpha=0.2, color='gray', label='±1 SD')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)  # Focus on autoregulation range
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Zoomed view of autoregulation range
ax2.set_title('Autoregulation Range (0.005-0.1 Hz)', fontsize=14, fontweight='bold')
ax2.plot(all_frequencies, mean_coherence, 'k-', linewidth=2)
ax2.fill_between(all_frequencies, 
                 mean_coherence - std_coherence, 
                 mean_coherence + std_coherence, 
                 alpha=0.2, color='gray')

# Mark peak frequency
ax2.axvline(peak_frequency, color='red', linestyle='--', linewidth=2, 
            label=f'Peak: {peak_frequency:.4f} Hz\n({1/peak_frequency:.0f}s period)')
ax2.scatter(peak_frequency, peak_coherence, color='red', s=100, zorder=5)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0.005, 0.1)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Frequency band analysis
bands = ['Very slow\n(0.005-0.02)', 'Slow\n(0.02-0.05)', 'Intermediate\n(0.05-0.1)', 'Higher\n(0.1-0.2)']
band_coherence = []

for _, f_low, f_high in freq_bands:
    band_mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(band_mask):
        band_coherence.append(np.mean(mean_coherence[band_mask]))
    else:
        band_coherence.append(0)

ax3.set_title('Mean Coherence by Frequency Band', fontsize=14, fontweight='bold')
bars = ax3.bar(bands, band_coherence, color=['darkred', 'red', 'orange', 'yellow'])
ax3.set_ylabel('Mean Coherence')
ax3.set_ylim(0, max(band_coherence) * 1.2)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# Add values on bars
for bar, value in zip(bars, band_coherence):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: PRx window recommendation
window_options = [60, 120, 180, 240, 300, 384, 480, 640]
cycles = [w * peak_frequency for w in window_options]

ax4.set_title('PRx Window Options vs. Autoregulation Cycles', fontsize=14, fontweight='bold')
ax4.plot(window_options, cycles, 'bo-', linewidth=2, markersize=8)
ax4.axhline(3, color='green', linestyle='--', label='3 cycles (minimum)')
ax4.axhline(5, color='red', linestyle='--', label='5 cycles (optimal)')
ax4.axvline(recommended_window_3cycles, color='green', linestyle=':', alpha=0.7)
ax4.axvline(recommended_window_5cycles, color='red', linestyle=':', alpha=0.7)

ax4.set_xlabel('PRx Window (seconds)')
ax4.set_ylabel('Number of Autoregulation Cycles')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# Print final recommendation
print(f"\n" + "="*60)
print("FINAL RECOMMENDATION BASED ON SPECTRAL ANALYSIS")
print("="*60)
print(f"Dominant autoregulation frequency: {peak_frequency:.4f} Hz ({1/peak_frequency:.0f}s period)")
print(f"For practical PRx calculation, recommended windows:")
print(f"  • Conservative (3 cycles): ~300-400 seconds")
print(f"  • Optimal (4-5 cycles):   ~400-500 seconds") 
print(f"  • Based on data (3 cycles): {recommended_window_3cycles} seconds")
print("="*60)



# %% Include All Animals - Analyze Valid Segments Only

print("Performing spectral analysis on ALL animals using valid data segments...")

all_coherence_results_expanded = []
all_frequencies = None

# Process all 21 animals
for animal_id, data in all_animal_data.items():
    print(f"Processing {animal_id}: ABP={data['abp_completeness']:.1f}%, ICP={data['icp_completeness']:.1f}%")
    
    result = calculate_coherence_spectrum(data['abp'], data['icp'])
    
    if result is not None:
        all_coherence_results_expanded.append({
            'animal_id': animal_id,
            'result': result,
            'abp_completeness': data['abp_completeness']
        })
        
        if all_frequencies is None:
            all_frequencies = result['frequencies']
        
        print(f"  → Success: {result['n_valid_points']} valid points used")
    else:
        print(f"  → Failed: insufficient valid data")

print(f"\nSuccessfully analyzed {len(all_coherence_results_expanded)} out of {len(all_animal_data)} animals")

# Calculate statistics across all animals
if all_coherence_results_expanded:
    # Extract coherence data
    coherence_matrix_expanded = np.array([item['result']['coherence'] for item in all_coherence_results_expanded])
    mean_coherence_expanded = np.mean(coherence_matrix_expanded, axis=0)
    std_coherence_expanded = np.std(coherence_matrix_expanded, axis=0)
    
    # Show which animals contributed
    contributing_animals = [item['animal_id'] for item in all_coherence_results_expanded]
    abp_quality = [item['abp_completeness'] for item in all_coherence_results_expanded]
    
    print(f"\nContributing animals: {contributing_animals}")
    print(f"ABP completeness range: {min(abp_quality):.1f}% to {max(abp_quality):.1f}%")
    print(f"Mean ABP completeness: {np.mean(abp_quality):.1f}%")
    
# %% Compare Spectral Results
    
# Focus on autoregulation range for both datasets
autoregulation_mask = (all_frequencies >= 0.005) & (all_frequencies <= 0.1)
autoregulation_freqs = all_frequencies[autoregulation_mask]

# Original analysis (4 good animals)
autoregulation_coherence_original = mean_coherence[autoregulation_mask]
peak_idx_original = np.argmax(autoregulation_coherence_original)
peak_frequency_original = autoregulation_freqs[peak_idx_original]

# Expanded analysis (all animals)
autoregulation_coherence_expanded = mean_coherence_expanded[autoregulation_mask]
peak_idx_expanded = np.argmax(autoregulation_coherence_expanded)
peak_frequency_expanded = autoregulation_freqs[peak_idx_expanded]

print("COMPARISON: 4 Good Animals vs All 21 Animals")
print("="*55)
print(f"Peak frequency (4 animals):   {peak_frequency_original:.4f} Hz ({1/peak_frequency_original:.0f}s period)")
print(f"Peak frequency (all animals):  {peak_frequency_expanded:.4f} Hz ({1/peak_frequency_expanded:.0f}s period)")
print(f"Peak coherence (4 animals):   {autoregulation_coherence_original[peak_idx_original]:.3f}")
print(f"Peak coherence (all animals):  {autoregulation_coherence_expanded[peak_idx_expanded]:.3f}")

# Recommended windows based on expanded analysis
recommended_window_3cycles_expanded = int(3 / peak_frequency_expanded)
recommended_window_5cycles_expanded = int(5 / peak_frequency_expanded)

print(f"\nRECOMMENDED PRx WINDOWS (based on all 21 animals):")
print(f"3 cycles: {recommended_window_3cycles_expanded} seconds")
print(f"5 cycles: {recommended_window_5cycles_expanded} seconds")

# Show coherence in frequency bands for expanded analysis
freq_bands = [
    ("Very slow waves", 0.005, 0.02),
    ("Slow waves", 0.02, 0.05),
    ("Intermediate", 0.05, 0.1),
    ("Higher freq", 0.1, 0.2)
]

print(f"\nCOHERENCE BY FREQUENCY BAND (all 21 animals):")
for band_name, f_low, f_high in freq_bands:
    band_mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(band_mask):
        band_coherence = mean_coherence_expanded[band_mask]
        max_coh = np.max(band_coherence)
        mean_coh = np.mean(band_coherence)
        print(f"{band_name:15s}: max={max_coh:.3f}, mean={mean_coh:.3f}")
        
        
# %% Create Final Figure - Clean Version
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Mean coherence with confidence bands
ax1.set_title('ABP-ICP Coherence Spectrum (All 21 Animals)', fontsize=14, fontweight='bold')
ax1.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=3, label='Mean coherence')
ax1.fill_between(all_frequencies, 
                 mean_coherence_expanded - std_coherence_expanded, 
                 mean_coherence_expanded + std_coherence_expanded, 
                 alpha=0.2, color='gray', label='±1 SD')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Autoregulation range with peak
ax2.set_title('Peak Autoregulation Frequency', fontsize=14, fontweight='bold')
ax2.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=2)
ax2.axvline(peak_frequency_expanded, color='red', linestyle='--', linewidth=2, 
            label=f'Peak: {peak_frequency_expanded:.4f} Hz')
ax2.scatter(peak_frequency_expanded, autoregulation_coherence_expanded[peak_idx_expanded], 
           color='red', s=100, zorder=5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0.005, 0.1)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Frequency bands
band_names = ['Very slow', 'Slow', 'Intermediate', 'Higher']
band_ranges = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)]
band_values = []

for f_low, f_high in band_ranges:
    mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(mask):
        band_values.append(np.mean(mean_coherence_expanded[mask]))
    else:
        band_values.append(0)

ax3.set_title('Coherence by Frequency Band', fontsize=14, fontweight='bold')
bars = ax3.bar(band_names, band_values, color=['darkred', 'red', 'orange', 'yellow'])
ax3.set_ylabel('Mean Coherence')
for bar, value in zip(bars, band_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Window recommendations
windows = [120, 180, 240, 300, 360, 480, 600]
cycles = [w * peak_frequency_expanded for w in windows]

ax4.set_title('PRx Window vs Autoregulation Cycles', fontsize=14, fontweight='bold')
ax4.plot(windows, cycles, 'bo-', linewidth=2, markersize=8)
ax4.axhline(3, color='green', linestyle='--', linewidth=2, label='3 cycles')
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='5 cycles')
ax4.set_xlabel('PRx Window (seconds)')
ax4.set_ylabel('Number of Cycles')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*60)
print("SPECTRAL ANALYSIS RESULTS - ALL 21 SHAM ANIMALS")
print("="*60)
print(f"Peak frequency: {peak_frequency_expanded:.4f} Hz ({1/peak_frequency_expanded:.0f}s period)")
print(f"Peak coherence: {autoregulation_coherence_expanded[peak_idx_expanded]:.3f}")
print(f"Recommended windows:")
print(f"  - 300s (practical 3-cycle)")
print(f"  - 480s (practical 5-cycle)")
print("="*60)



# %% Create Final Figure - With Matching Color Scheme

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Mean coherence with confidence bands
ax1.set_title('ABP-ICP Coherence Spectrum (All 21 Animals)', fontsize=14, fontweight='bold')
ax1.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=3, label='Mean coherence')
ax1.fill_between(all_frequencies, 
                 mean_coherence_expanded - std_coherence_expanded, 
                 mean_coherence_expanded + std_coherence_expanded, 
                 alpha=0.2, color='gray', label='±1 SD')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Define consistent color scheme
band_colors_light = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']  # For overlays
band_colors_dark = ['darkblue', 'darkgreen', 'orange', 'coral']  # For bars

# Plot 2: Autoregulation range with peak AND frequency band overlays
ax2.set_title('Autoregulation Frequency Bands', fontsize=14, fontweight='bold')

# Add frequency band background colors FIRST (so they're behind the line)
band_labels = ['Very slow waves\n(0.005-0.02 Hz)', 'Slow waves\n(0.02-0.05 Hz)', 
               'Intermediate\n(0.05-0.1 Hz)', 'Higher freq\n(0.1-0.2 Hz)']
band_ranges = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)]

for i, (f_low, f_high) in enumerate(band_ranges):
    if f_high <= 0.15:  # Only show bands in our plot range
        ax2.axvspan(f_low, f_high, alpha=0.3, color=band_colors_light[i], 
                   label=band_labels[i])

# Plot the coherence line on top
ax2.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=3, label='Mean coherence')
ax2.fill_between(all_frequencies, 
                 mean_coherence_expanded - std_coherence_expanded, 
                 mean_coherence_expanded + std_coherence_expanded, 
                 alpha=0.2, color='gray')

# Mark the peak frequency
ax2.axvline(peak_frequency_expanded, color='red', linestyle='--', linewidth=3, 
            label=f'Peak: {peak_frequency_expanded:.4f} Hz\n({1/peak_frequency_expanded:.0f}s period)')
ax2.scatter(peak_frequency_expanded, autoregulation_coherence_expanded[peak_idx_expanded], 
           color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0.005, 0.15)
ax2.set_ylim(0, max(mean_coherence_expanded[autoregulation_mask]) * 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Frequency bands with MATCHING colors
band_names = ['Very slow', 'Slow', 'Intermediate', 'Higher']
band_values = []

for f_low, f_high in band_ranges:
    mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(mask):
        band_values.append(np.mean(mean_coherence_expanded[mask]))
    else:
        band_values.append(0)

ax3.set_title('Mean Coherence by Frequency Band', fontsize=14, fontweight='bold')
bars = ax3.bar(band_names, band_values, color=band_colors_dark)  # Using the darker matching colors
ax3.set_ylabel('Mean Coherence')
ax3.set_ylim(0, max(band_values) * 1.2)

for bar, value in zip(bars, band_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Window recommendations
windows = [120, 180, 240, 300, 360, 480, 600]
cycles = [w * peak_frequency_expanded for w in windows]

ax4.set_title('PRx Window Selection Guide', fontsize=14, fontweight='bold')
ax4.plot(windows, cycles, 'bo-', linewidth=2, markersize=8, label='Autoregulation cycles')
ax4.axhline(3, color='green', linestyle='--', linewidth=2, label='3 cycles (minimum)')
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='5 cycles (optimal)')

# Highlight practical recommendations
ax4.axvline(300, color='green', linestyle=':', alpha=0.7, linewidth=2)
ax4.axvline(480, color='red', linestyle=':', alpha=0.7, linewidth=2)

ax4.set_xlabel('PRx Window (seconds)')
ax4.set_ylabel('Number of Autoregulation Cycles')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_ylim(1, 6)

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*60)
print("SPECTRAL ANALYSIS RESULTS - ALL 21 SHAM ANIMALS")
print("="*60)
print(f"Peak frequency: {peak_frequency_expanded:.4f} Hz ({1/peak_frequency_expanded:.0f}s period)")
print(f"Peak coherence: {autoregulation_coherence_expanded[peak_idx_expanded]:.3f}")
print(f"Frequency band: Very slow waves (0.005-0.02 Hz)")
print(f"Recommended PRx windows:")
print(f"  - 300s (practical 3-cycle window)")
print(f"  - 480s (practical 5-cycle window)")
print("="*60)

# %% Create Final Figure - With Red-to-Yellow Color Scheme

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Mean coherence with confidence bands
ax1.set_title('ABP-ICP Coherence Spectrum (All 21 Animals)', fontsize=14, fontweight='bold')
ax1.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=3, label='Mean coherence')
ax1.fill_between(all_frequencies, 
                 mean_coherence_expanded - std_coherence_expanded, 
                 mean_coherence_expanded + std_coherence_expanded, 
                 alpha=0.2, color='gray', label='±1 SD')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coherence')
ax1.set_xlim(0, 0.15)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Define the red-to-yellow color scheme
bar_colors = ['darkred', 'red', 'orange', 'yellow']  # From your original code
overlay_colors = ['lightcoral', 'lightpink', 'lightsalmon', 'lightyellow']  # Lighter versions for overlays

# Plot 2: Autoregulation range with frequency band overlays
ax2.set_title('Autoregulation Frequency Bands', fontsize=14, fontweight='bold')

# Add frequency band background colors FIRST
band_labels = ['Very slow waves\n(0.005-0.02 Hz)', 'Slow waves\n(0.02-0.05 Hz)', 
               'Intermediate\n(0.05-0.1 Hz)', 'Higher freq\n(0.1-0.2 Hz)']
band_ranges = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)]

for i, (f_low, f_high) in enumerate(band_ranges):
    if f_high <= 0.15:  # Only show bands in our plot range
        ax2.axvspan(f_low, f_high, alpha=0.3, color=overlay_colors[i], 
                   label=band_labels[i])

# Plot the coherence line on top
ax2.plot(all_frequencies, mean_coherence_expanded, 'k-', linewidth=3, label='Mean coherence')
ax2.fill_between(all_frequencies, 
                 mean_coherence_expanded - std_coherence_expanded, 
                 mean_coherence_expanded + std_coherence_expanded, 
                 alpha=0.2, color='gray')

# Mark the peak frequency
ax2.axvline(peak_frequency_expanded, color='red', linestyle='--', linewidth=3, 
            label=f'Peak: {peak_frequency_expanded:.4f} Hz\n({1/peak_frequency_expanded:.0f}s period)')
ax2.scatter(peak_frequency_expanded, autoregulation_coherence_expanded[peak_idx_expanded], 
           color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Coherence')
ax2.set_xlim(0.005, 0.15)
ax2.set_ylim(0, max(mean_coherence_expanded[autoregulation_mask]) * 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Frequency bands with matching red-to-yellow colors
band_names = ['Very slow', 'Slow', 'Intermediate', 'Higher']
band_values = []

for f_low, f_high in band_ranges:
    mask = (all_frequencies >= f_low) & (all_frequencies < f_high)
    if np.any(mask):
        band_values.append(np.mean(mean_coherence_expanded[mask]))
    else:
        band_values.append(0)

ax3.set_title('Mean Coherence by Frequency Band', fontsize=14, fontweight='bold')
bars = ax3.bar(band_names, band_values, color=bar_colors)  # darkred, red, orange, yellow
ax3.set_ylabel('Mean Coherence')
ax3.set_ylim(0, max(band_values) * 1.2)

for bar, value in zip(bars, band_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Window recommendations
windows = [120, 180, 240, 300, 360, 480, 600]
cycles = [w * peak_frequency_expanded for w in windows]

ax4.set_title('PRx Window Selection Guide', fontsize=14, fontweight='bold')
ax4.plot(windows, cycles, 'bo-', linewidth=2, markersize=8, label='Autoregulation cycles')
ax4.axhline(3, color='green', linestyle='--', linewidth=2, label='3 cycles')
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='5 cycles')

# Highlight practical recommendations
ax4.axvline(300, color='green', linestyle=':', alpha=0.7, linewidth=2)
ax4.axvline(480, color='red', linestyle=':', alpha=0.7, linewidth=2)

ax4.set_xlabel('PRx Window (seconds)')
ax4.set_ylabel('Number of Autoregulation Cycles')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_ylim(1, 6)

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*60)
print("SPECTRAL ANALYSIS RESULTS - ALL 21 SHAM ANIMALS")
print("="*60)
print(f"Peak frequency: {peak_frequency_expanded:.4f} Hz ({1/peak_frequency_expanded:.0f}s period)")
print(f"Peak coherence: {autoregulation_coherence_expanded[peak_idx_expanded]:.3f}")
print(f"Frequency band: Very slow waves (0.005-0.02 Hz)")
print(f"Recommended PRx windows:")
print(f"  - 300s (practical 3-cycle window)")
print(f"  - 480s (practical 5-cycle window)")
print("="*60)










