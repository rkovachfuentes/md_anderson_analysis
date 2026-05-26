import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import config
from area_testing import backtrack_to_local_max

showPlot = config.showPlot

def find_noise_threshold(signal, idx=500):
    # Use the first 500 points to establish the 'normal' wiggle
    first_noise = signal[:idx]
    sigma = np.std(first_noise)
    # If the signal is perfectly flat, provide a tiny floor to prevent 0-thresholds
    return 10 * sigma

def return_before_first_spike(signal, buffer=200):
    cleaned_signal = signal.copy()
    noise_limit = find_noise_threshold(signal)
    
    # Find the first time the machine error breaks the noise floor
    spike_indices = np.where(np.abs(signal) > noise_limit)[0]
    
    first_spike_idx = 0
    if len(spike_indices) > 0:
        first_spike_idx = spike_indices[0]
        # Zero out the spike PLUS a buffer so we don't catch the 'tail' of the error
        cleaned_signal[0 : first_spike_idx + buffer] = 0
        
    return cleaned_signal, first_spike_idx + buffer, noise_limit

def get_drop_event(time, signal):
    # 1. Clean the signal once
    cleaned, end_of_error, threshold = return_before_first_spike(signal)
    full_clean, _, _ = return_before_first_spike(np.flip(cleaned))
    final_clean = np.flip(full_clean)
    
    # 2. Look for the drop AFTER the error window
    drop_threshold = -threshold
    # Search the zone after the machine error
    search_zone = final_clean[end_of_error:]
    drop_indices = np.where(search_zone < drop_threshold)[0]
    
    if len(drop_indices) > 0:
        start_idx = end_of_error + drop_indices[0]
        shifted_start = backtrack_to_local_max(time, signal, start_idx)
        print("shifted start found: ")
        # Recovery: Find where it crosses back to 0
        recovery = np.where(final_clean[start_idx:] >= 0)[0]
        end_idx = shifted_start + recovery[0] if len(recovery) > 0 else len(signal)
        return True, start_idx, end_idx, drop_threshold
    
    return False, None, None, drop_threshold

# --- Execution ---
def plot_cleaned(file_path):
    df = pd.read_csv(file_path)
    raw_signal = df["CH1"].values
    time = df["TIME"].values
    
    # Process
    found, start, end, d_thresh = get_drop_event(time, raw_signal)
    cleaned_display, idx, _ = return_before_first_spike(raw_signal)
    full_clean, _, _ = return_before_first_spike(np.flip(cleaned_display))

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(raw_signal, color='gray', alpha=0.3, label='Raw')
    plt.plot(np.flip(full_clean), color='blue', alpha=0.7, label='Cleaned')
    
    # Draw the threshold line so we can see what the code is "thinking"
    plt.axhline(d_thresh, color='red', linestyle='--', label='Drop Threshold')
    print("plotter reporting for duty")
    
    if found:
        # Highlight the drop in Green
        plt.plot(np.arange(start, end), cleaned_display[start:end], color='limegreen', linewidth=2, label='Detected Drop')
        plt.title("DETECTION SUCCESS: {}".format(os.path.basename(file_path)))
    else:
        plt.title("No Drop Found: {}".format(os.path.basename(file_path)))
        
    plt.legend()
    if showPlot: plt.show()
    plt.savefig("cleaned_figs/cleaned_{}.png".format(os.path.basename(file_path)))
    plt.close()
    return time, np.flip(full_clean), found