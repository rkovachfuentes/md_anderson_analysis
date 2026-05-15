import numpy as np
import area_testing
import oct_file_generator
import dose_area_plots
from pathlib import Path
import time
from tqdm import trange
import pandas as pd
import platform
import subprocess
import os
from PIL import Image

# some gemini-generated functions for noise reduction
def remove_voltage_spikes(voltages, window_size=5, n_sigmas=3):
    """
    Removes outliers using a sliding Hampel Filter.
    
    Parameters:
    voltages (np.array): Input voltage readings.
    window_size (int): Number of neighbors to check on each side.
    n_sigmas (int): Sensitivity; higher means fewer points are flagged.
    """
    data = np.array(voltages, copy=True)
    n = len(data)
    
    # Constant for Gaussian consistency
    L = 1.4826
    
    for i in range(window_size, n - window_size):
        # Define the local window
        window = data[i - window_size : i + window_size + 1]
        
        # Calculate local median and local MAD
        local_median = np.median(window)
        local_mad = np.median(np.abs(window - local_median))
        local_sigma = L * local_mad
        
        # Check if the current point is a "spike"
        if np.abs(data[i] - local_median) > (n_sigmas * local_sigma):
            # Replace spike with the local median
            data[i] = local_median
            
    return data

def identify_noise_threshold(voltages, sigma_scale=3):
    """
    Identifies the noise threshold of a voltage signal using 
    Median Absolute Deviation (MAD).
    
    Parameters:
    voltages (list or np.array): The input voltage readings.
    sigma_scale (int): How many standard deviations above the 
                       median to set the threshold.
    
    Returns:
    float: The calculated voltage threshold.
    """
    data = np.array(voltages)
    
    # 1. Find the median of the signal
    median = np.median(data)
    
    # 2. Calculate the Median Absolute Deviation (MAD)
    # This is the median of the absolute differences from the median
    mad = np.median(np.abs(data - median))
    
    # 3. Convert MAD to an unbiased estimate of Standard Deviation
    # For a normal distribution, sigma is approx 1.4826 * MAD
    sigma_est = mad * 1.4826
    
    # 4. Define threshold as (median + k * sigma)
    threshold = median + (sigma_scale * sigma_est)
    
    return threshold

def keep_consecutive_stable_points(data, tolerance=0.05):
    """
    Zeros out points that deviate too much from the previous point.
    Only 'stable' sequences are kept.
    
    Parameters:
    -----------
    data : array-like
        The input data (e.g., Charge Rate or Mu).
    tolerance : float
        The maximum allowed difference to be considered 'stable'.
        Values with a diff > tolerance will be zeroed.
        
    Returns:
    --------
    np.array
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 2:
        return data

    # 1. Calculate absolute differences between current and previous point
    # We prepend the first value so the first point always has a diff of 0 (it stays)
    diffs = np.abs(np.diff(data, prepend=data[0]))
    
    # 2. Identify 'Jumps': where the difference is OUTSIDE the small range
    mask_jumps = diffs > tolerance
    
    # 3. Zero out the jumps
    filtered_data = data.copy()
    filtered_data[mask_jumps] = 0.0
    
    return filtered_data

def open_top_scope_files(csv_path, top_n=10):
    # 1. Load the data
    df = pd.read_csv(csv_path)
    df = df[df['HV'] == 40]
    
    # 2. Sort by Area_Vs descending and take the top N
    # We use 'nlargest' which is efficient for this specific task
    top_hits = df.nlargest(top_n, 'Area_Vs')
    
    print(f"Opening top {top_n} files based on Area_Vs...")
    
    for index, row in top_hits.iterrows():
        file_path = row['filename_scope']
        file_num = row['filename_scope'].replace("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-11-20/scope-results-2025-11-20-","")
        file_num = file_num.replace(".csv","")
        file_num = int(file_num)
        area_val = row['Area_Vs']

        cleaned_path = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/cleaned_figs/cleaned_scope-results-2025-11-20-{file_num:04d}.png"
        
        if os.path.exists(cleaned_path):
            print(f"Opening: {cleaned_path} (Area: {area_val:.2e})")
            open_file_default(cleaned_path)
        else:
            print(f"Warning: File not found at {cleaned_path}")

def open_img(image_path):
    start = np.nan
    try:
        # Don't use 'with' if you want the image to persist for a moment
        img = Image.open(image_path)
        
        # Re-assign to catch the resized object
        img_small = img.resize((800, 800)) # 100x100 is tiny! Try 800x800
        img_small.show()
        time.sleep(1)
        img.close() # Clean up manually after the input
    except Exception as e:
        print(f"Error in open_img: {e}")
    return start

def open_file_default(filepath):
    """Cross-platform command to open a file with the default application."""
    system_name = platform.system()
    if system_name == 'Windows':
        os.startfile(filepath)
    elif system_name == 'Darwin':  # macOS
        # Inside your loop in open_top_scope_files:
        if os.path.exists(filepath):
            print(f"Opening: {filepath}")
            open_img(filepath)
            time.sleep(0.5)  # Give Preview half a second to initialize
    else:  # Linux
        subprocess.call(('xdg-open', filepath))

# --- Usage ---
# open_top_scope_files("your_data_results.csv")


if __name__ == "__main__":
    sensor = "SiC"
    month, day = 11, 20
    date="2025-11-20"
    idx_list = ['1463','0009','0763']
    pulse_list = [2.0,0.5,1.0]
    dist_list = [40.0,60.0,60.0]
    beam_list = ["Electron 110V","Electrons 85V","Electrons 85V"]
    range_dict = {}
    open_top_scope_files("sic_data.csv")
    