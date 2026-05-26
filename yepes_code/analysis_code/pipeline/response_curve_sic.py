import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
from tqdm import tqdm
import config
import area_testing
from scipy.signal import savgol_filter
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit
from PIL import Image
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from scipy.signal import medfilt


dose_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/dose_csvs/dose_scaling.csv"
dose_scale_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/dose_csvs/dose_scale_factors.csv"

def find_noise_threshold(signal, idx=500):
    first_noise = signal[:idx]
    sigma = np.std(first_noise)
    return max(10 * sigma, 0.001)

def dose_model(d, A, z0, b):
    # A is scale, z0 is the source offset, b is the power (usually near 2)
    return A / ((d + z0)**b)

def convert_dose(dose_ref_csv, dose_scale_factors_csv, beam, dist_m, pulse_width, collimator_length_cm, dist_from_col=True):
    beam = ''.join(filter(str.isdigit, beam))
    # read first csv file containing 85v pulse info
    dose_df = pd.read_csv(dose_ref_csv,skiprows=2)
    # filter by selected collimator length for a 1.0 length pulse
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] != "Uncollimated"]
    dose_df["Collimation (cm, diameter)"] = dose_df["Collimation (cm, diameter)"].astype(float)
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] == float(collimator_length_cm)]
    dose_df = dose_df[dose_df["PW (electron pulse, us, FWHM)"] == 1.01]
    # dist_from_col is a bool which pulls from the appropriate column depending on where dist_cm is measured from
    # the resulting points are used in an exponential regression to fit the inputted distance and extrapolate its dose
    if dist_from_col:
        x_points = dose_df["dist. collimator exit (m)"]
    else:
        x_points = dose_df["dist. beam exit (m)"]
    try:
        # modified inverse square law fit (shifted by an offset to accommodate points near the x minimum and maximum)
        y_points = (dose_df["Gy/P"]).astype(float) #/(dose_df["PW (electron pulse, us, FWHM)"]).astype(float)
        ylog_points = np.log(y_points)
        if config.verbose>1: print(f"log data points: {ylog_points}")
        initial_guess = [max(y_points), 0.05, 2.0]
        # we constrain z0 and b to stay physically realistic
        params, _ = curve_fit(dose_model, x_points, y_points, p0=initial_guess, 
                            bounds=(0, [np.inf, 1.0, 5.0]))
        A_fit, z0_fit, b_fit = params
        # calculate the specific desired dose - note this result is a rate (Gy/P, P=1 us)
        dose_gy = dose_model(dist_m, A_fit, z0_fit, b_fit) # this will give dose in Gy
        '''smooth_range = np.linspace(min(x_points),max(x_points),100)
        print(smooth_range)
        plt.plot(x_points, y_points, label="original fit data")
        plt.plot(smooth_range, dose_model(smooth_range, A_fit, z0_fit, b_fit), label='modified inverse square fit')
        plt.legend()
        plt.show()'''
        # determine dose in gy for 85V beam based on this extrapolated fit and the given distance in m
        # dose_gy = (np.exp(coeffs[1]) * np.exp(coeffs[0]*(float(dist_m))))
        # to change from 1.0 beam (table 1) to 0.5 beam (table 2) divide by the appropriate scaling factor
        dose_gy = dose_gy/1.3

        # Load scale factors
        dsf = pd.read_csv(dose_scale_factors_csv)
        
        # Clean dtypes to ensure the match works regardless of CSV formatting
        dsf["Beam (V)"] = dsf["Beam (V)"].astype(str)
        dsf["Nominal PW (us)"] = dsf["Nominal PW (us)"].astype(float)
        
        # Single robust filter
        matched_factor = dsf[
            (dsf["Beam (V)"] == str(beam)) & 
            (dsf["Nominal PW (us)"] == float(pulse_width))
        ]['Relative output (relative to 85V beam, 0.5us)']

        if matched_factor.empty:
            print(f"!!! MISSING SCALE FACTOR for Beam: {beam}, Pulse: {pulse_width} !!!")
            return np.nan # Using NaN is better than 0 to distinguish "Error" from "No Signal"
        
        scale_factor = matched_factor.values[0]
        if config.verbose > 1: print(f"final dose: {dose_gy * scale_factor}")
        return dose_gy * scale_factor
    except Exception as e:
        print(e)
        print("Error, returning 0 dose")
        return 0
    
def rename_files_in_dir(directory_path):
    for filename in os.listdir(directory_path):
        # Ensure you are not renaming subdirectories by checking if it is a file
        if os.path.isfile(os.path.join(directory_path, filename)):
            # 3. Construct the full source and destination paths
            source = os.path.join(directory_path, filename)
            destination = os.path.join(directory_path, filename.replace(".csv",""))

            # 4. Rename the file
            try:
                os.rename(source, destination)
                print(f"Renamed '{filename}' to '{os.path.basename(destination)}'")
            except FileExistsError:
                print(f"Error: '{os.path.basename(destination)}' already exists. Skipping '{filename}'.")
            except Exception as e:
                print(f"An error occurred while renaming '{filename}': {e}")

def identify_real_curves(signal, noise_floor):
    # Find all peaks above the noise floor
    # 'width' tells scipy to calculate the Full Width at Half Maximum (FWHM)
    peaks, properties = find_peaks(
        signal, 
        height=noise_floor * 1.5, 
        prominence=noise_floor * 1.0,
        width=10 # Minimum number of points wide at half-height
    )
    
    # peaks contains only the indices of peaks that belong to large curves
    return peaks, properties

def return_before_first_spike(signal, buffer=0, max_mode=False):
    # =========================================================================
    # STEP 1: DC BASELINE CORRECTION 
    # =========================================================================
    first_500 = signal[:500] if len(signal) >= 500 else signal
    baseline_offset = np.median(first_500)
    zero_centered_signal = signal - baseline_offset
    
    # =========================================================================
    # STEP 2: ROBUST NOISE AND ENVELOPE CALCULATION
    # =========================================================================
    sigma = np.std(zero_centered_signal[:500])
    if sigma < 0.001: 
        sigma = 0.001
    
    pulse_peak_depth = np.abs(np.min(zero_centered_signal))
    statistical_limit = 10.0 * sigma
    peak_relative_floor = pulse_peak_depth * 0.15
    absolute_hardware_floor = 0.010 
    
    sensitive_limit = max(statistical_limit, peak_relative_floor, absolute_hardware_floor)
    short_window = 12

    # =========================================================================
    # FORWARD PASS (Standard Mode): Pure Split-Window Scan
    # =========================================================================
    pulse_start_idx = None
    skip_until_idx = 0
    clean_until = int(len(signal) * 0.10)

    for idx in range(short_window, len(zero_centered_signal) - short_window):
        if idx < skip_until_idx:
            clean_until = idx
            continue

        left_win = zero_centered_signal[idx - short_window : idx]
        right_win = zero_centered_signal[idx : idx + short_window]
        
        mean_left = np.mean(left_win)
        mean_right = np.mean(right_win)
        
        if mean_right < -sensitive_limit and mean_right < mean_left:
            if np.sum(right_win < -sigma) >= int(0.90 * short_window):
                pulse_start_idx = idx
                break

        if np.abs(zero_centered_signal[idx]) <= sensitive_limit:
            clean_until = idx
        else:
            pct_dropping = np.sum(np.diff(right_win) < 0) / len(right_win)
            if pct_dropping < 0.45:
                clean_until = idx
                skip_until_idx = idx + 6  

    # =========================================================================
    # STEP 3: DYNAMIC DERIVATIVE BOUNDARY SNAPPING
    # =========================================================================
    if pulse_start_idx is not None:
        corrected_start = pulse_start_idx
        
        while corrected_start > 3:
            local_check = zero_centered_signal[corrected_start - 3 : corrected_start]
            if np.mean(local_check) >= -1.5 * sigma:
                break
            corrected_start -= 1
            
        while corrected_start > 1:
            current_point = zero_centered_signal[corrected_start]
            left_point = zero_centered_signal[corrected_start - 1]
            local_slope = current_point - left_point
            if local_slope >= -0.5 * sigma:
                break
            corrected_start -= 1
            
        clean_until = corrected_start
        
        actual_buffer = 0
        for b in range(1, buffer + 1):
            target_idx = clean_until + b
            if target_idx >= len(signal):
                break
            if zero_centered_signal[target_idx] < -sensitive_limit:
                break
            actual_buffer = b
        clean_until = clean_until + actual_buffer
    else:
        clean_until = int(len(signal) * 0.10)

    clean_until = max(0, min(clean_until, len(signal) - 1))

    cleaned_signal = signal.copy()
    cleaned_signal[0 : clean_until] = 0
    
    return cleaned_signal, clean_until, sensitive_limit

'''
def return_before_first_spike(signal, buffer=0, max_mode=False):
    # =========================================================================
    # STEP 1: DC BASELINE CORRECTION 
    # =========================================================================
    first_500 = signal[:500] if len(signal) >= 500 else signal
    baseline_offset = np.median(first_500)
    zero_centered_signal = signal - baseline_offset
    
    # =========================================================================
    # STEP 2: ROBUST NOISE AND ENVELOPE CALCULATION
    # =========================================================================
    sigma = np.std(zero_centered_signal[:500])
    if sigma < 0.001: 
        sigma = 0.001
    
    pulse_peak_depth = np.abs(np.min(zero_centered_signal))
    statistical_limit = 10.0 * sigma
    peak_relative_floor = pulse_peak_depth * 0.15
    absolute_hardware_floor = 0.010 
    
    sensitive_limit = max(statistical_limit, peak_relative_floor, absolute_hardware_floor)
    short_window = 12

    # =========================================================================
    # MODE 2: BACKWARD PASS (max_mode=True) - Cleans Tail Spikes
    # =========================================================================
    if max_mode:
        flipped_signal = np.flip(zero_centered_signal)
        first_spike_idx = np.argmax(np.abs(flipped_signal))
        
        # Convert the true peak index back to standard forward coordinates
        forward_peak_idx = len(signal) - 1 - first_spike_idx
        
        # Default fallback: Start cleaning from the peak to the end of the file
        clean_from = forward_peak_idx 
        
        # Scan from the end of the flipped signal (the true tail end) up toward the peak
        for idx in range(short_window, first_spike_idx):
            next_block = flipped_signal[idx : idx + short_window]
            diffs = np.diff(next_block)
            pct_increasing = np.sum(diffs >= 0) / len(diffs)
            
            # The moment the tail stops being flat baseline and starts dropping/moving
            # toward the main peak body, establish the cut-off boundary.
            if pct_increasing < 0.75:
                forward_idx = len(signal) - 1 - idx
                clean_from = forward_idx
                break
                
        # Apply user buffer, ensuring we never push the boundary past the peak itself
        clean_from = max(forward_peak_idx, min(clean_from - buffer, len(signal) - 1))
        
        cleaned_signal = signal.copy()
        
        # WIPE TAIL SPIKES: Zero out everything after our detected boundary
        cleaned_signal[clean_from : len(signal)] = 0
        
        return cleaned_signal, clean_from, sensitive_limit
    # =========================================================================
    # MODE 1: FORWARD PASS (max_mode=False) - Cleans Prefix Spikes
    # =========================================================================
    pulse_start_idx = None
    skip_until_idx = 0
    clean_until = int(len(signal) * 0.10)

    for idx in range(short_window, len(zero_centered_signal) - short_window):
        if idx < skip_until_idx:
            clean_until = idx
            continue

        left_win = zero_centered_signal[idx - short_window : idx]
        right_win = zero_centered_signal[idx : idx + short_window]
        
        mean_left = np.mean(left_win)
        mean_right = np.mean(right_win)
        
        if mean_right < -sensitive_limit and mean_right < mean_left:
            if np.sum(right_win < -sigma) >= int(0.90 * short_window):
                pulse_start_idx = idx
                break

        if np.abs(zero_centered_signal[idx]) <= sensitive_limit:
            clean_until = idx
        else:
            pct_dropping = np.sum(np.diff(right_win) < 0) / len(right_win)
            if pct_dropping < 0.45:
                clean_until = idx
                skip_until_idx = idx + 6  

    # =========================================================================
    # STEP 3: DYNAMIC BOUNDARY SNAPPING (Restored with Tight Window Boundary)
    # =========================================================================
    if pulse_start_idx is not None:
        # Define a strictly constrained zone to isolate ONLY the initial spike drop
        # This prevents it from looking at the main pulse body further down the file
        max_search_limit = min(pulse_start_idx + short_window, len(zero_centered_signal) - 1)
        spike_body_zone = zero_centered_signal[pulse_start_idx : max_search_limit]
        
        if len(spike_body_zone) > 0:
            local_drop_end = np.argmin(spike_body_zone)
            absolute_spike_end = pulse_start_idx + local_drop_end
            
            # Stop zeroing out exactly where this initial spike drop ends
            clean_until = absolute_spike_end + 1
        else:
            clean_until = pulse_start_idx

        # !!! BUFFER FIX: Subtract the buffer to move the clean zone LEFT (backward),
        # which safely protects the signal from being eaten.
        clean_until = max(0, clean_until - buffer)
    else:
        # Safe fallback: only zero out the first 10% if no spike triggers
        clean_until = int(len(signal) * 0.10)

    # Hard array limit clamp to prevent complete file obliteration
    clean_until = max(0, min(clean_until, len(signal) - 1))

    # WIPE UNWANTED REGION (Wipes everything from 0 up to the frame where the spike ends)
    cleaned_signal = signal.copy()
    cleaned_signal[0 : clean_until] = 0
    
    return cleaned_signal, clean_until, sensitive_limit
'''

def get_drop_event(time, signal):
    # Process standard forward signal pass
    final_clean, end_of_error, threshold = return_before_first_spike(signal, max_mode=False)
    
    manual_min = np.argmin(final_clean)
    manual_max = np.argmax(final_clean)
    
    # Calculate baseline locally for zero-centering
    first_500 = signal[:500] if len(signal) >= 500 else signal
    baseline_offset = np.median(first_500)
    
    sigma = np.std(signal[:500] - baseline_offset)
    if sigma < 0.001: 
        sigma = 0.001
    
    drop_threshold = -threshold
    search_zone = final_clean[end_of_error:]
    drop_indices = np.where(search_zone < drop_threshold)[0]
    
    # ---------------------------------------------------------------------
    # !!! ADD THE DENSITY CONDITION HERE !!!
    # ---------------------------------------------------------------------
    # Set your minimum required consecutive/total points (e.g., 15 points)
    MIN_REQUIRED_POINTS = 15 
    
    if len(drop_indices) < MIN_REQUIRED_POINTS:
        if config.verbose > 1:
            print(f"REJECTED: Only found {len(drop_indices)} points below threshold. Expected >= {MIN_REQUIRED_POINTS}.")
        # Exit early: returns No Signal Detected, forcing the plot to say "No Drop Found"
        return final_clean - baseline_offset, False, None, None, drop_threshold, None, None
    # ---------------------------------------------------------------------
    
    # If it passes the check, proceed with normal processing
    if len(drop_indices) > 0:
        start_idx = end_of_error + drop_indices[0]
        
        # Aggressive depth validation check
        max_drop_depth = np.abs(np.min(final_clean))
        if max_drop_depth < (8.0 * sigma) or max_drop_depth < 0.015:
            return final_clean - baseline_offset, False, None, None, drop_threshold, None, None
        
        zero_centered_raw = signal - baseline_offset
        shifted_start = area_testing.backtrack_to_local_max(time, zero_centered_raw, start_idx)
        
        if config.verbose > 1:
            print(f"shifted start dif: {shifted_start - start_idx}")
            
        recovery = np.where(final_clean[start_idx:] >= 0)[0]
        end_idx = shifted_start + recovery[0] if len(recovery) > 0 else len(signal)
        
        return final_clean - baseline_offset, True, shifted_start, end_idx, drop_threshold, manual_min, manual_max
    
    return final_clean - baseline_offset, False, None, None, drop_threshold, None, None

def plot_averaged_linearity(csv_file, detector_name="SiC", min_area_threshold=1e-10):
    # 1. Load and Prepare
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06

    # Put this right before creating your mask to see why rows are dying
    print(f"Total raw points loaded: {len(df)}")
    print(f" -> Pulse > 0.5 count: {sum(df['Pulse'] > 0.5)}")
    print(f" -> DoseRate_sec > 0 count: {sum((df['Dose'] / df['Pulse'] * 1e6) > 0)}")
    print(f" -> HV == 40 count: {sum(df['HV'].astype(float) == 40.0)}")
    print(f" -> Area above floor count: {sum(df['Area'] > 1e-10)}")
    print(f" -> Area below ceiling count: {sum(df['Area'] < 2e-08)}")
    print(f" -> Unique values in Beam column: {df['Beam'].unique()}")

    # Isolate just the 40V data to see which filter kills it
    df_40 = df[df['HV'].astype(float) == 40.0]

    print(f"--- 40V SUBSET AUDIT ---")
    print(f"Total 40V points raw: {len(df_40)}")
    print(f" -> Passing Pulse > 0.5: {sum(df_40['Pulse'] > 0.5)}")
    print(f" -> Passing Area > floor: {sum(df_40['Area'] > 1e-10)}")
    print(f" -> Passing Area < ceiling: {sum(df_40['Area'] < 2e-08)}")
    print(f" -> Not Electron 85V: {sum(df_40['Beam'] != 'Electron 85V')}")
            
    # 2. Filtering
    mask = (
        (df['Area'] > min_area_threshold) & 
        (df['DoseRate_sec'] > 0) &
        (df['Pulse'] > 0.5) & 
        (df['Beam'] != "Electron 85V")
    )
    df_clean = df[mask].copy()

    # 3. Group by Configuration (HV, Beam, Pulse)
    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        # 4. Calculate Mean and Std Dev for each unique Dose level
        # We group by 'DoseRate_sec' because that is our X-axis
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std', 'count']).reset_index()
        
        # If there's only one pulse per dose, std will be NaN; fill with 0
        stats_df['std'] = stats_df['std'].fillna(0)

        if len(stats_df) < 2:
            continue

        # 5. Create Figure
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot with Error Bars
        plt.errorbar(
            stats_df['DoseRate_sec'], 
            stats_df['mean'], 
            yerr=stats_df['std'], 
            fmt='o',           # Circle markers
            capsize=5,         # Top/bottom caps on error bars
            color='darkblue', 
            ecolor='red',      # Red error bars for visibility
            label='Mean Area ± 1σ'
        )

        # 6. Linear Fit on the Averaged Data
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            stats_df['DoseRate_sec'], stats_df['mean']
        )
        
        # Plot the Fit Line
        x_fit = stats_df['DoseRate_sec']
        plt.plot(x_fit, slope * x_fit + intercept, color='gray', linestyle='--', alpha=0.7,
                 label=f'Linear Fit (R²={r_value**2:.4f})')

        # Formatting
        plt.title(f"Averaged Response: {detector_name}\nHV: {hv} | {beam} | {pulse}µs", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Averaged Integrated Area (V·s)", fontsize=11)
        
        # Stats Box
        fit_text = f"Slope: {slope:.2e}\nR²: {r_value**2:.4f}\nN points: {len(group)}"
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.legend()
        plt.tight_layout()

def plot_dynamic_linear_fit(csv_file, detector_name="SiC", r2_threshold=0.999):
    """
    1) Plots averaged data with error bars.
    2) Dynamically finds the longest linear sequence starting from the lowest dose.
    3) Extrapolates the linear line to show the 'Saturation Droop'.
    """
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0) & (df['Beam'] != "Electron 85V")
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        # Aggregate stats
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().sort_values('DoseRate_sec')
        x_data = stats_df['DoseRate_sec'].values
        y_data = stats_df['mean'].values

        if len(x_data) < 3:
            continue

        # --- DYNAMIC LINEAR REGION DETECTION ---
        best_idx = 2
        final_r2 = 0
        
        for i in range(3, len(x_data) + 1):
            x_subset = x_data[:i]
            y_subset = y_data[:i]
            
            slope, intercept, r_val, _, _ = stats.linregress(x_subset, y_subset)
            current_r2 = r_val**2
            
            if current_r2 >= r2_threshold:
                best_idx = i
                final_r2 = current_r2
            else:
                # If adding this point breaks linearity, stop here
                break

        # Final fit on the identified linear region
        slope, intercept, r_val, _, _ = stats.linregress(x_data[:best_idx], y_data[:best_idx])

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot all data (Gray out saturated points)
        plt.errorbar(x_data, y_data, yerr=stats_df['std'], fmt='o', color='black', alpha=0.3)
        
        # Highlight Linear Region
        # plt.scatter(x_data[:best_idx], y_data[:best_idx], color='forestgreen', s=80, label='Identified Linear Region', zorder=5)

        # Plot Linear Fit (Extrapolated)
        # x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        # plt.plot(x_fit, slope * x_fit + intercept, color='red', linestyle='--', label=f'Linear Fit (R²={final_r2:.5f})')

        # Formatting
        plt.title(f"Dynamic Linearity Analysis: {detector_name}\nHV: {hv} | {beam} | {pulse}µs", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Integrated Area (V·s)", fontsize=11)
        
        # Vertical Line at Saturation Point
        sat_point = x_data[best_idx-1]
        # plt.axvline(sat_point, color='orange', linestyle=':', label=f'Saturation Start (~{sat_point:.1e} Gy/s)')

        fit_text = f"Sensitivity: {slope:.2e} V·s/(Gy/s)\nLinear Limit: {sat_point:.2e} Gy/s"
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

def plot_dynamic_quadratic_fit(csv_file, detector_name="SiC", r2_threshold=0.999):
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0) & (df['Beam'] != "Electron 85V")
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().sort_values('DoseRate_sec')
        x_data = stats_df['DoseRate_sec'].values
        y_data = stats_df['mean'].values
        print(f"Processing Group: {hv}, {beam}, {pulse} - Points: {len(x_data)}")

        if len(x_data) < 4: # Quadratic fits need more points to be meaningful
            continue

        # --- DYNAMIC QUADRATIC REGION DETECTION ---
        best_idx = 3
        final_r2 = 0
        best_coeffs = None
        
        for i in range(4, len(x_data) + 1):
            x_subset = x_data[:i]
            y_subset = y_data[:i]
            
            # Fit 2nd degree polynomial: y = ax^2 + bx + c
            coeffs = np.polyfit(x_subset, y_subset, 2)
            p = np.poly1d(coeffs)
            
            # Calculate R² manually
            y_pred = p(x_subset)
            current_r2 = r2_score(y_subset, y_pred)
            
            if current_r2 >= r2_threshold:
                best_idx = i
                final_r2 = current_r2
                best_coeffs = coeffs
            else:
                break

        if best_coeffs is None:
            print(f"SKIPPED: {hv}V {pulse}us - Could not meet R2 threshold of {r2_threshold}")
            continue

        # Create the model function for plotting
        quadratic_model = np.poly1d(best_coeffs)

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot all data
        plt.errorbar(x_data, y_data, yerr=stats_df['std'], fmt='o', color='black', alpha=0.3, label='Data Points')
        
        # Highlight Quadratic Region
        plt.scatter(x_data[:best_idx], y_data[:best_idx], color='royalblue', s=80, label='Identified Quadratic Region', zorder=5)

        # Plot Quadratic Fit (Extrapolated)
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        plt.plot(x_fit, quadratic_model(x_fit), color='crimson', linestyle='--', label=f'Quad Fit (R²={final_r2:.5f})')

        # Formatting
        plt.title(f"Dynamic Quadratic Analysis: {detector_name}\nHV: {hv} | {beam} | {pulse}µs", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Integrated Area (V·s)", fontsize=11)
        
        sat_point = x_data[best_idx-1]
        plt.axvline(sat_point, color='orange', linestyle=':', label=f'Fit Limit (~{sat_point:.1e} Gy/s)')

        # Equation string for the text box
        a, b, c = best_coeffs
        fit_text = (f"y = {a:.2e}x² + {b:.2e}x + {c:.2e}\n"
                    f"R² = {final_r2:.5f}\n"
                    f"Limit: {sat_point:.2e} Gy/s")
        
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # --- Inside the Loop ---
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save instead of just showing
        save_name = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/generated_plots/QuadFit_{detector_name}_{hv}V_{pulse}us_{beam}.png"
        plt.show()
        plt.pause(0.1) # Brief pause to allow the GUI to draw the window
        plt.savefig(save_name, dpi=150)
        plt.close()
        
        # CRITICAL: Close the figure to free up RAM
        plt.close() 
        print(f"Saved plot: {save_name}")

def plot_quadratic_fit_no_threshold(csv_file, detector_name="SiC"):
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0) & (df['Beam'] != "Electron 85V")
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().sort_values('DoseRate_sec')
        x_data = stats_df['DoseRate_sec'].values
        y_data = stats_df['mean'].values

        # Requirement: You still need at least 3 points to define a parabola
        if len(x_data) < 3:
            print(f"Skipping {hv}V {pulse}us: Only {len(x_data)} points available.")
            continue

        # --- FIT ENTIRE DATA RANGE ---
        # We no longer loop to find a 'best_idx'. We use everything.
        coeffs = np.polyfit(x_data, y_data, 2)
        quadratic_model = np.poly1d(coeffs)
        
        # Calculate R2 just for display purposes
        y_pred = quadratic_model(x_data)
        final_r2 = r2_score(y_data, y_pred)

        # --- PLOTTING ---
        fig = plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot all data
        plt.errorbar(x_data, y_data, yerr=stats_df['std'], fmt='o', 
                     color='black', label='Experimental Data', zorder=3)
        
        # Plot Quadratic Fit across the full range
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        plt.plot(x_fit, quadratic_model(x_fit), color='crimson', 
                 linewidth=2, label=f'Full Quad Fit (R²={final_r2:.5f})')

        # Formatting
        plt.title(f"Quadratic Regression: {detector_name}\nHV: {hv} | {beam} | {pulse}µs", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Integrated Area (V·s)", fontsize=11)
        
        # Equation Display
        a, b, c = coeffs
        fit_text = (f"y = ({a:.2e})x² + ({b:.2e})x + {c:.2e}\n"
                    f"R² = {final_r2:.5f}")
        
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', family='monospace', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Display and then Close to prevent memory/display issues
        plt.show()
        plt.pause(0.1) 
        plt.close(fig)

def plot_dynamic_exponential_fit(csv_file, detector_name="SiC", r2_threshold=0.99):
    """
    1) Uses Exponential Fit: Area = A * exp(B * DoseRate)
    2) Dynamically detects the best exponential region by monitoring R-squared.
    """
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    # Pre-filter for valid signal
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0)
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        # Aggregate to mean values for fitting
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().sort_values('DoseRate_sec')
        
        x_data = stats_df['DoseRate_sec'].values
        y_data = stats_df['mean'].values

        if len(x_data) < 3:
            continue

        # --- DYNAMIC REGION DETECTION ---
        # We find the longest sequence starting from point 0 that maintains a high R^2 in log-space
        best_idx = 2
        for i in range(3, len(x_data) + 1):
            x_part = x_data[:i]
            y_part = np.log(y_data[:i]) # Linearize exponential for regression
            
            _, _, r_val, _, _ = stats.linregress(x_part, y_part)
            if (r_val**2) >= r2_threshold:
                best_idx = i
            else:
                break # Stop if the fit quality degrades

        # Final Fit on the detected region
        x_fit_region = x_data[:best_idx]
        y_fit_region_log = np.log(y_data[:best_idx])
        
        slope, intercept, r_val, _, _ = stats.linregress(x_fit_region, y_fit_region_log)
        
        # Coefficients for Area = A * exp(B * x)
        A = np.exp(intercept)
        B = slope

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # 1. Plot all data points
        plt.errorbar(x_data, y_data, yerr=stats_df['std'], fmt='o', color='gray', alpha=0.4, label='Outside Fit Region')
        
        # 2. Highlight detected exponential region
        plt.scatter(x_data[:best_idx], y_data[:best_idx], color='magenta', s=70, label='Detected Exp Region', zorder=5)

        # 3. Plot the Exponential Curve
        # Generate smooth x-values for the curve
        x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
        y_smooth = A * np.exp(B * x_smooth)
        plt.plot(x_smooth, y_smooth, color='darkviolet', linestyle='--', linewidth=2, 
                 label=f'Exp Fit (R²={r_val**2:.4f})')

        # Formatting
        plt.title(f"Dynamic Exponential Fit: {detector_name} | HV: {hv} | {beam}", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Integrated Area (V·s)", fontsize=11)
        plt.yscale('log') # Use log scale to visually verify the exponential linearity
        
        fit_text = f"Model: Area = {A:.2e} * exp({B:.2e} * DoseRate)\nRegion Points: {best_idx}/{len(x_data)}"
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend()
        plt.tight_layout()
        plt.show()

    return

def plot_and_fit_linear_regions(csv_file, detector_name="SiC", min_area_threshold=1e-10):
    # 1. Load and Prepare
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    # 2. Filtering
    mask = (
        (df['Area'] > min_area_threshold) & 
        (df['Area'] < 2e-08) &
        (df['DoseRate_sec'] > 0) &
        (df['Pulse'] > 0.5) & 
        (df['Beam'] != "Electron 85V") # &
        # (df['HV'] == 40.0)
    )
    df_clean = df[mask].copy()

    # 3. Iterate through every unique combination
    # We group by HV, Beam, and Pulse to isolate individual curves
    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        if len(group) < 3: # Need at least 3 points for a meaningful fit
            continue
            
        # Sort for plotting
        group = group.sort_values('DoseRate_sec')
        x = group['DoseRate_sec'].values
        y = group['Area'].values

        # 4. Perform Linear Regression
        # stats.linregress returns: slope, intercept, r_value, p_value, std_err
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value**2

        # 5. Create Figure
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot Data Points
        plt.scatter(x, y, label='Experimental Data', color='royalblue', s=50)
        
        # Plot Linear Fit Line
        line = slope * x + intercept
        plt.plot(x, line, color='red', linestyle='--', alpha=0.8, 
                 label=f'Linear Fit (R²={r_squared:.4f})')

        # Formatting
        plt.title(f"{detector_name} Linearity | HV: {hv} | {beam} | {pulse}µs", fontsize=14)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=12)
        plt.ylabel("Integrated Area (V·s)", fontsize=12)
        
        # Add Equation Text Box
        fit_text = f"Slope: {slope:.2e}\nIntercept: {intercept:.2e}\nR²: {r_squared:.4f}"
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    return len(groups)

def plot_total_dose_vs_area(csv_file, detector_name="SiC", HV=40.0, min_area_threshold=1e-10):
    """
    Plots Total Dose per Pulse (Gy) vs Area, creating a separate 
    plot for each unique HV value found in the data.
    """
    # 1. Load the data
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty.")
        return 0.0

    # 2. Data Preparation
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['Pulse_Label'] = df['Pulse'].apply(lambda x: f"{x} µs" if pd.notnull(x) else "Unknown")
    
    # We use 'Dose' directly (assumed to be Gy per pulse in your CSV)
    # Ensure it is numeric
    df['Dose'] = pd.to_numeric(df['Dose'], errors='coerce')

    # Put this right before creating your mask to see why rows are dying
    print(f"Total raw points loaded: {len(df)}")
    print(f" -> Pulse > 0.5 count: {sum(df['Pulse'] > 0.5)}")
    print(f" -> DoseRate_sec > 0 count: {sum((df['Dose'] / df['Pulse'] * 1e6) > 0)}")
    print(f" -> HV == 40 count: {sum(df['HV'].astype(float) == 40.0)}")
    print(f" -> Area above floor count: {sum(df['Area'] > 1e-10)}")
    print(f" -> Area below ceiling count: {sum(df['Area'] < 2e-08)}")
    print(f" -> Unique values in Beam column: {df['Beam'].unique()}")

    # Isolate just the 40V data to see which filter kills it
    df_40 = df[df['HV'].astype(float) == 40.0]

    print(f"--- 40V SUBSET AUDIT ---")
    print(f"Total 40V points raw: {len(df_40)}")
    print(f" -> Passing Pulse > 0.5: {sum(df_40['Pulse'] > 0.5)}")
    print(f" -> Passing Area > floor: {sum(df_40['Area'] > 1e-10)}")
    print(f" -> Passing Area < ceiling: {sum(df_40['Area'] < 2e-08)}")
    print(f" -> Not Electron 85V: {sum(df_40['Beam'] != 'Electron 85V')}")

    # 3. Apply Filters 
    # (Updated to filter based on Dose instead of DoseRate_sec)
    mask = (
        df['Area'].notna() & 
        (df['Area'] > min_area_threshold) & 
        df['Dose'].notna() & 
        (df['Dose'] > 0) &
        (df['Pulse'] > 0.5)
    )
    df_clean = df[mask].copy()

    print("Post-filtering bucket sizes:")
    print(df_clean.groupby(['HV', 'Beam', 'Pulse','Z']).size())


    if df_clean.empty:
        print("No valid data points left after filtering.")
        return 0.0

    # 4. Loop through unique HV values
    unique_hvs = df['HV'].unique()
    for hv_val in unique_hvs:
        hv_df = df_clean[df_clean['HV'] == hv_val].copy()
        # Sort by Dose now for a clean line plot
        hv_df = hv_df.sort_values(by=['Beam', 'Pulse', 'Dose'])

        # 5. Create the Plot
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        sns.lineplot(
            data=hv_df, 
            x='Dose', # Changed from DoseRate_sec
            y='Area', 
            hue='Pulse_Label', 
            style='Beam', 
            markers=True, 
            dashes=True,
            linewidth=2,
            palette="viridis"
        )
        
        plt.title(f"Dose Linearity ({detector_name}) | HV: {hv_val}", fontsize=14)
        plt.xlabel("Total Dose per Pulse (Gy)", fontsize=12) # Updated Label
        plt.ylabel("Integrated Area (V·s)", fontsize=12)
        plt.legend(title="Pulse Width & Beam", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

    total_points = len(df)
    filter_percentage = ((total_points - len(df_clean)) / total_points) * 100
    return filter_percentage

def plot_instantaneous_dose_vs_area(csv_file, detector_name="SiC", HV=40.0, min_area_threshold=1e-10):
    """
    Plots Instantaneous Dose Rate (Gy/s) vs Area, creating a separate 
    plot for each unique HV value found in the data.
    """
    # 1. Load the data
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty.")
        return 0.0

    # 2. Data Preparation
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['Pulse_Label'] = df['Pulse'].apply(lambda x: f"{x} µs" if pd.notnull(x) else "Unknown")
    
    # Calculate Gy/s: (Gy / Pulse_us) * 1e6
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06

    # Put this right before creating your mask to see why rows are dying
    print(f"Total raw points loaded: {len(df)}")
    print(f" -> Pulse > 0.5 count: {sum(df['Pulse'] > 0.5)}")
    print(f" -> DoseRate_sec > 0 count: {sum((df['Dose'] / df['Pulse'] * 1e6) > 0)}")
    print(f" -> HV == 40 count: {sum(df['HV'].astype(float) == 40.0)}")
    print(f" -> Area above floor count: {sum(df['Area'] > 1e-10)}")
    print(f" -> Area below ceiling count: {sum(df['Area'] < 2e-08)}")
    print(f" -> Unique values in Beam column: {df['Beam'].unique()}")

    # Isolate just the 40V data to see which filter kills it
    df_40 = df[df['HV'].astype(float) == 40.0]

    print(f"--- 40V SUBSET AUDIT ---")
    print(f"Total 40V points raw: {len(df_40)}")
    print(f" -> Passing Pulse > 0.5: {sum(df_40['Pulse'] > 0.5)}")
    print(f" -> Passing Area > floor: {sum(df_40['Area'] > 1e-10)}")
    print(f" -> Passing Area < ceiling: {sum(df_40['Area'] < 2e-08)}")
    print(f" -> Not Electron 85V: {sum(df_40['Beam'] != 'Electron 85V')}")

    # 3. Apply Filters
    mask = (
        df['Area'].notna() & 
        (df['Area'] > min_area_threshold) & 
        df['DoseRate_sec'].notna() & 
        (df['DoseRate_sec'] > 0) &
        (df['Pulse'] > 0.5)
    )
    df_clean = df[mask].copy()

    print("Post-filtering bucket sizes:")
    print(df_clean.groupby(['HV', 'Beam', 'Pulse']).size())

    if df_clean.empty:
        print("No valid data points left after filtering.")
        return 0.0

    # 4. Loop through unique HV values
    unique_hvs = df['HV'].unique()
    print(f"Generating plots for HV values: {unique_hvs}")

    for hv_val in unique_hvs:
        # Filter dataframe for this specific HV
        hv_df = df_clean[df_clean['HV'].astype(float) == float(hv_val)].copy()
        hv_df = hv_df.sort_values(by=['Beam', 'Pulse', 'DoseRate_sec'])

        # 5. Create the Plot
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        sns.lineplot(
            data=hv_df, 
            x='DoseRate_sec', 
            y='Area', 
            hue='Pulse_Label', 
            style='Beam', 
            markers=True, 
            dashes=True,
            linewidth=2,
            palette="viridis"
        )
        
        plt.title(f"Response Curve ({detector_name}) | HV: {hv_val}", fontsize=14)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=12)
        plt.ylabel("Integrated Area (V·s)", fontsize=12)
        plt.legend(title="Pulse Width & Beam", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Optional: Save each plot automatically
        # plt.savefig(f"response_curve_HV_{hv_val}.png")
        
        plt.show()

    total_points = len(df)
    filter_percentage = ((total_points - len(df_clean)) / total_points) * 100
    return filter_percentage

def plot_selective_linear_fit(csv_file, detector_name="SiC", max_linear_dose=4e06):
    """
    Plots averaged data but only fits the linear regression to points 
    where DoseRate_sec <= max_linear_dose.
    """
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    # Global filters (Noise floor and Beam energy)
    mask = (df['Area'] > 1e-11) & (df['DoseRate_sec'] > 0) & (df['Beam'] != "Electron 85V")
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        # 1. Aggregate stats for the plot
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().fillna(0)
        
        # 2. ISOLATE THE LINEAR REGION FOR FITTING
        # We only take points below the saturation threshold
        linear_region = stats_df[stats_df['DoseRate_sec'] <= max_linear_dose]
        
        if len(linear_region) < 2:
            continue

        # 3. Calculate Fit (Linear Region Only)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            linear_region['DoseRate_sec'], linear_region['mean']
        )

        # 4. Create Figure
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot ALL points (including saturated ones)
        plt.errorbar(stats_df['DoseRate_sec'], stats_df['mean'], yerr=stats_df['std'], 
                     fmt='o', color='gray', alpha=0.5, label='All Data')
        
        # Highlight the Linear Region points
        plt.scatter(linear_region['DoseRate_sec'], linear_region['mean'], 
                    color='darkblue', s=80, label='Linear Region (Used for Fit)')

        # Plot Fit Line (Extrapolated across the whole range to show deviation)
        x_range = np.linspace(stats_df['DoseRate_sec'].min(), stats_df['DoseRate_sec'].max(), 100)
        plt.plot(x_range, slope * x_range + intercept, color='red', linestyle='--', 
                 label=f'Linear Fit (R²={r_value**2:.4f})')

        # 5. Formatting
        plt.title(f"Segmented Fit: {detector_name} | HV: {hv} | {beam}", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)", fontsize=11)
        plt.ylabel("Averaged Area (V·s)", fontsize=11)
        
        # Add a vertical line showing where the fit stops
        plt.axvline(max_linear_dose, color='orange', alpha=0.3, linestyle=':', label='Fit Cutoff')

        fit_text = f"Linear Sensitivity: {slope:.2e} V·s/(Gy/s)\nR²: {r_value**2:.4f}"
        plt.text(0.05, 0.92, fit_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.legend()
        plt.tight_layout()
        plt.show()

def calculate_drop_area(signal, time, start, end):
    segment_signal = signal[start:end]
    smoothed_segment_signal = savgol_filter(signal, 200, 2)
    segment_time = time[start:end]
    # np.trapz uses the actual time values to calculate the physical area
    area = np.trapz(segment_signal, segment_time)
    return abs(area)

def find_nearest(array, value):
    array = np.asarray(array)
    # Find the index of the minimum absolute difference
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def plot_dose_vs_area_by_energy(csv_file, detector_name="SiC", min_area_threshold=1e-10):
    """
    Plots Dose vs Area, filtering out zero doses and areas below the noise floor.
    
    Args:
        csv_file: Path to the generator output.
        detector_name: Name of sensor for plot title.
        min_area_threshold: The 'Area' (V*s) below which points are considered noise.
    """
    # 1. Load the data
    df = pd.read_csv(csv_file)
    total_points = len(df)
    
    if total_points == 0:
        print("CSV file is empty.")
        return 0.0

    # 2. Advanced Filtering
    # Filters: 
    # - Area must be non-NaN and greater than our custom noise floor
    # - Dose must be non-NaN and greater than 0
    mask = (
        df['Area'].notna() & 
        (df['Area'] > min_area_threshold) & 
        df['Dose'].notna() & 
        (df['Dose'] > 0)
    )
    df_clean = df[mask].copy()
    
    # 3. Calculate Filter Percentage
    kept_points = len(df_clean)
    removed_points = total_points - kept_points
    filter_percentage = (removed_points / total_points) * 100
    
    print(f"--- Data Integrity Report ---")
    print(f"Total points in CSV: {total_points}")
    print(f"Points filtered (Area < {min_area_threshold:.1e} or Dose <= 0): {removed_points}")
    print(f"Filter Rate: {filter_percentage:.2f}%")
    print(f"-----------------------------")

    if df_clean.empty:
        print("No valid data points left after filtering.")
        return filter_percentage

    # 4. Sorting for clean line connections
    df_clean = df_clean.sort_values(by=['Beam', 'Dose'])

    # 5. Create Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.lineplot(
        data=df_clean, 
        x='Dose', 
        y='Area', 
        hue='Beam', 
        style='Beam', 
        markers=True, 
        dashes=False,
        linewidth=2
    )
    
    plt.title(f"Response Curve: Dose Rate vs. Area ({detector_name})", fontsize=14)
    plt.xlabel("Extrapolated Dose (Gy/sec)", fontsize=12)
    plt.ylabel("Integrated Area (V·s)", fontsize=12)
    plt.legend(title="Beam Energy", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return filter_percentage

def calculate_drop_area_sensitive(signal, time, start, end):
    """
    Improved integration that subtracts the local baseline 
    to prevent small signals from being 'washed out'.
    """
    # 1. Extract the segment
    v_segment = signal[start:end]
    t_segment = time[start:end]
    
    if len(v_segment) < 2:
        return 0.0

    # 2. Local Baseline Correction
    # We take the average of the first 20 points of the window 
    # (assuming the signal hasn't dropped yet) to find 'zero'
    local_zero = np.mean(v_segment[:20])
    v_corrected = v_segment - local_zero
    
    # 3. Smoothing (Optional but helpful for weak 191V signals)
    # Using a smaller window than 200 to avoid 'flattening' small peaks
    if len(v_corrected) > 31:
        v_final = savgol_filter(v_corrected, 31, 2)
    else:
        v_final = v_corrected

    # 4. Integrate
    # We use the absolute value because 'drops' are negative (V*s)
    area = np.trapz(v_final, t_segment)
    return abs(area)

# --- Execution ---
def denoise_and_get_area(min_file, max_file, sensor, date, args):
    file_range = range(min_file, max_file + 1)
    areas = []
    doses = []
    filepaths = []
    manual_diff = None

    for file_num in file_range:
        arg_string = ""
        for key, value in args.items():
            arg_string += f"{key}: {value}\n"
            
        file_path = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{str(file_num).zfill(4)}.csv"
        df = pd.read_csv(file_path)
        raw_signal = df["CH1"].values
        time = df["TIME"].values
        
        # 1. Process Signal (Unpacks original 7 arguments exactly)
        final_clean, found, start, end, d_thresh, local_min, local_max = get_drop_event(time, raw_signal)

        # 2. Setup Plot Canvas Context
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Calculate local raw baseline for zero-centering on the plot
        first_500 = raw_signal[:500] if len(raw_signal) >= 500 else raw_signal
        baseline_offset = np.median(first_500)
        aligned_raw_signal = raw_signal - baseline_offset

        if found:
            fallback = False
            print(f"DEBUG PIPELINE -> Raw Start: {start}, Raw End: {end}, Total Length: {len(time)}")

            start = min(int(start), len(time) - 1)
            end = min(int(end), len(time) - 1)

            manual_diff = time[end] - time[start]
            if manual_diff < 0.1e-06:
                manual_diff = time[end] - time[start]
                fallback = True
                arg_string += "WARNING: error calculating time, using signal width fallback\n"
            
            arg_string += f"actual pulse length: {manual_diff/1e-6:.2f} us\n"
            arg_string += f"resorted pulse length: {find_nearest([0.5, 1.0, 2.0, 3.0], manual_diff/1e-06)}\n"
        else:
            manual_diff = None

        # =========================================================================
        # PLOT RENDER GENERATION (Perfect Alignment guaranteed by native mapping)
        # =========================================================================
        ax.plot(time, aligned_raw_signal, label='Raw Zero-Centered Signal', color='gray', alpha=0.5)
        ax.plot(time, final_clean, label='Cleaned Signal', color='blue', linewidth=1.5)
        ax.axhline(d_thresh, color='red')

        if found:
            ax.fill_between(time, final_clean, 0, 
                            where=(time >= time[start]) & (time <= time[end]), 
                            color='purple', alpha=0.3, label='Area Region')

            ax.axvline(x=time[start], color='green', linestyle='--', label='Start Trigger')
            ax.axvline(x=time[end], color='red', linestyle='--', label='End Trigger')
            
            if local_min is not None and local_max is not None:
                ax.scatter(time[local_min], raw_signal[local_min] - baseline_offset, color='pink', zorder=5)
                ax.scatter(time[local_max], raw_signal[local_max] - baseline_offset, color='pink', zorder=5)

        ax.text(0.05, 0.95, arg_string, transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # =========================================================================
        # AREA MATH & EXPORT STRATEGY
        # =========================================================================
        if found:
            filepaths.append(file_path)
            event_area = calculate_drop_area_sensitive(raw_signal, time, start, end)
            
            if event_area < 1e-10:
                found = False
                ax.set_title(f"No Drop Found, Area Too Small: {os.path.basename(file_path)}")
                event_area = None
                dose = None
            else:
                dose = convert_dose(dose_file, dose_scale_file, args["beam"], args["Z"], args["pulse"], 2.0, True)
                ax.set_title(f"DETECTION SUCCESS: {os.path.basename(file_path)}")
        else:
            ax.set_title(f"No Drop Found: {os.path.basename(file_path)}")
            event_area = None
            dose = None

        areas.append(event_area)
        doses.append(dose)
            
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        if config.showPlot: 
            plt.show()
            
        clean_filename = os.path.basename(file_path).replace(".csv", "")
        fig.savefig(f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/waveforms/cleaned_figs/cleaned_{clean_filename}.png", dpi=150)
        plt.close(fig)
        
    return areas, doses, filepaths, manual_diff

def generator(log_file, output_file, date, sensor):
    pulses = [0.5, 1.0, 2.0, 3.0]
    log_df = pd.read_csv(log_file)
    Detector = sensor
    if not os.path.exists(output_file):
        print("does not exist")
        df = pd.DataFrame(columns=['Detector','Channel','Beam','Pulse','Dose','X','Z','File','ch1_area','ch2_area','ch1_peaks','ch2_peaks', 'ch1_osc_count', 'ch2_osc_count'])
        df.to_csv(output_file)
    areas_out = []
    doses_out = []
    pulse_out = []
    z_out = []
    beam_out = []
    HV_out = []
    filenames = []
    manual_pw = []
    categorized_pw = []
    # TEST swapping HV and Z cols
    log_df[['HV', 'Z']] = log_df[['Z', 'HV']].values
    log_df['Z'] = log_df['Z']*100 # converting to cm
    for pulse in pulses:
        matching_rows = log_df[
            (log_df["Detector"] == Detector) &
            (log_df["Pulse"] == str(pulse))
        ]
        if config.verbose>1:
            print(f"MATCHING ROWS for {Detector} and {pulse}")
            print(matching_rows)
            # print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
        for _, row in tqdm(matching_rows.iterrows(), desc="Processing log file rows..."):
            file_min = str(row["FileMin"])
            file_max = str(row["FileMax"])
            Z = float(row["Z"])
            HV = row.get("HV", "Unknown")
            beam = row["Beam"]
            args = {"Z":Z,"HV":HV,"beam":beam,"pulse":pulse}
            file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_min_num = int(file_min_num.replace(".csv",""))
            file_max_num = file_max.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_max_num = int(file_max_num.replace(".csv",""))
            # interate over all files
            areas, doses, filepaths, manual_diff = denoise_and_get_area(file_min_num,file_max_num,sensor,date,args)
            # if manual_diff is not None:
            areas_out += areas
            doses_out += doses
            manual_pw += [manual_diff]
            # if manual_diff is not None:
            #     categorized_pw += [find_nearest([0.5,1.0,2.0,3.0],manual_diff)]
            # else:
            #     categorized_pw += [None]
            count = len(areas)
            z_out += [Z] * count
            HV_out += [HV] * count
            filenames += filepaths * count
            beam_out += [beam] * count
            pulse_out += [pulse] * count
    data = {
    'Pulse': pulse_out,
    'Z': z_out,
    'HV': HV_out,
    'Beam': beam_out,
    'Dose': doses_out,
    'Area': areas_out,
    # 'Filename': filenames,
    'manual_pw': manual_pw,
    'categorized_pw': categorized_pw
    }

    # Add this for debugging
    for key, value in data.items():
        if isinstance(value, list):
            print(f"Column: {key} | Length: {len(value)}")
        else:
            print(f"Column: {key} | (Scalar Value)")

    output_df = pd.DataFrame(data)
    output_df.to_csv(output_file, index=False)
    return output_file

def remove_group_outliers(df, column='Area', factor=1.5):
    """
    Removes outliers from each group (Z) using the IQR method.
    factor=1.5 is standard; 1.0 is more aggressive, 3.0 is for extreme outliers only.
    """
    def filter_func(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

    # We group by the experimental parameters to find outliers within repeats
    print(f"outliers removed: {len(df) - len(df.groupby(['HV', 'pulsewidth', 'Z', 'Q_nC'], group_keys=False).apply(filter_func))}")
    return df.groupby(['HV', 'pulsewidth', 'Z'], group_keys=False).apply(filter_func)

def plot_dynamic_linear_fit_clean(csv_file, detector_name="SiC", r2_threshold=0.999, outlier_factor=1.5):
    
    # 1. Load and Prepare
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    # Initial Mask
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0) & (df['Beam'] != "Electron 85V")
    df_filtered = df[mask].copy()

    # 2. OUTLIER REMOVAL
    # This removes the "wild" points within each dose repeat
    before_count = len(df_filtered)
    df_clean = remove_group_outliers(df_filtered, column='Area', factor=outlier_factor)
    after_count = len(df_clean)
    
    print(f"Outlier Removal: Dropped {before_count - after_count} points ({((before_count-after_count)/before_count)*100:.1f}%)")

    # 3. Grouping for Plotting
    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        group = group[(group['DoseRate_sec'] < 145000) | (group['DoseRate_sec'] > 160000)]
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std', 'count']).reset_index().sort_values('DoseRate_sec')
        x_data = stats_df['DoseRate_sec'].values
        y_data = stats_df['mean'].values

        if len(x_data) < 3:
            continue

        # --- DYNAMIC LINEAR REGION DETECTION ---
        best_idx = 2
        for i in range(3, len(x_data) + 1):
            slope, intercept, r_val, _, _ = stats.linregress(x_data[:i], y_data[:i])
            if (r_val**2) >= r2_threshold:
                best_idx = i
            else:
                break

        slope, intercept, r_val, _, _ = stats.linregress(x_data[:best_idx], y_data[:best_idx])

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        
        # Plot individual points to show the "spread" after outlier removal
        plt.scatter(group['DoseRate_sec'], group['Area'], color='gray', alpha=0.2, label='Cleaned Pulses')
        
        # Plot Means with Error Bars
        plt.errorbar(x_data, y_data, yerr=stats_df['std'], fmt='o', color='black', 
                     capsize=4, label='Averaged Response')
        
        # Highlight Fit Region
        plt.plot(x_data[:best_idx], slope * x_data[:best_idx] + intercept, color='red', 
                 linewidth=2, label=f'Linear Fit (R²={r_val**2:.5f})')

        plt.title(f"Cleaned Dynamic Fit: {detector_name} | HV: {hv} | {beam} | Pulse: {pulse}", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)")
        plt.ylabel("Integrated Area (V·s)")
        plt.legend()
        plt.savefig(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/generated_plots/swapped_response_curve_plots/isolated_{pulse}_{beam}.png")

def plot_increasing_quadratic_fit(csv_file, detector_name="SiC"):
    df = pd.read_csv(csv_file)
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06
    
    mask = (df['Area'] > 1e-12) & (df['DoseRate_sec'] > 0)
    df_clean = df[mask].copy()

    groups = df_clean.groupby(['HV', 'Beam', 'Pulse'])

    for (hv, beam, pulse), group in groups:
        # Aggregate and sort by dose rate
        stats_df = group.groupby('DoseRate_sec')['Area'].agg(['mean', 'std']).reset_index().sort_values('DoseRate_sec')
        x_raw = stats_df['DoseRate_sec'].values
        y_raw = stats_df['mean'].values

        # --- FIND STRICTLY INCREASING REGION ---
        increasing_idx = 0
        current_max = -np.inf
        
        for i in range(len(y_raw)):
            if y_raw[i] > current_max:
                current_max = y_raw[i]
                increasing_idx = i + 1 # Include this point
            else:
                # The data has dipped or flattened; stop here
                break
        
        x_data = x_raw[:increasing_idx]
        y_data = y_raw[:increasing_idx]

        # Safety check: Need at least 3 points for a quadratic fit
        if len(x_data) < 3:
            print(f"Skipping {hv}V {pulse}us: Only {len(x_data)} increasing points.")
            continue

        # --- QUADRATIC FIT ---
        coeffs = np.polyfit(x_data, y_data, 2)
        model = np.poly1d(coeffs)
        
        y_pred = model(x_data)
        r2 = r2_score(y_data, y_pred)

        # --- PLOTTING ---
        fig = plt.figure(figsize=(10, 6))
        
        # Plot full data in light gray to show what was excluded
        plt.errorbar(x_raw, y_raw, yerr=stats_df['std'], fmt='o', color='gray', 
                     alpha=0.3, label='Excluded (Non-increasing)')
        
        # Plot data used for fit
        plt.errorbar(x_data, y_data, yerr=stats_df['std'][:increasing_idx], fmt='o', 
                     color='black', label='Increasing Data', zorder=4)
        
        # Plot the fit
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        plt.plot(x_fit, model(x_fit), color='crimson', lw=2.5, label=f'Quad Fit (R²={r2:.5f})')

        # Formatting
        plt.title(f"Strictly Increasing Fit: {detector_name}\nHV: {hv} | {beam} | {pulse}µs", fontsize=13)
        plt.xlabel("Instantaneous Dose Rate (Gy/s)")
        plt.ylabel("Integrated Area (V·s)")
        
        # Labeling the "Saturation" point
        plt.axvline(x_data[-1], color='orange', ls='--', alpha=0.6, label='Peak Response')

        # Equation Text
        a, b, c = coeffs
        fit_text = (f"y = {a:.2e}x² + {b:.2e}x + {c:.2e}\n"
                    f"R² = {r2:.5f}\n"
                    f"Max Linear Rate: {x_data[-1]:.2e} Gy/s")
        
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, 
                 va='top', family='monospace', bbox=dict(facecolor='white', alpha=0.8))

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/generated_plots/quadfit/quadfit_{beam}beam_HV{hv}_pulse{pulse}us.png")
        plt.show()
        plt.pause(0.1)
        plt.close(fig)

if __name__ == "__main__":
    # df = pd.read_csv("total_10_15_CH2_alternate_dose.csv")
    # df_new = remove_group_outliers(df,'Q_nC',1.0)
    # df_new.to_csv("total_10_15_CH2_outliers_removed.csv")
    # convert_dose(dose_file, dose_scale_file, "Electron 110V",0.4,2.0,2.0,True)
    '''log_file = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/backup-lgad-2025-11-20-log.csv"
    output_file = "out.csv"
    date = "2025-11-20"
    df = pd.read_csv("tmp.csv")
    print(df.head())
    selected_df = df[df["Z"] == 40.0]
    selected_df = selected_df[selected_df["pulsewidth"] > 0.5 ]
    print(selected_df[[
        'pulsewidth',
        'Z',
        'HV',
        'Q_nC'
    ]])
    new_doses = []
    for i, row in df.iterrows():
        new_doses.append(convert_dose(dose_file,dose_scale_file,row["Beam"],row["Z"]/100,row["pulsewidth"],"2",True))
    df["Dose"] = new_doses
    df.to_csv("tmp_dose_changed.csv")'''
    
    # generator(log_file, output_file,"2025-11-20","SiC")
    log_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/log_files/backup-lgad-2025-11-20-log.csv"
    output_file = "generator_out.csv"
    date = "2025-11-20"
    generator(log_file, output_file,"2025-11-20","SiC")
    plot_total_dose_vs_area("/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/out_csvs/swapped_dose_hv.csv","SiC", HV=40)
    plot_instantaneous_dose_vs_area("/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/out_csvs/swapped_dose_hv.csv","SiC", HV=40)
    '''df = pd.read_csv(log_file)
    print(df.columns)
    print(df["Z"].min())
    print(df["Z"].max())
    df = df[df["Z"] == float(10.0)]
    print("z filter")
    print(df.head())
    df = df[df["Beam"] == "Electron 110V"]
    print("beam filter")
    print(df.head())
    print(df["Pulse"])
    df = df[(df["Pulse"] == float(0.5)) | (df["Pulse"] == float(3.0))]
    df = df[df["HV"] == 40.0]
    print("pulse filter")
    print(df.head())
    selected_columns = df[["Beam", "Pulse", "Z", "FileMin", "FileMax"]]
    # print(selected_columns)
    # print("mean 110:")
    # print(selected_columns[selected_columns["Beam"] ==  "Electron 110V"]["Area"].mean())
    # print("mean 191:")
    # print(selected_columns[selected_columns["Beam"] ==  "Electron 191V"]["Area"].mean())
    for index, row in df.iterrows():
        print(row)
        print(f"DEBUG: Raw FileMin: {row['FileMin']} | Raw FileMax: {row['FileMax']}")
        file_min = row["FileMin"]
        file_max = row["FileMax"]
        file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
        file_min_num = int(file_min_num.replace(".csv",""))
        file_max_num = file_max.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
        file_max_num = int(file_max_num.replace(".csv",""))
        file_path = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/cleaned_figs/cleaned_scope-results-{date}-{str(file_min_num).zfill(4)}.csv.png"
        img = Image.open(f'{file_path}')
        img.show()
        print("end of batch")'''