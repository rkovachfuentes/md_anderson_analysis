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
import old_response_curve_sic
from pathlib import Path
import re

import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


dose_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/dose_csvs/dose_scaling.csv"
dose_scale_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/dose_csvs/dose_scale_factors.csv"

def find_noise_threshold(signal, idx=500):
    first_noise = signal[:idx]
    sigma = np.std(first_noise)
    return max(10 * sigma, 0.001)

def dose_model(d, A, z0, b):
    # added safety checks to avoid dividing by zero / near infinite values
    # Ensure inputs are treated as numpy arrays or scalars cleanly
    d = np.asarray(d)
    
    # 1. Calculate the effective distance (denominator base)
    effective_distance = d + z0
    
    # 2. Define an ultra-small threshold to prevent dividing by zero 
    # or entering negative territory (which causes complex numbers if b is a float)
    epsilon = 1e-6
    
    # Handle scalar inputs safely
    if effective_distance.ndim == 0:
        if effective_distance < epsilon:
            # If the distance is physically inside the source or zero, 
            # return NaN or a high upper-bound limit.
            return np.nan 
    else:
        # Handle array inputs (if curve_fit passes an array of x_points)
        if np.any(effective_distance < epsilon):
            # Create a safe copy to avoid modifying original data
            safe_dist = np.where(effective_distance < epsilon, epsilon, effective_distance)
            result = A / (safe_dist ** b)
            # Mask the invalid indices to NaN so curve_fit knows they are bad points
            return np.where(effective_distance < epsilon, np.nan, result)

    # Standard safe calculation
    return A / (effective_distance ** b)

def convert_dose(dose_ref_csv, dose_scale_factors_csv, beam, dist_m, pulse_width, collimator_length_cm, dist_from_col=True):
    # check if dist_m is less than 0; if so, treat it as 0 m from beam pipe
    if dist_m < 0:
        dist_from_col = False
        dist_m = 0.0
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
        # Enforce that b must stay physically close to an inverse-square factor
        # Exponent lower bound = 1.0, z0 offset lower bound = 0.001
        lower_bounds = [0.0, 0.001, 1.0]
        upper_bounds = [np.inf, 1.0, 5.0]

        params, _ = curve_fit(dose_model, x_points, y_points, p0=initial_guess, 
                            bounds=(lower_bounds, upper_bounds))
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
        print(f"Exception: {e}")
        print("Returning NaN dose")
        return np.nan
    
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
                import pdb; pdb.set_trace()

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

import numpy as np

def return_before_first_spike(signal, buffer=0, polarity="positive"):
    if signal is None or signal.size < 24:
        return signal.copy() if signal is not None else np.array([]), 0, 0
    
    # 1. Zero-center using the median of the quiet baseline (first 500 samples)
    baseline_window = signal[:500] if len(signal) >= 500 else signal
    baseline_offset = np.median(baseline_window)
    zero_centered = signal - baseline_offset

    # 2. Estimate noise strictly from the quiet baseline (ignore all peaks)
    baseline_noise = np.std(zero_centered[:500])
    
    # Fallback in case the noise measurement is near zero
    if baseline_noise < 1e-4:
        baseline_noise = 0.0005  # 0.5 mV default noise floor

    # 3. Set threshold strictly as 4x the baseline noise floor
    # NO peak relative scaling, NO hardware floor offsets
    sensitive_limit = 4.0 * baseline_noise

    # Safety clamp: Ensure it never goes above 0.01 V (10 mV)
    # even if baseline window accidentally contains a small spike
    sensitive_limit = min(sensitive_limit, 0.010)

    return zero_centered.copy(), 0, sensitive_limit

def get_drop_event(time, signal, polarity="positive", min_consecutive_points=10, min_peak_amplitude=0.008):
    """
    Pulse event detector with transient spike filtering AND flat-line noise rejection.
    
    Parameters:
        min_consecutive_points: Ignores narrow noise spikes.
        min_peak_amplitude: Rejects flat traces where noise triggers consecutive 
                            samples but lacks a true signal peak (e.g. requires >= 8 mV).
    """
    final_clean, end_of_error, threshold = return_before_first_spike(
        signal, polarity=polarity
    )

    if final_clean is None or final_clean.size == 0:
        thresh_val = threshold if polarity == "positive" else -threshold
        return np.zeros_like(signal), False, None, None, thresh_val
    
    first_500 = signal[:500] if len(signal) >= 500 else signal
    baseline_offset = np.median(first_500)
    zero_centered = final_clean - baseline_offset
    
    search_zone = zero_centered[end_of_error:]

    if polarity == "positive":
        signal_threshold = threshold
        above_thresh = (search_zone > signal_threshold)
    else:
        signal_threshold = -threshold
        above_thresh = (search_zone < signal_threshold)
    
    # 1. Identify contiguous blocks of threshold crossings
    bounded = np.pad(above_thresh, (1, 1), mode='constant', constant_values=False)
    starts = np.where(bounded[1:] & ~bounded[:-1])[0]
    ends = np.where(~bounded[1:] & bounded[:-1])[0]
    
    valid_start_idx = None
    
    # 2. Loop through candidate blocks and verify BOTH duration and peak amplitude
    for s, e in zip(starts, ends):
        # Check duration
        if (e - s) >= min_consecutive_points:
            # Check peak height within/around candidate region to confirm real signal presence
            region_peak = np.max(search_zone[s:min(s + 500, len(search_zone))]) if polarity == "positive" else abs(np.min(search_zone[s:min(s + 500, len(search_zone))]))
            
            if region_peak >= min_peak_amplitude:
                valid_start_idx = s
                break
            
    # If no region passes duration + amplitude checks, treat trace as flat signal
    if valid_start_idx is None:
        return zero_centered, False, None, None, signal_threshold

    start_idx = end_of_error + valid_start_idx
    shifted_start = start_idx

    # 1. Look for recovery back below the threshold (or 0.5 * threshold) rather than strict <= 0
    if polarity == "positive":
        recovery = np.where(zero_centered[shifted_start:] <= (signal_threshold * 0.5))[0]
    else:
        recovery = np.where(zero_centered[shifted_start:] >= (-signal_threshold * 0.5))[0]

    # 2. If signal never recovers (flat DC line / clipping), reject it
    if len(recovery) == 0:
        return zero_centered, False, None, None, signal_threshold

    end_idx = shifted_start + recovery[0]

    # 3. Guard against unphysically long pulses (e.g. max pulse width of 4.5 microseconds)
    dt = time[1] - time[0] if len(time) > 1 else 1e-9
    pulse_duration = (end_idx - start_idx) * dt
    
    if pulse_duration > 4.5e-6:  # 4.5 µs upper limit
        return zero_centered, False, None, None, signal_threshold

    return zero_centered, True, shifted_start, end_idx, signal_threshold

def calculate_drop_area_sensitive(signal, time, start, end, polarity="positive"):
    """
    Integrates absolute area using consistent boundaries and zero-centered baseline.
    """
    v_segment = signal[start:end]
    t_segment = time[start:end]
    
    if len(v_segment) < 2:
        return 0.0

    # Optional: Apply Savitzky-Golay smoothing if array is large enough, 
    # but DO NOT shift local_zero on the pulse edge itself
    if len(v_segment) > 31:
        v_final = savgol_filter(v_segment, 31, 2)
    else:
        v_final = v_segment

    # Direct integration on zero-centered signal
    area = np.trapz(v_final, t_segment)
    return abs(area)

def plot_averaged_linearity(csv_file, detector_name="SiC", min_area_threshold=1e-11):
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

def plot_total_dose_vs_area(csv_file, detector_name="SiC", HV=40.0, min_area_threshold=1e-10, mode="total"):
    """
    Plots Total Dose per Pulse (Gy) vs Area, creating a separate 
    plot for each unique HV value found in the data.
    """
    # 1. Load the data
    print(f"Absolute file path being read: {os.path.abspath(csv_file)}")
    print(f"File size on disk: {os.path.getsize(csv_file)} bytes")
    
    with open(csv_file, 'r') as f:
        print("FIRST 3 LINES OF CSV FILE ON DISK:")
        for _ in range(3):
            print(f.readline().strip())

    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty.")
        return 0.0
    
    # ignore lines with no area found
    df = df[df['Beam'] != 'Noise/Rejected']
    
    # check for resorted pulses only
    # cast as true pulse column
    df = df.rename(columns={'Resorted_width_us': 'Pulse'})
    print(df['Pulse'])

    # Calculate DoseRate safely now that Pulse and Dose are clean numbers
    df['DoseRate_sec'] = (df['Dose'] / df['Pulse']) * 1e06

    # ==========================================
    # 🔍 FILTER BOUNDARY AUDIT PRINT STATEMENTS
    # ==========================================
    print(f"\n--- 🧪 POST-CLEANING DATA AUDIT ---")
    print(f"Total raw rows in DataFrame:  {len(df)}")
    print(f" -> Valid (non-NaN) Areas:    {df['Area'].notna().sum()}")
    print(f" -> Valid (non-NaN) Doses:    {df['Dose'].notna().sum()}")
    print(f" -> Valid (non-NaN) Pulses:   {df['Pulse'].notna().sum()}")
    print(f" -> Rows where Dose >= 0:     {(df['Dose'] >= 0).sum()}")
    print(f"-> Rows where Dose < 0.1: {(df['Dose'] <= 0.1).sum()}")
    print(f" -> Rows where Pulse >= 0.5:  {(df['Pulse'] >= 0.5).sum()}")
    
    # Apply inclusive filters
    mask = (
        df['Area'].notna() & 
        df['Dose'].notna() & 
        (df['Dose'] >= 0)  
    )
    df_clean = df[mask].copy()

    # 3. Apply Filters 
    # (Updated to filter based on Dose instead of DoseRate_sec)
    mask = (
        df['Area'].notna() & 
        df['Dose'].notna() & 
        (df['Dose'] > 0)
    )
    df_clean = df[mask].copy()

    if df_clean.empty:
        print("No valid data points left after filtering.")
        return 0.0
    
    # compare original df to cleaned df in desired columns
    print(df[['Area','Dose','Pulse','HV']])
    print(df_clean[['Area','Dose','Pulse','HV']]),

    # 4. Loop through unique HV values
    unique_hvs = df['HV'].unique()
    for hv_val in unique_hvs:
        hv_df = df_clean[df_clean['HV'] == hv_val].copy()
        # change minima to positive values (amplitudes)
        hv_df['max_V'] = hv_df['max_V'].abs()
        # Sort by Dose now for a clean line plot
        hv_df = hv_df.sort_values(by=['Beam', 'Pulse','Z'])
        # 5. Create the Plot
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        # Draw the lineplot first so matplotlib generates the line colors
        if mode=="total":
            ax = sns.lineplot(
                data=hv_df, 
                x='Dose', 
                y='max_V', 
                hue='Pulse', 
                style='Beam', 
                palette="viridis"
            )
        elif mode=="instantaneous":
            hv_df['DoseRate_sec'] = (hv_df['Dose'] / hv_df['Pulse']) * 1e06
            ax = sns.lineplot(
                data=hv_df, 
                x='DoseRate_sec', 
                y='max_V', 
                hue='Pulse', 
                style='Beam', 
                palette="viridis"
            )
        else:
            print("Invalid mode")
            return

        # ---------------------------------------------------------------------
        # 🎨 EXTRACT SEABORN'S COLOR PALETTE FROM THE LEGEND MAPPING
        # ---------------------------------------------------------------------
        # We parse the legend handles to see exactly what colors were mapped to what labels
        color_mapping = {}
        handles, labels = ax.get_legend_handles_labels()
        
        # Seaborn groups legend items by title, so we track our current active properties
        current_pulse = None
        for handle, label in zip(handles, labels):
            # Check if this legend item is a line and has a valid color
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
                # If the label looks like a number, it's a Pulse Width value
                try:
                    current_pulse = float(label)
                    color_mapping[current_pulse] = color
                except ValueError:
                    # Not a pulse width string (e.g. it's a Beam style name string), skip it
                    continue

        # ---------------------------------------------------------------------
        # 🏷️ LABEL MULTI-LINE COMBINATIONS WITH MATCHING COLORS
        # ---------------------------------------------------------------------
        labeled_combinations = set()

        for index, row in hv_df.iterrows():
            current_beam = row["Beam"]
            current_pulse = row["Pulse"]
            current_z = row["Z"]
            
            if mode=='total':
                current_dose = row["Dose"]
            else:
                current_dose = row["DoseRate_sec"]
            current_area = row["Area"]

            tracking_key = (current_beam, current_pulse, current_z)

            if tracking_key not in labeled_combinations:
                # Fallback to black if a specific color lookup edge case misses
                line_color = color_mapping.get(float(current_pulse), 'black')
                
                # Dynamic scaling offsets (1% right, 5% up)
                x_offset = current_dose * 1.01 if current_dose > 0 else 0.001
                y_offset = current_area * 1.05 if current_area > 0 else 1e-11

                '''plt.text(
                    x=x_offset,  
                    y=y_offset,  
                    s=f"{current_z} m", 
                    fontdict=dict(
                        size=8, 
                        color=line_color,     # <-- Applied matched color!
                        weight='bold'         # Bolded to make colored text readable
                    ),
                    va='bottom',          
                    ha='left',
                    # Box with subtle padding to keep the text crisp over crossing lines
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none')
                )'''
                
                labeled_combinations.add(tracking_key)
        
        plt.title(f"Dose vs Amplitude ({detector_name}) | HV: {hv_val}", fontsize=14)
        if mode=='total':
            plt.xlabel("Total Dose per Pulse (Gy)", fontsize=12) # Updated Label
            plt.ylabel("Amplitude (V)", fontsize=12)
        else:
            plt.xlabel("Dose per Second (Gy/s)", fontsize=12)
            plt.ylabel("Amplitude (V)", fontsize=12)
        plt.legend(title="Pulse Width & Beam", bbox_to_anchor=(1.05, 1), loc='upper left')

        # plt.xlim(0,5.0)

        plt.tight_layout()
        plt.show()

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
    # df = df.drop(columns=['Resorted_Width_us'])
    # df = df.dropna()

    df = df[df['Beam'] != 'Noise/Rejected']
    
    # check for resorted pulses only
    # cast as true pulse column
    df = df.rename(columns={'Resorted_width_us': 'Pulse'})

    # unravel the columns which are stored as arrays
    # Extract the 0th element from every numpy array directly
    # 1. Force the column to be treated as a string text field
    df['Area'] = df['Area'].astype(str)

    # 2. Extract only the numerical value (including decimals, minus signs, and scientific notation 'e')
    # This converts '[np.float64(2.494e-09)]' directly into '2.494e-09'
    df['Area'] = df['Area'].str.extract(r'([0-9.eE+-]+)')

    # 3. Convert the cleaned string column into a proper numeric float64 column
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
        
    # 2. Data Preparation
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')
    df['Pulse_Label'] = df['Pulse'].apply(lambda x: f"{x} µs" if pd.notnull(x) else "Unknown")

        # 1. Force both columns to strings so the regex engine can read them safely
    df['Dose'] = df['Dose'].astype(str)
    df['Pulse'] = df['Pulse'].astype(str)

    # 2. Extract only the numerical components (handling decimals, signs, and scientific notation)
    # This instantly converts things like '[np.float64(0.5)]' or '0.5' into clean numeric string text
    df['Dose'] = df['Dose'].str.extract(r'([0-9.eE+-]+)')
    df['Pulse'] = df['Pulse'].str.extract(r'([0-9.eE+-]+)')

    # 3. Safe-cast both columns to actual float64 numerical data types
    # (errors='coerce' turns empty cells or unparseable text into safe NaNs)
    print("\n")
    print("PULSE VALUES")
    print(df['Pulse'])
    print("PULSE VALUES IN US")
    print((df['Pulse'] * 1e-06))
    print("DF DOSE RAW VALUES")
    print(df['Dose'])
    df['Dose'] = pd.to_numeric(df['Dose'], errors='coerce')
    df['Pulse'] = pd.to_numeric(df['Pulse'], errors='coerce')

    # 4. Now this calculation is mathematically safe and will execute flawlessly
    df['DoseRate_sec'] = (df['Dose'] / (df['Pulse']*1e-06))

    # Put this right before creating your mask to see why rows are dying
    print(f"\n--- 🔍 MASK FILTER BREAKDOWN FOR THE {len(df)} POINTS ---")
    print(f"1. Has valid Area (not NaN): {sum(df['Area'].notna())}")
    print(f"2. Has valid Dose Rate (not NaN): {sum(df['DoseRate_sec'].notna()) if 'DoseRate_sec' in df else 'Column Missing'}")
    print(f"3. Dose Rate is strictly > 0: {sum(df['DoseRate_sec'] > 0) if 'DoseRate_sec' in df else 0}")
    print(f"4. Pulse is strictly > 0.5: {sum(df['Pulse'] > 0.5) if 'Pulse' in df else 0}")
    print(f"5. Pulse is >= 0.5 (Inclusive): {sum(df['Pulse'] >= 0.5) if 'Pulse' in df else 0}")
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
        plt.ylabel("Amplitude (V)", fontsize=12)
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

# --- Execution ---
def denoise_and_get_area(min_file, max_file, sensor, date, args, outpath):
    file_range = range(min_file, max_file + 1)
    areas = []
    doses = []
    filepaths = []
    manual_diffs = []
    mins = []
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    for file_num in file_range:
        arg_string = ""
        for key, value in args.items():
            arg_string += f"{key}: {value}\n"
            
        file_path = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{str(file_num).zfill(4)} (1).csv"
        check_path = Path(file_path)
        if not check_path.is_file():
            file_path = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{str(file_num).zfill(4)}.csv"
        df = pd.read_csv(file_path)
        raw_signal = df["CH1"].values
        time = df["TIME"].values
        
        # 1. Process Signal (Unpacks original 7 arguments exactly)
        final_clean, found, start, end, d_thresh = get_drop_event(time, raw_signal, polarity='positive')

        # 2. Setup Plot Canvas Context
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Calculate local raw baseline for zero-centering on the plot
        first_500 = raw_signal[:500] if len(raw_signal) >= 500 else raw_signal
        baseline_offset = np.median(first_500)
        aligned_raw_signal = raw_signal - baseline_offset

        if found:
            fallback = False

            signal_only = final_clean[start:end]

            start = min(int(start), len(time) - 1)
            end = min(int(end), len(time) - 1)

            # extracts the manual mins and maxes
            manual_min = start + np.argmin(signal_only)
            manual_max = np.argmax(final_clean)
            max_Val = np.max(signal_only)

            manual_diff = time[manual_max] - time[manual_min]

            if manual_diff < 0.1e-06:
                manual_diff = time[end] - time[start]
                fallback = True
                arg_string += "WARNING: error calculating time, using signal width fallback\n"
            
            arg_string += f"actual pulse length: {manual_diff/1e-6:.2f} us\n"
            arg_string += f"resorted pulse length: {find_nearest([0.5, 1.0, 2.0, 3.0], manual_diff/1e-06)}\n"
        else:
            manual_diff = None
            max_Val = 0

        # =========================================================================
        # PLOT RENDER GENERATION (Perfect Alignment guaranteed by native mapping)
        # =========================================================================
        ax.plot(time, aligned_raw_signal, label='Raw Zero-Centered Signal', color='gray', alpha=0.5)
        ax.axhline(d_thresh, color='red')
        
        # =========================================================================
        # AREA MATH & EXPORT STRATEGY
        # =========================================================================
        if found:
            event_area = calculate_drop_area_sensitive(final_clean, time, start, end, polarity='positive')
            
            if event_area < 0.5e-10:
                found = False
                ax.set_title(f"No Drop Found, Area Too Small: {os.path.basename(file_path)}")
                event_area = None
                dose = None
                final_clean = np.zeros_like(final_clean)
            else:
                dose = old_response_curve_sic.convert_dose(dose_file, dose_scale_file, args["beam"], args["Z"], args["pulse"], 2.0, True)
                ax.set_title(f"DETECTION SUCCESS: {os.path.basename(file_path)}")
        else:
            ax.set_title(f"No Drop Found: {os.path.basename(file_path)}")
            event_area = None
            dose = None
            final_clean[0:-1] = 0

        ax.plot(time, final_clean, label='Cleaned Signal', color='blue', linewidth=1.5)

        if found:
            ax.fill_between(time, final_clean, 0, 
                            where=(time >= time[start]) & (time <= time[end]), 
                            color='purple', alpha=0.3, label='Area Region')

            ax.axvline(x=time[start], color='green', linestyle='--', label='Start Trigger')
            ax.axvline(x=time[end], color='red', linestyle='--', label='End Trigger')
            
            if manual_min is not None and manual_max is not None:
                ax.scatter(time[manual_min], raw_signal[manual_min] - baseline_offset, color='pink', zorder=5)
                ax.scatter(time[manual_max], raw_signal[manual_max] - baseline_offset, color='pink', zorder=5)

        ax.text(0.05, 0.95, arg_string, transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        areas.append(event_area)
        doses.append(dose)
        filepaths.append(file_path)
        manual_diffs.append(manual_diff)
        mins.append(max_Val)
            
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        if config.showPlot: 
            plt.show()
            
        clean_filename = os.path.basename(file_path).replace(".csv", "")
        fig.savefig(f"{outpath}/cleaned_{clean_filename}.png", dpi=150)
        plt.close(fig)
        
    return areas, doses, filepaths, manual_diffs, mins

def generator(log_file, output_file, outpath, date, sensor, HV=None):
    pulses = [0.5, 1.0, 2.0, 3.0]

    # to be saved to outfile
    compiled_data = []

    log_df = pd.read_csv(log_file)
    Detector = sensor
    if not os.path.exists(output_file):
        print("does not exist")
        df = pd.DataFrame(columns=['Detector','Channel','Beam','Pulse','Dose','X','Z','File','ch1_area','ch2_area','ch1_peaks','ch2_peaks', 'ch1_osc_count', 'ch2_osc_count'])
        df.to_csv(output_file)

    # TEST swapping HV and Z cols
    # log_df[['HV', 'Z']] = log_df[['Z', 'HV']].values

    for pulse in pulses:
        if HV is not None:
            matching_rows = log_df[
                (log_df["Detector"] == Detector) &
                (log_df["Pulse"].astype(str) == str(pulse)) &
                (log_df["HV"] == HV)
            ]
        else:
            matching_rows = log_df[
                (log_df["Detector"] == Detector) &
                (log_df["Pulse"].astype(str) == str(pulse))
            ]
        if config.verbose>1:
            print(f"MATCHING ROWS for {Detector} and {pulse}")
            print(matching_rows)
            # print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
        for _, row in tqdm(matching_rows.iterrows(),desc="Processing log file rows..."):
            file_min = str(row["FileMin"])
            file_max = str(row["FileMax"])
            Z = float(row["Z"])
            # Z is in m
            HV = row.get("HV", "Unknown")
            beam = row["Beam"]
            collimator = row['Collimator']
            resistance = row['Resistance'].split("ohm")[0]
            resistance = resistance.strip()
            if "1k" in resistance:
                resistance = "1000"
            args = {"Z":Z,"HV":HV,"beam":beam,"pulse":pulse}
            file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_min_num = int(re.sub(r"\D", "", file_min_num))
            print(file_min_num)
            # temporary patch since there are no ranges
            file_max_num = file_min_num
            # interate over all files
            areas, doses, filenames, manual_diffs, mins = denoise_and_get_area(file_min_num,file_max_num,sensor,date,args,outpath)
            # Append everything to our dataset rows
            for area, dose, filename, manual_diff, max_Val in zip(areas, doses, filenames, manual_diffs, mins):
                if area is not None:
                    compiled_data.append({
                        "Filename": filename,              # <--- ADDED FILENAME
                        "HV": HV,
                        "beam": int(re.sub(r"\D", "", beam)),
                        "Z": Z,
                        "pulse": pulse,
                        "dose": dose,
                        "area": area,
                        "Calculated_Width_us": manual_diff/1e-06, # <--- ADDED TIME WIDTH
                        "Resorted_width_us": find_nearest([0.5, 1.0, 2.0, 3.0], manual_diff/1e-06),
                        "max_V": max_Val,
                        "collimator": collimator,
                        "resistance": resistance
                    })
                else:
                    # label rejected files as noise
                    compiled_data.append({
                        "Filename": filename,
                        "HV": np.nan,
                        "beam": "Noise/Rejected",
                        "Z": Z,
                        "pulse": np.nan,
                        "dose": np.nan,
                        "area": 0.0,
                        "Calculated_Width_us": 0.0,
                        "Resorted_Width_us": 0.0,
                        "max_V": max_Val,
                        "collimator": collimator,
                        "resistance": resistance
                    })

    output_df = pd.DataFrame(compiled_data)
    print(output_df.head())
    # output_csv_path = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/compiled_results.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Successfully saved master database to: {output_file}")
    
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
        plt.xlim(0,2.0)
        
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

def inspect_point(area_csv, beam_string, pulse_us, z, out_dir):
    # Create a Path object
    directory = Path(out_dir)

    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(area_csv)
    df = df.rename(columns={'Resorted_width_us': 'Pulse'})
    df = df[df['Beam'] == beam_string]
    df = df[df['Pulse'] == pulse_us]
    df = df[df['Z'] == z]
    if df.empty:
        print("No files found matching input criteria. Check datatypes")
        return
    else:
        for _, row in df.iterrows():
            file_min_num = row['Filename'].replace(f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/data/2025-11-20/scope-results-2025-11-20-", "")
            file_min_num = int(file_min_num.replace(".csv",""))
            file_max_num = file_min_num
            print(f" file num: {file_min_num}")
            args = {"Z":row['Z'],"HV":row['HV'],"beam":row['Beam'],"pulse":row['Pulse']}
            areas, doses, filenames, manual_diffs, mins = denoise_and_get_area(file_min_num,file_max_num,"SiC",date,args,out_dir)
    return

if __name__ == "__main__":
    # generator(log_file, output_file,"2025-11-20","SiC")
    log_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/log_files/lgad-2026-09-22-log-mod.csv"
    output_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/pipeline/0922_generator_out.csv"
    output_zero_bias = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/pipeline/0922_generator_out_zero_bias.csv"
    date = "2026-09-22"
    outpath = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/0922_waveforms/cleaned_figs"
    generator(log_file, output_file, outpath,"2026-09-22","SiC New Board",HV=100.0)
    generator(log_file, output_zero_bias, outpath, "2026-09-22","SiC New Board", HV=0.0)
    # plot_total_dose_vs_area("generator_out.csv", HV=40,mode='total')
    # plot_total_dose_vs_area("generator_out.csv", HV=40,mode='instantaneous')
    # plot_instantaneous_dose_vs_area("/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/pipeline/generator_out.csv","SiC", HV=40)
    beam_string = "Electron 191V"
    pulse_us = 0.5
    Z = 0 # Z in distance from collimator NOT beam exit
    out_dir = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/pipeline/isolated_point_{beam_string}_{pulse_us}_{Z}Z"
    # inspect_point("generator_out.csv",beam_string,pulse_us,Z,out_dir)
