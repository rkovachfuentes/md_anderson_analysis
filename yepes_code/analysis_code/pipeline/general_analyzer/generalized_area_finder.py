import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from tqdm import tqdm
import config
from scipy.signal import savgol_filter
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit
from PIL import Image
from scipy.signal import find_peaks
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_path = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/data/"
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
    if dist_m < 0:
        return np.nan
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
    except:
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
    
    if pulse_duration > 10e-6:  # 4.5 µs upper limit
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

# --- Execution ---
def denoise_and_get_area(min_file, max_file, sensor, date, args, outpath, polarity='positive', extrapolate_dose=False):
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
            
        file_path = f"{data_path}{date}/scope-results-{date}-{str(file_num).zfill(4)} (1).csv"
        check_path = Path(file_path)
        if not check_path.is_file():
            file_path = f"{data_path}/{date}/scope-results-{date}-{str(file_num).zfill(4)}.csv"
        df = pd.read_csv(file_path)
        raw_signal = df["CH1"].values
        time = df["TIME"].values
        
        # 1. Process Signal (Unpacks original 7 arguments exactly)
        final_clean, found, start, end, d_thresh = get_drop_event(time, raw_signal, polarity=polarity)

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

        # generates plot centered at 0
        ax.plot(time, aligned_raw_signal, label='Raw Zero-Centered Signal', color='gray', alpha=0.5)
        ax.axhline(d_thresh, color='red')
        
        # if area found, calculate
        if found:
            event_area = calculate_drop_area_sensitive(final_clean, time, start, end, polarity=polarity)
            
            if event_area < 0.5e-11:
                found = False
                ax.set_title(f"No Drop Found, Area Too Small: {os.path.basename(file_path)}")
                event_area = None
                dose = None
                final_clean = np.zeros_like(final_clean)
            else:
                if extrapolate_dose:
                    dose = convert_dose(dose_file, dose_scale_file, args["beam"], args["Z"], args["pulse"], 2.0, True)
                else:
                    dose = None
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

def generator(log_file, output_file, outpath, date, sensor, HV=None, polarity="positive", extrapolate_dose=False):
    pulses = [0.5, 1.0, 2.0, 3.0]

    # to be saved to outfile
    compiled_data = []

    log_df = pd.read_csv(log_file)
    Detector = sensor
    if not os.path.exists(output_file):
        print("does not exist")
        df = pd.DataFrame(columns=['Detector','Channel','Beam','Pulse','Dose','X','Z','File','ch1_area','ch2_area','ch1_peaks','ch2_peaks', 'ch1_osc_count', 'ch2_osc_count'])
        df.to_csv(output_file)

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
            if 'Collimator' in log_df.columns:
                collimator = row['Collimator']
            else:
                collimator = np.nan
            if 'V Scan' in log_df.columns:
                vscan = row['V Scan']
            else:
                vscan = np.nan
            if 'Resistance' in log_df.columns:
                resistance = row['Resistance'].split("ohm")[0]
                resistance = int(re.sub(r"\D", "", resistance))
                if resistance==1:
                    resistance = 1000
                if (resistance == 10) or (resistance == 1000):
                    print("*****************NON-100 RESISTANCE DETECTED!!!!!")
            else:
                resistance = np.nan
            args = {"Z":Z,"HV":HV,"beam":beam,"pulse":pulse}
            file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_min_num = int(re.sub(r"\D", "", file_min_num))
            print(file_min_num)
            # temporary patch since there are no ranges
            file_max_num = file_min_num
            # interate over all files
            areas, doses, filenames, manual_diffs, mins = denoise_and_get_area(file_min_num,file_max_num,sensor,date,args,outpath,polarity,extrapolate_dose)
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
                        "resistance": resistance,
                        "vscan": vscan
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
                        "resistance": resistance,
                        "vscan": vscan
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
    log_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/log_files/lgad-2026-09-22-log-mod.csv"
    output_file = "/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/pipeline/0922_generator_out.csv"
    date = "2026-09-22"
    outpath = f"/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/analysis_code/0922_waveforms/cleaned_figs"
    generator(log_file, output_file, outpath,"2026-09-22","SiC New Board",HV=100.0)
