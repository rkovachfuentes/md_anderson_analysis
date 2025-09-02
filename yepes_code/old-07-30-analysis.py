#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import re
from scipy.integrate import simpson  # Use `simpson` instead of `simps`
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

verbose=1
savePlot=True
showPlot=False

#==================================================================================================
#
#==================================================================================================
def read_csv(file_path_init, selected_channel="CH1"):
    # need to skip all header fields so it reads data correctly
    """Reads the oscilloscope CSV file and extracts time and channel data."""
    time = []
    ch1 = []
    ch2 = []
    num_columns = 0
    file_path_check = Path(str(file_path_init))
    if file_path_check.is_file():
        print("filepath found")
        file_path = file_path_init
        pass
    else:
        file_path = str(file_path_init)
        file_path = file_path.replace("/home/lgad/data","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code")
        print(file_path)
    try:
        with open(file_path, mode='r') as file:
            print("opened reader")
            reader = csv.reader(file)
            data_started = False
            for row in reader:
                if row[0]  == "TIME": # and row[0]
                    data_started = True
                    continue
                if data_started and row:
                    try:
                        num_columns = len(row)
                        if num_columns == 0:
                            break
                        if num_columns >= 2:
                            time.append(float(row[0]))
                            ch1.append(float(row[1]))
                            if num_columns == 3:
                                ch2.append(float(row[2]))  # Read third column if it exists
                    except ValueError:
                        print(f"Skipping invalid data row: {row}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return np.array([]), np.array([])
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array(time), np.array(ch1)

    if selected_channel=="CH1":
        return np.array(time), np.array(ch1)
    if selected_channel=="CH2":
        return np.array(time), np.array(ch2)
 
#==================================================================================================
#
#==================================================================================================
def get_area(file_path, pulse=2, Z=60, HV=0, beam="Electrons 85V", ifile=1560, selected_channel="CH1" ):
    time, signal = read_csv(file_path, selected_channel)
    # remove any signal less than 0
    if signal.size == 0:
        print("ZERO SIGNAL")
        return
    # Determine the number of points corresponding to 0.1 microseconds
    time_step = time[1] - time[0]  # Time difference between consecutive points
    lookback_points = int(0.3e-6 / time_step)  # Number of points in 0.1 microseconds
    w = savgol_filter(signal, 701, 2)
    # outlier_indices = np.where(signal > 0)
    # signal = signal.remove(outlier_indices[0])

    # Initialize variables to track the largest increase
    largest_increase = -np.inf  # Start with the smallest possible number
    start_index = 0
    end_index = 0

    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_range = signal_max-signal_min
    shifted_min  = signal_min+0.8*signal_range
    # outlier_top = (signal_range*0.95)+signal_min
    # outlier_bottom = (signal_range*0.05)+signal_min

    threshold = 0.30 * signal_range
    # Find indices where fluctuation exceeds the threshold
    fluctuation_indices = []
    currmin = signal_max
    currmin_i = 0
    for i in range(1, len(signal)):
        req_width = 10
        if abs(signal[i] - signal[i-req_width]) > threshold:
            fluctuation_indices.append(i)
            if (signal[i] < currmin):
                currmin = signal[i]
                currmin_i = len(fluctuation_indices)-1

    # Identify the start of fluctuations
    if len(fluctuation_indices) > 1:
        # if pulse <= 1:
        start_of_fluctuations = fluctuation_indices[0]
        if start_of_fluctuations <= 10:
            fluctuation_indices = fluctuation_indices[1]
        end_of_fluctuations = fluctuation_indices[-1]
        print(f"Signal starts fluctuating beyond 5% of max range at index {start_of_fluctuations}")
        print(f"Signal starts fluctuating beyond 5% of max range at index {end_of_fluctuations}")
        '''else:
            fluctuation_indices.sort(reverse=True)
            if fluctuation_indices[0] > fluctuation_indices[1]:
                end_of_fluctuations = fluctuation_indices[3]
                start_of_fluctuations = fluctuation_indices[2]
            else:
                end_of_fluctuations = fluctuation_indices[2]
                start_of_fluctuations = fluctuation_indices[3]'''
    else:
        print("No significant fluctuation found.")
        start_of_fluctuations = 0

    first_good_bin = start_of_fluctuations+int(5e-7/time_step)
    first_good_bin = 1
    last_good_bin = 1

    sigma = 10  # Standard deviation of the Gaussian kernel
    # Loop backwards through the array
    for i in range(len(signal) - 1, lookback_points - 1, -1):
        # Calculate the signal difference over the lookback window
        if signal[i] > shifted_min: continue
        if i < first_good_bin: continue
        signal_difference = signal[max(0,i - lookback_points)] - signal[i]
        #print("signal_difference ", signal_difference, " time ", time[i]," signal ", signal[i]) 
        #if time[i] < 0.0: continue
        # Update if the current difference is the largest increase
        if signal_difference > largest_increase:
            '''if (time[i] <= 0):
                continue
            else:'''
            largest_increase = signal_difference
            start_index = i - lookback_points
            end_index = i

    # Initialize variables to track the largest increase
    largest_increase = -np.inf  # Start with the smallest possible number
    second_largest_increase = -np.inf
    start_index_back = 0
    end_index_back = 0

    # moving BACKWARDS to find the end of the signal window
    for i in range(lookback_points - 1, len(signal) - 1, 1):
        # Calculate the signal difference over the lookback window
        if signal[i] > shifted_min: continue
        if i < last_good_bin: continue
        signal_difference = signal[max(0,i - lookback_points)] - signal[i]
        #print("signal_difference ", signal_difference, " time ", time[i]," signal ", signal[i]) 
        #if time[i] < 0.0: continue
        # Update if the current difference is the largest increase
        if signal_difference > largest_increase:
            # print("**************** largest increase yet ")
            if (time[i] <= 0):
                continue
            else:
                largest_increase_back = signal_difference
                start_index_back = i - lookback_points
                end_index_back = i
            # print(start_index_back)

    # Output the results
    if verbose>1:
        print(f"Largest increase: {largest_increase:.2f}")
        print(f"Start index: {start_index}, Time: {time[start_index]:.6e} s signal {signal[start_index]}")
        print(f"End index: {end_index}, Time: {time[end_index]:.6e} s signal {signal[end_index]}")


    # Shift the starting point by 0.1 microseconds
    shift_points = int(0.2e-6 / time_step)  # Number of points to shift earlier
    shifted_start_index = max(0, start_index)  # Ensure index is not negative  - shift_points
    if pulse >= 1:
        print("pulse over 1")
        shifted_start_index = fluctuation_indices[0]
        print("shifted start index")
        print(shifted_start_index)
        # shifted_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))
        shifted_end_index = straight_line_across(time, signal, shifted_start_index, signal_range)
    else:
        # shifted_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))
        shifted_end_index = straight_line_across(time, signal, shifted_start_index, signal_range)

    # Determine the region to remove
    remove_start_index = shifted_start_index
    remove_end_index = shifted_end_index

    # Remove the area by setting the signal to NaN in this region
    signal_removed = np.copy(signal)
    signal_removed[remove_start_index:remove_end_index + 1] = np.nan
    if math.isnan(signal_removed[0]):
        signal_removed[0] = signal[0]
    if math.isnan(signal_removed[-1]):
        signal_removed[-1] = signal[-1]
    #if pulse>2:
    #   signal_removed[remove_start_index:len(signal)] = np.nan

    # Interpolate the missing values
    # if there is no signal: select 2 signal points to remove
    valid_mask = ~np.isnan(signal_removed)  # Mask to keep valid values
    if np.all(valid_mask == False):
        valid_mask[-1] = True
        valid_mask[0] = True
    temp_interpolated_baseline = np.interp(time, time[valid_mask], signal_removed[valid_mask])

    baseline_smoothed = gaussian_filter(temp_interpolated_baseline, sigma=sigma)

    #  Interpolate the missing values after smoothing
    interpolated_baseline = np.interp(time, time[valid_mask], baseline_smoothed[valid_mask])

    corrected_signal = signal-interpolated_baseline

    # Calculate the area under the interpolated curve in the removed region
    removed_time = time[remove_start_index:remove_end_index + 1]  # Time in the removed region
    corrected_signal_removed_values = corrected_signal[remove_start_index:remove_end_index + 1]
    signal_removed_values = signal[remove_start_index:remove_end_index + 1]
    signal_area   = abs(simpson(y=corrected_signal_removed_values, x=removed_time))

    bins_around=30
    baseline_offset=0
    ch1=corrected_signal
    selected_ch1=corrected_signal_removed_values
    #dynamic_baseline = np.array([
    #         np.mean(selected_ch1[max(0, i - bins_around):min(len(selected_ch1), i + bins_around)]) - baseline_offset
    #         for i in selected_ch1 
    #     ])
    dynamic_baseline = np.array([
        np.mean(selected_ch1[max(0, idx - bins_around):min(len(selected_ch1), idx + bins_around)]) - baseline_offset
            for idx in range(len(selected_ch1))
    ])
    

    selected_ch1 = np.nan_to_num(selected_ch1, nan=0.0, posinf=0.0, neginf=0.0)
    dynamic_baseline = np.nan_to_num(dynamic_baseline, nan=0.0, posinf=0.0, neginf=0.0)
            # Find peaks below the dynamic baseline
    inverted_ch1 = -(selected_ch1 - dynamic_baseline)
    peaks, _ = find_peaks(inverted_ch1, height=0)
    peak_times = removed_time[peaks]
    peak_values = selected_ch1[peaks]
    nPeaks = len(peaks)
    npeaks_list.append(nPeaks)
    corresponding_files.append(file_path)
    pulse_widths.append(pulse)
    widths, width_heights, left_ips, right_ips = peak_widths(signal, peaks, rel_height=0.5)
    if widths.size != 0:
        max_width_idx = np.where(widths == max(widths))

    # Output the results
    if verbose>1:
        print(f"Shifted start index: {shifted_start_xsindex}, Time: {time[shifted_start_index]:.6e} s")
        print(f"Removed area start time: {time[remove_start_index]:.6e} s")
        print(f"Removed area end time: {time[remove_end_index]:.6e} s")
        print(f"Area under interpolated curve in signal region: {signal_area:.3e} ")

    # Visualization
    if savePlot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, signal, label="Original Signal", color="blue")
        # plt.plot(time, signal2, label="channel 2", color="orange")
        plt.plot(time, corrected_signal, label="Corrected Signal", color="green")
        plt.plot(time, interpolated_baseline, label="Interpolated Baseline", color="red")
        plt.plot(time, w, 'black', label="smoothed signal (savgol)")  # high frequency noise removed
        plt.axvspan(time[remove_start_index], time[remove_end_index], color="yellow", alpha=0.3, label="Removed Region")
        plt.axvline(time[shifted_start_index], color="green", linestyle="--", label="Shifted Start")
        plt.scatter(peak_times, peak_values, color='purple', marker='x', label="Peaks in Signal Zone")
        # if widths.size != 0:
            # plt.plot(peaks[max_width_idx[0]],selected_ch1[max_width_idx[0]], label="highest peak", color='black') 
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal Suppression and Interpolation")
        plt.text(
            0.6 * max(time),  # X-coordinate (adjust based on your plot range)
            min(signal)+0.05 * (max(signal)-min(signal)),  # Y-coordinate (adjust based on your plot range)
            f"Signal Area: {signal_area:.3e}",  # Text for the label
            fontsize=12,
            #color="purple",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple')  # Optional styling
        )
        
        plt.legend()
        plt.grid()

        # graphicDir=f"new-plot-dose-2025-07-30/Beam={beam}-Z={Z}-HV={HV}"
        graphicDir = f"new-plot-dose-2025-07-30/testing-area"
        if not os.path.exists(graphicDir):
            os.makedirs(graphicDir)
            print(f"Directory created: {graphicDir}")
        else:
            print(f"Directory already exists: {graphicDir}")
        
        graphicFile=f"{graphicDir}/dose-calc-pulse={pulse}-{ifile}-ch2.jpg"
        if verbose>2: print("graphicFile ", graphicFile)
        plt.savefig(graphicFile, format="jpeg", dpi=300)
        if showPlot:
            plt.show()
        plt.close()
        # 699 - pulse 0.5
    return signal_area, nPeaks

def cleanup_files(directory_name, range_dict):
    if not os.path.exists(directory_name):
        print("directory does not exist")
        return
    for entry in os.listdir(directory_name):
        full_path = os.path.join(directory_name, entry)
        if not (("-ch2" in full_path) or ("-ch1" in full_path)):
            os.remove(full_path)
            continue
        else:
            split_list = entry.split("-")
            print(split_list)
            pulse = split_list[2].replace("pulse=","")
            entry_val = split_list[3]
            if (int(entry_val) < range_dict[pulse][0]) or (int(entry_val) > range_dict[pulse][1]):
                os.remove(full_path)
    return

def straight_line_across(time_data, channel_data, drop_idx, signal_range):
    print("straight line across")
    concat_channel_data = channel_data[(drop_idx):]
    drop_value = channel_data[drop_idx]
    print("drop value", drop_value)
    print(concat_channel_data)
    indices = [i for i, x in enumerate(concat_channel_data) if x >= drop_value]
    prev = 0
    for idx in indices:
        if (idx - prev) > 100:
            break
    print("returning value",channel_data[drop_idx+idx])
    print("drop idx",drop_idx)
    print("returning idx", drop_idx+idx)
    return (drop_idx+idx)

def plot_dose_vs_area(log_file,group_col="Pulse"):
    log_df = pd.read_csv(log_file)
    grouped_data = log_df.groupby(group_col)
    color_list = ["red","black","blue","green","yellow","purple","brown"]
    pulse_list = ["0.1","0.5","1","2","3"]
    dose = log_df["Dose"]
    area = log_df["area"]
    peaks = log_df["peaks"]
    pulse = log_df["Pulse"]
    # Create a boolean mask where True indicates a NaN
    nan_mask_area = np.isnan(area)
    # Invert the mask to select non-NaN values
    non_nan_mask_area = ~nan_mask_area
    # Use boolean indexing to get the array without NaNs
    area_clean = np.array(area)[non_nan_mask_area]
    dose_clean = np.array(dose)[non_nan_mask_area]
    pulse_clean = np.array(pulse)[non_nan_mask_area]
    split_idx = []
    init_pulse = pulse_clean[0]
    for i in range(0,len(pulse_clean)):
        if (pulse_clean[i] != init_pulse):
            split_idx.append(i)
            init_pulse = pulse_clean[i]
    split_pulses = np.split(pulse_clean,split_idx,axis=0)
    split_doses = np.split(dose_clean,split_idx,axis=0)
    split_areas = np.split(area_clean,split_idx,axis=0)
    for i in range(0,len(split_pulses)-1):
        plt.figure(figsize=(10, 6))
        for i in range(0,len(pulse_list)-1):
            plt.scatter(split_doses[i], split_areas[i], label=split_pulses[i][0], color=color_list[i], alpha=0.6)
            # add in generating a linear fit for each pulse width plot 
            '''try:
                m, b = np.polyfit(split_doses[i], split_areas[i], 1)
                plt.plot(split_doses[i], m*split_doses[i] + b, color=color_list[i], label=f'Linear Fit: {split_pulses[i][0]:.2f}')
            except:
                pass'''
    plt.xlabel("Dose")
    plt.ylabel("Area")
    plt.title("Dose vs Area, Grouped By Pulse")
    plt.legend()
    plt.grid()
    plt.show()


        
#==============================================================================
def generate_baseline(log_file, selected_pulse=0.1, Z=60, HV=0, beam="Electrons 85V", ifile=1560, selected_channel="CH1"):
    log_df = pd.read_csv(log_file)
    signal1_array = []
    signal2_array = []
    signal1_lengths = []
    signal2_lengths = []
    for pulse in pulses:
        matching_rows = log_df[
            (log_df["Detector"] == Detector) &
            (log_df["Pulse"] == selected_pulse)
        ]
        if verbose>1:
            print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
            continue
        for _, row in matching_rows.iterrows():
            file_min = str(row["FileMin"])
            file_max = str(row["FileMax"])
            file_min_num = file_min.replace("/home/lgad/data/2025-07-30/scope-results-2025-07-30-", "")
            file_min_num = int(file_min_num.replace(".csv",""))
            file_max_num = file_max.replace("/home/lgad/data/2025-07-30/scope-results-2025-07-30-", "")
            file_max_num = int(file_max_num.replace(".csv",""))

            dose = row['Dose']
            signal_areas = []
            signal_peaks = []
            for i in range(file_min_num, file_max_num + 1):
                #if not i== 1422: continue
                file_path = re.sub(r'\d{4}(?=\.csv)', f"{i:04}", file_min)
                file_path = file_path.replace("/home/pyepes/data/","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/")
                
                #signal_area, signal_peak=get_area(file_path, pulse=pulse,Z=Z,X=X, ifile=i)
                try:
                    time, signal1 = read_csv(file_path, selected_channel="CH1")
                    _, signal2 = read_csv(file_path, selected_channel="CH2")
                    signal1_array.append(signal1[:6039])
                    signal1_lengths.append(len(signal1))
                    signal2_array.append(signal2[:6039])
                    signal2_lengths.append(len(signal2))
                except:
                    break
    ch1_min_length = np.min(signal1_lengths)
    ch2_min_length = np.min(signal2_lengths)
    ch1_max_length = np.max(signal1_lengths)
    ch2_max_length = np.max(signal2_lengths)
    print("max lengths")
    print(ch1_max_length, ch2_max_length)
    print("min lengths")
    print(ch1_min_length, ch2_min_length)
    print("shapes")
    print(np.array(signal1_array).shape)
    print(np.array(signal2_array).shape)
    ch1_baseline = np.average(signal1_array,axis=0)
    ch2_baseline = np.average(signal2_array,axis=0)
    print("shapes")
    print(ch1_baseline.shape)
    print(ch1_baseline.shape)

    plt.figure(figsize=(10, 6))
    plt.plot(time, ch1_baseline, label="ch1 baseline", color="purple")
    plt.plot(time, ch2_baseline, label="ch2 baseline", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Baselines from 0.1 Pulse Data")
    plt.legend()
    plt.grid()

    graphicDir=f"plot-dose-2025-07-30/Beam={beam}-Z={Z}-HV={HV}"
    if not os.path.exists(graphicDir):
        os.makedirs(graphicDir)
        print(f"Directory created: {graphicDir}")
    else:
        print(f"Directory already exists: {graphicDir}")
    
    graphicFile=f"{graphicDir}/baselines_from_tenth.jpg"
    if verbose>2: print("graphicFile ", graphicFile)
    plt.savefig(graphicFile, format="jpeg", dpi=300)  
    plt.show()
    plt.close()


    return signal_area, nPeaks


log_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/07-30-clipped.csv"
if not os.path.exists(log_file):
    print(f"log file {log_file} not found")
    exit()

log_df = pd.read_csv(log_file)
npeaks_list = []
corresponding_files = []
pulse_widths = []

Detector="BNL"
# X=0
# Z=135
# Z=35
# Z=-26
# Z=60
# Shield="No"
# HV=30
# LV="5.7"

verbose=0

pulses=[1]
pulses=[0.1,0.5,1,2,3]

mean_signal_areas=[]
mean_signal_peaks=[]
doses=[]
pulse_tracker = []

showPlot = True
var1, var2 = get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/2025-07-30/scope-results-2025-07-30-0849.csv",pulse=3,selected_channel="CH1")

'''
verbose=1
for pulse in pulses:
    matching_rows = log_df[
        (log_df["Detector"] == Detector) &
        # (log_df["Channel"] == Channel) &
        # (log_df["Beam intensity (V)"] == Beam) &
        # (log_df["Distance (cm)"] == distance) &
        # (log_df["Lateral displacement (cm)"] == Z) &
        # (log_df["Shielding or not"] == Shield) & 
        # (log_df["HV"] == HV) & 
        # (log_df["LV"] == LV) & 
     #  (log_df["Comment"] == Comment) & 
        (log_df["Pulse"] == pulse)
    ]
    if verbose>1:
       print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
       continue
    for _, row in matching_rows.iterrows():
        file_min = str(row["FileMin"])
        file_max = str(row["FileMax"])
        file_min_num = file_min.replace("/home/lgad/data/2025-07-30/scope-results-2025-07-30-", "")
        file_min_num = int(file_min_num.replace(".csv",""))
        file_max_num = file_max.replace("/home/lgad/data/2025-07-30/scope-results-2025-07-30-", "")
        file_max_num = int(file_max_num.replace(".csv",""))

        dose = row['Dose']
        signal_areas = []
        signal_peaks = []
        for i in range(file_min_num, file_max_num + 1):
            #if not i== 1422: continue
            file_path = re.sub(r'\d{4}(?=\.csv)', f"{i:04}", file_min)
            file_path = row["FileMin"].replace(f"{file_min_num:04}",f"{i:04}")
            file_path = file_path.replace("/home/lgad/data/","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/")
            if not os.path.exists(file_path):
               file_path = file_path.replace("CH1","CH2")
               if not os.path.exists(file_path):
                  print("{file_path} does not exist, skip.")
                  continue
            
            #signal_area, signal_peak=get_area(file_path, pulse=pulse,Z=Z,X=X, ifile=i)
            try:
                signal_area, signal_peak=get_area(file_path, pulse=pulse,ifile=i, selected_channel="CH2")
            except:
                break
            signal_areas.append(signal_area)
            signal_peaks.append(signal_peak)
            if verbose>-1:
               print(f"i {i} pulse {pulse} signal_area {signal_area}", )

        mean_signal_area = np.mean(signal_areas)
        mean_signal_peak = np.mean(signal_peaks)
        if verbose>1:
           print("signal_areas ", signal_areas)
           print(f"Pulse {pulse} Dose {dose} mean signal_area {mean_signal_area} ")
        mean_signal_areas.append(mean_signal_area)
        mean_signal_peaks.append(mean_signal_peak)
        doses.append(dose)
        pulse_tracker.append(pulse)
        matching_columns = ["Detector", "Beam", "Z", "X", "Pulse", "Dose"]
        # matching_rows[["Detector","Recorded Dose (C*10^-8)", "Pulse length (microseconds)","Distance (cm)","Lateral displacement (cm)"]])
    # Find rows that match on all specified columns
        row['area']=mean_signal_area
        row['peaks']=mean_signal_peak

        row_copy = row.copy()  # Prevents modification issues
        # print("row ", row_copy[matching_columns])

        df_row = pd.DataFrame([row_copy])
        
        mask = (log_df[matching_columns] == df_row[matching_columns].iloc[0]).all(axis=1)

        if mask.any():
        # Remove the matching rows before appending the new row
           #print("remove and add ")
           #print("df_row ", df_row)
           log_df = log_df[~mask]
           log_df = pd.concat([log_df, df_row], ignore_index=True)
           log_df.to_csv(log_file, index=False)
        else:
           #print("row not found")
           # row_copy.to_csv(log_file, index=False)
           pass

# extracting fit parameters for exponential fit
# Initial guess for parameters (a, b, c) - can be adjusted
initial_guess = [1.0, -0.5, 0.0]
# Perform the curve fit
# Create a NumPy array with NaN values
# Create a boolean mask where True indicates a NaN
nan_mask_doses = np.isnan(doses)
# Invert the mask to select non-NaN values
non_nan_mask_doses = ~nan_mask_doses
# Use boolean indexing to get the array without NaNs
mean_signal_areas_clean = np.array(mean_signal_areas)[non_nan_mask_doses]
doses_clean = np.array(doses)[non_nan_mask_doses]
nan_mask_mean_signal = np.isnan(mean_signal_areas_clean)
inf_mask_mean_signal = (np.isinf(mean_signal_areas_clean))
nan_mask_mean_signal = np.array([nan_mask_mean_signal[i] or inf_mask_mean_signal[i] for i in range(len(nan_mask_mean_signal))])
non_nan_mask_mean_signal = ~np.array(nan_mask_mean_signal)
mean_signal_areas_clean = np.array(mean_signal_areas_clean)[non_nan_mask_mean_signal]
doses_clean = np.array(doses_clean)[non_nan_mask_mean_signal]
pulse_tracker = np.array(pulse_tracker)[non_nan_mask_doses]
pulse_tracker = np.array(pulse_tracker)[non_nan_mask_mean_signal]

# Perform logarithmic transformation on x_data
log_x_data = np.log(doses_clean)

# Fit a linear regression to log_x_data and y_data
# The degree of the polynomial is 1 for linear regression
try:
    a, b = np.polyfit(log_x_data, mean_signal_areas_clean, 1)
except:
    a = 0
    b = 0

# Generate the fitted curve
try:
    x_fit = np.linspace(min(doses_clean), max(doses_clean), 100)
    y_fit = a * np.log(x_fit) + b
except:
    x_fit = np.linspace(0,10,100)
    y_fit = a * np.log(x_fit) + b
    pulse_tracker = [0]

# Plot the original data and the fitted curve


peaks_df = pd.DataFrame({'npeaks': npeaks_list, 'filenames': corresponding_files, 'pulse_widths': pulse_widths})
peaks_df.to_csv('peaks_df.csv', index=False)
plt.figure(figsize=(10, 6))
color_list = ["black","blue","green"]
init_pulse = pulse_tracker[0]
split_idx = []
for i in range(0,len(pulse_tracker)):
    if (pulse_tracker[i] != init_pulse):
        split_idx.append(i)
        init_pulse = pulse_tracker[i]
split_pulses = np.split(pulse_tracker,split_idx,axis=0)
split_doses = np.split(doses_clean,split_idx,axis=0)
split_mean_signal_areas = np.split(mean_signal_areas_clean,split_idx,axis=0)
for i in range(0,len(split_pulses)-1):
    plt.scatter(split_doses[i], split_mean_signal_areas[i], label=split_pulses[i][0], color=color_list[i], alpha=0.6)
    # add in generating a linear fit for each pulse width plot 
    try:
        m, b = np.polyfit(split_doses[i], split_mean_signal_areas[i], 1)
        plt.plot(split_doses[i], m*split_doses[i] + b, color=color_list[i], label=f'Linear Fit: {split_pulses[i][0]:.2f}')
    except:
        pass
plt.plot(x_fit, y_fit, color='red', label='overall logarithmic fit', alpha=0.6)
#plt.plot(doses, mean_signal_peaks, label="Dose vs Peaks", color="blue", alpha=0.6)

plt.xlabel("Dose (1e-8 C)")
plt.ylabel("Area")
plt.title(f"Dose vs Area")
plt.legend()
plt.grid()
plt.show()

range_dict = {"0.1":[699,np.inf],"0.5":[551, 1105],"1":[-np.inf,1121],"2":[-np.inf,1134],"3":[253,1103]}
cleanup_files("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-07-30/Beam=Electrons 85V-Z=60-HV=0",range_dict)

plot_dose_vs_area(log_file,group_col="Pulse")'''