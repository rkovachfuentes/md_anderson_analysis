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
from scipy.signal import ShortTimeFFT

verbose=1
showPlot = True
savePlot=True

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
                    print("data started!")
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
def get_area(file_path, range_dict, date, pulse=2, Z=60, HV=0, beam="Electrons 85V", ifile=1560, selected_channel="CH1" ):
    npeaks_list = []
    corresponding_files = []
    pulse_widths = []
    time, signal = read_csv(file_path, selected_channel)
    print(date)
    print(file_path)
    # remove any signal less than 0
    if signal.size == 0:
        print("ZERO SIGNAL")
        return
    # Determine the number of points corresponding to 0.1 microseconds
    time_step = time[1] - time[0]  # Time difference between consecutive points
    lookback_points = int(0.3e-6 / time_step)  # Number of points in 0.1 microseconds
    w = savgol_filter(signal, 200, 2)

    # Initialize variables to track the largest increase
    largest_increase = -np.inf  # Start with the smallest possible number
    start_index = 0
    end_index = 0

    # Find indices where fluctuation exceeds the threshold
    fluctuation_indices = []
    if pulse >= 1:
        apparent_signal = w
    else:
        apparent_signal = signal
    signal_min = np.min(apparent_signal)
    signal_max = np.max(apparent_signal)
    signal_range = signal_max-signal_min
    shifted_min  = signal_min+0.8*signal_range
    threshold = 0.30 * signal_range
    currmin = signal_max
    currmin_i = 0
    print("apparent signal",apparent_signal)
    print("threshold", threshold)
    for i in range(5, len(apparent_signal)):
        req_width = 10
        if abs(apparent_signal[i] - apparent_signal[i-5]) > threshold:
            fluctuation_indices.append(i)
            if (apparent_signal[i] < currmin):
                currmin = apparent_signal[i]
                currmin_i = len(fluctuation_indices)-1
    print("fluctuation indices",fluctuation_indices)
    # Identify the start of fluctuations
    if len(fluctuation_indices) > 1:
        # if pulse <= 1:
        fluctuation_indices.sort(reverse=False)
        start_of_fluctuations = fluctuation_indices[0]
        if start_of_fluctuations <= 10:
            fluctuation_indices = fluctuation_indices[1]
        end_of_fluctuations = fluctuation_indices[-1]
        print(f"Signal starts fluctuating beyond 5% of max range at index {start_of_fluctuations}")
        print(f"Signal starts fluctuating beyond 5% of max range at index {end_of_fluctuations}")
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
        # Update if the current difference is the largest increase
        if signal_difference > largest_increase:
            if (time[i] <= 0):
                continue
            else:
                largest_increase_back = signal_difference
                start_index_back = i - lookback_points
                end_index_back = i

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
        print("shifted start index") 
        print(shifted_start_index)
        # shifted_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))
        shifted_end_index = straight_line_across(time, signal, shifted_start_index, signal_range)
        if ((shifted_end_index - shifted_start_index) <= 100):
            too_narrow = True
            counter = 1
            while too_narrow:
                new_end = straight_line_across(time, signal, shifted_end_index, signal_range)
                shifted_start_index = shifted_end_index
                shifted_end_index = new_end
                counter += 1
                if (((shifted_end_index - shifted_start_index) >= 100) or (counter > 10)):
                    too_narrow = False
    else:
        # shifted_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))
        shifted_end_index = straight_line_across(time, signal, shifted_start_index, signal_range)

    # Determine the region to remove
    remove_start_index = shifted_start_index
    remove_end_index = shifted_end_index

    # count large oscillations in the signal region
    osc_count = 0
    savgol_peak_i, _ = find_peaks(w[remove_start_index:remove_end_index], height=None, distance=50, prominence=0.07)
    savgol_neg_peak_i, _ = find_peaks(-w[remove_start_index:remove_end_index], height=None, distance=50, prominence=0.07)
    nsavgol_peaks = len(savgol_peak_i) + len(savgol_neg_peak_i)
    if nsavgol_peaks > 1:
        for i in range(0,len(savgol_neg_peak_i)):
            try:
                if savgol_neg_peak_i[i] < savgol_peak_i[i]:
                    osc_count += 1
            except:
                print("oops")
                break
    print("~~~~~")
    print("nsavgol peaks")
    print(nsavgol_peaks)
    print(remove_end_index-remove_start_index)
    print("~~~~~")
    print(osc_count)
    region_min = signal[remove_start_index:remove_end_index + 1].min()
    region_max = signal[remove_start_index:remove_end_index + 1].max()
    region_range = region_max-region_min
    large_osc_threshold = 0.3*region_range
    large_oscs = 0
    for i in range(remove_start_index+10, remove_end_index+10):
        if abs(w[i] - signal[i]) > large_osc_threshold:
            large_oscs += 1
    osc_percentage = large_oscs/(remove_end_index-remove_start_index)
    print(f"OSC COUNT: {large_oscs}")
    print(f"LARGE OSC PERCENTAGE: {osc_percentage}")
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
    saturation_corrected = saturation_correction(remove_start_index+savgol_peak_i, remove_start_index+savgol_neg_peak_i, time, corrected_signal, w, exp_mode=True)
    print("SATURATION CORRECTED")
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
            for idx in range(len(selected_ch1))])
    

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
    print("lengths needed")
    print(len(time))
    print(len(saturation_corrected))
    print(f"dif: {len(time)-len(saturation_corrected)}")
    if savePlot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, signal, label="Original Signal", color="blue")
        plt.plot(time, corrected_signal, label="Corrected Signal", color="green")
        # plt.plot(time, interpolated_baseline, label="Interpolated Baseline", color="red")
        plt.plot(time, w, 'black', label="smoothed signal (savgol)")  # high frequency noise removed
        plt.plot(time, saturation_corrected, 'red', label="saturation corrected - simple connector")
        plt.axvspan(time[remove_start_index], time[remove_end_index], color="yellow", alpha=0.3, label="Removed Region")
        plt.axvline(time[shifted_start_index], color="green", linestyle="--", label="Shifted Start")
        plt.scatter(time[remove_start_index+savgol_peak_i], corrected_signal[remove_start_index+savgol_peak_i], color='orange', marker='x', label="oscillation peaks", zorder=2)
        plt.scatter(time[remove_start_index+savgol_neg_peak_i], corrected_signal[remove_start_index+savgol_neg_peak_i], color='orange', marker='x', zorder=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal Suppression and Interpolation")
        plt.text(
            0.6 * max(time),  # X-coordinate (adjust based on your plot range)
            min(signal)+0.05 * (max(signal)-min(signal)),  # Y-coordinate (adjust based on your plot range)
            f"Signal Area: {signal_area:.3e}\nPulse: {pulse}\nZ: {Z}\nHV: {HV}\nosc_count: {osc_count}\nfile: {file_path[-14:-1]}v\nchannel: {selected_channel}",
            fontsize=12,
            #color="purple",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple')  # Optional styling
        )
        
        plt.legend()
        plt.grid()

        graphicDir = f"new-plot-dose-{date}/all_files"
        print(graphicDir)
        if not os.path.exists(graphicDir):
            os.makedirs(graphicDir)
            print(f"Directory created: {graphicDir}")
        else:
            print(f"Directory already exists: {graphicDir}")
        graphicFile=f"{graphicDir}/dose-calc-pulse={pulse}-{ifile}-{selected_channel}.jpg"
        if verbose>2: print("graphicFile ", graphicFile)
        plt.savefig(graphicFile, format="jpeg", dpi=300)
        if showPlot:
            print("SHOW")
            plt.show()
        plt.close()

    return signal_area, nPeaks, osc_count

def saturation_correction(positive_peaks_i, negative_peaks_i, time, signal, w, exp_mode=False):
    corrected_signal = []
    interval_lengths = []
    inner_interval_lengths = []
    print("signal lengths")
    print(len(w))
    print(len(signal))
    print(len(time))
    print("peak indices")
    print(negative_peaks_i[0],negative_peaks_i[1])
    print(negative_peaks_i)
    interval_lengths.append(negative_peaks_i[0]+1)
    if exp_mode:
            x_points = time[negative_peaks_i]
            y_points = signal[negative_peaks_i]
            ylog_points = np.log(-y_points)
            print(f"log data points: {ylog_points}")
            coeffs = np.polyfit(x_points, ylog_points, 1)
            print("exp mode coeffs")
            print(coeffs)
            corrected_signal = -(np.exp(coeffs[1]) * np.exp(coeffs[0]*(time[negative_peaks_i[0]:negative_peaks_i[-1]])))
            corrected_signal = list(corrected_signal)
    else:
        for segment in range(0,len(negative_peaks_i)-1):
            print("SEGMENT")
            print(segment)
            print(time[negative_peaks_i[segment]])
            x_points = [time[negative_peaks_i[segment]], time[negative_peaks_i[segment+1]]]
            y_points = [signal[negative_peaks_i[segment]], signal[negative_peaks_i[segment+1]]]
            print("x points, y points")
            print(x_points)
            print(y_points)
            coeffs = np.polyfit(x_points, y_points, 1)
            p = np.poly1d(coeffs)
            polyfit_data = p(time[negative_peaks_i[segment]:negative_peaks_i[segment+1]])
            print(f"polyfit data shape: {polyfit_data.shape}")
            corrected_signal = corrected_signal + list(polyfit_data)
            print(f"interval length: {(negative_peaks_i[segment+1]-negative_peaks_i[segment])}")
            interval_lengths.append(negative_peaks_i[segment+1]-negative_peaks_i[segment])
            inner_interval_lengths.append(negative_peaks_i[segment+1]-negative_peaks_i[segment])
            print(f"polyfit data length: {len(polyfit_data)}")
            polyfit_data = []
    print(f"inner interval corrected signal length: {len(corrected_signal)}")
    print(f"pre-interval data length: {negative_peaks_i[0]}")
    print(f"post-interval data length: {len(signal)-negative_peaks_i[-1]}")
    interval_lengths.append(len(signal)-negative_peaks_i[-1])
    corrected_signal = list(signal[0:negative_peaks_i[0]+1]) + corrected_signal + list(signal[negative_peaks_i[-1]:-1])
    print(f"interval lengths list: {interval_lengths}")
    print(f"total length: {sum(interval_lengths)}")
    print(f"pre and post interval total length: {negative_peaks_i[0]+len(signal)-negative_peaks_i[-1]}")
    print(f"inner interval total length: {sum(inner_interval_lengths)}")
    return corrected_signal

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

def separate_comment(log_file):
    log_df = pd.read_csv(log_file)
    comments_new = []
    for comment in log_df["Comment"]:
        comment_clean = comment.replace(", chamber in", "")
        comment_clean = comment.replace("5 cm collimator, HV ","")
        split_comment = comment_clean.split(",")
        comments_new.append(split_comment[0].replace(" V",""))
    log_df["Comment"] = comments_new
    log_df.rename(columns={'Comment': 'HV (V)'})
    log_df.to_csv(log_file, index=False)
    return


def plot_dose_vs_area(log_file,group_col="Pulse"):
    log_df = pd.read_csv(log_file)
    log_df = log_df[log_df['Comment'] == 50]
    print(log_df.head())
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
    print(pulse)
    area_clean = np.array(area)
    dose_clean = np.array(dose)
    pulse_clean = np.array(pulse)
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
            plt.plot(split_doses[i], split_areas[i], label=split_pulses[i][0], color=color_list[i], linestyle='--')
            # add in generating a linear fit for each pulse width plot 
    plt.xlabel("Dose")
    plt.ylabel("Area")
    plt.title("Ch1 Mean Dose vs Area at 50 HV, All Distances, Grouped By Pulse")
    plt.legend()
    plt.grid()
    plt.show()

def plot_area_vs_distance(log_file,selected_voltage=50):
    log_df = pd.read_csv(log_file)
    log_df = log_df[log_df['Comment'] == selected_voltage]
    print(log_df.head())
    color_list = ["red","black","blue","green","yellow","purple","brown"]
    pulse_list = ["0.1","0.5","1","2","3"]
    dose = log_df["Z"]
    area = log_df["area"]
    peaks = log_df["peaks"]
    pulse = log_df["Pulse"]
    # Create a boolean mask where True indicates a NaN
    nan_mask_area = np.isnan(area)
    # Invert the mask to select non-NaN values
    non_nan_mask_area = ~nan_mask_area
    # Use boolean indexing to get the array without NaNs
    print(pulse)
    area_clean = np.array(area)
    dose_clean = np.array(dose)
    pulse_clean = np.array(pulse)
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
            plt.scatter(split_doses[i], split_areas[i], label=f"{split_pulses[i][0]} us pulse", color=color_list[i], alpha=0.6)
            try:
                m, b = np.polyfit(split_doses[i], split_areas[i], 1)
                plt.plot(split_doses[i], m*split_doses[i] + b, color=color_list[i])
            except:
                pass
    plt.xlabel("Distance (cm)")
    plt.ylabel("Dose")
    plt.title(f"Dose vs Distance at {str(selected_voltage)} HV, Grouped by Pulse")
    plt.legend()
    plt.grid()
    plt.show()

# def plot_dose_vs_distance(log_file,selected_voltage=50):

        
#==============================================================================

pulses=[1]
pulses=[0.1,0.5,1,2,3]
range_dict = {"0.1":[699,np.inf],"0.5":[551, 1105],"1":[-np.inf,1121],"2":[-np.inf,1134],"3":[253,1103]}

def generator(log_file, output_file, date):
    log_df = pd.read_csv(log_file)
    npeaks_list = []
    corresponding_files = []
    pulse_widths = []
    ch1_mean_signal_areas=[]
    ch2_mean_signal_areas=[]
    ch1_mean_signal_peaks=[]
    ch2_mean_signal_peaks=[]
    doses=[]
    pulse_tracker = []
    Detector = "BNL"

    showPlot = True
    if not os.path.exists(output_file):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df = pd.DataFrame(columns=['Detector','Channel','Beam','Pulse','Dose','X','Z','File','ch1_area','ch2_area','ch1_peaks','ch2_peaks', 'ch1_osc_count', 'ch2_osc_count'])
        df.to_csv(output_file)
    output_file_df = pd.read_csv(output_file)

    for pulse in pulses:
        matching_rows = log_df[
            (log_df["Detector"] == Detector) &
            (log_df["Pulse"] == pulse)
        ]
        if verbose>1:
            print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
        for _, row in matching_rows.iterrows():
            file_min = str(row["FileMin"])
            file_max = str(row["FileMax"])
            file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_min_num = int(file_min_num.replace(".csv",""))
            file_max_num = file_max.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_max_num = int(file_max_num.replace(".csv",""))

            dose = row['Dose']
            ch1_signal_areas = []
            ch1_signal_peaks = []
            ch2_signal_areas = []
            ch2_signal_peaks = []
            ch1_osc_count = []
            ch2_osc_count = []
            for i in range(file_min_num, file_max_num + 1):
                file_path = re.sub(r'\d{4}(?=\.csv)', f"{i:04}", file_min)
                file_path = row["FileMin"].replace(f"{file_min_num:04}",f"{i:04}")
                file_path = file_path.replace("/home/lgad/data/","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/")
                if not os.path.exists(file_path):
                    file_path = file_path.replace("CH1","CH2")
                    if not os.path.exists(file_path):
                        print("{file_path} does not exist, skip.")
                        continue
                # get area for both ch1 and ch2
                try:
                    print("CHANNEL 1")
                    ch1_signal_area, ch1_signal_peak, ch1_osc = get_area(file_path, range_dict, date, pulse=pulse,ifile=i, selected_channel="CH1")
                    print("outputs")
                    print(ch1_signal_area, ch1_signal_peak, ch1_osc)
                    ch2_signal_area, ch2_signal_peak, ch2_osc = get_area(file_path, range_dict, date, pulse=pulse,ifile=i, selected_channel="CH2")
                    print("oscs:")
                    print(ch1_osc)
                    print(ch2_osc)
                except:
                    break
                if ((ch1_signal_area > 0.0) and (ch2_signal_area) > 0.0):
                    print("appending new row")
                    new_row_data = {'Detector': row['Detector'], 'Channel': row['Channel'], 'Beam': row['Beam'], 'Pulse': row['Pulse'], 'Dose': row['Dose'], 'HV': row['Comment'], 'X': row['X'], 'Z': row['Z'], 'File': file_path, 'ch1_area': ch1_signal_area, 'ch2_area': ch2_signal_area, 'ch1_peaks': ch1_signal_peak, 'ch2_peaks': ch2_signal_peak, 'ch1_osc_count': ch1_osc, 'ch2_osc_count': ch2_osc}
                    new_row_df = pd.DataFrame([new_row_data])
                    # Concatenate the DataFrames
                    output_file_df = pd.concat([output_file_df, new_row_df], ignore_index=True)
                    if verbose>-1:
                        print(f"ch1 i {i} pulse {pulse} signal_area {ch1_signal_area}", )
                else:
                    print("no signal area")
                    continue
        print("pulse completed")
    print(output_file_df.head())
    output_file_df.to_csv(output_file,index=False)
    output_file_df = output_file_df.drop_duplicates()
    # output_file_df = output_file_df.dropna(subset=['ch1_osc_count', 'ch1_osc_count'])
    return(output_file)


# running the generator
'''verbose = 1
showPlot = False
savePlot = True
oct_dates = ["2025-10-14","2025-10-15"]
file_path = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0050.csv"
for date in oct_dates:
    generator(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-{date}-log.csv", f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/outfiles_for_plotting/output-file-gen-{date}.csv", date)
'''
showPlot = True
savePlot = True
# testing on a single file with a lot of oscillations
get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0339.csv", range_dict, "2025-10-15", pulse=3, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH2" )

# a single file with very few oscillations
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0897.csv", range_dict, "2025-10-15", pulse=1, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH1" )

# a weird one
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-1272.csv", range_dict, "2025-10-15", pulse=3, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH1" )