import oct_file_generator
import config
import warnings
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
import plot_with_linear_reg_by_pulse

showPlot = True
savePlot = True

def slope(x1, y1, x2, y2):
  s = (y2-y1)/(x2-x1)
  return s

def check_if_flat(time, signal):
    poly = plot_with_linear_reg_by_pulse.generate_polynomial_fit(time, signal, degree=1)
    print(poly)
    coefficients = np.array(poly)
    slope = coefficients[0]
    if abs(slope) < 1e2:
        return True
    else:
        return False

def slope_list(time, signal, current_point_idx, length):
    new_slope_list = []
    for i in range(0,length-1):
        idx = current_point_idx + i
        new_slope = slope(time[idx],signal[idx],time[idx+1],signal[idx+1])
        new_slope_list.append(new_slope)
    return new_slope_list

def backtrack_to_local_max(time, signal, current_point_idx):
    print(current_point_idx)
    step = 1
    current_point = signal[current_point_idx]
    prev_point = signal[current_point_idx-step]
    prev_prev_point = signal[current_point_idx-(2*step)]
    current_slope = slope(time[current_point_idx], signal[current_point_idx], time[current_point_idx+step], signal[current_point_idx+step])
    prev_slope = slope(time[current_point_idx-step], signal[current_point_idx-step], time[current_point_idx], signal[current_point_idx])
    prev_prev_slope = slope(time[current_point_idx-(2*step)], signal[current_point_idx-(2*step)], time[current_point_idx], signal[current_point_idx])
    print(f"prev slope: {prev_slope}")
    try:
        slopes = slope_list(time, signal, current_point_idx, 100)
        prev_slopes = slope_list(time, signal, current_point_idx-100, 100)
        if (current_point >= np.max(signal[current_point_idx:current_point_idx+100])) and (current_point > np.min(signal[current_point_idx-100:current_point_idx])):
            print(check_if_flat(time[current_point_idx-500:current_point_idx], signal[current_point_idx-500:current_point_idx]))
            if config.verbose > 1: print(f"current point return: {current_point_idx}")
            return current_point_idx
        else:
            if config.verbose > 1: print("backtrack again")
            return backtrack_to_local_max(time, signal, current_point_idx-step)
    except Exception as e:
        print(f"Exception: {e}")
        return 0
            

def get_area(file_path, range_dict, date, pulse=2, Z=60, HV=0, beam="Electrons 85V", ifile=1560, selected_channel="CH1", show_saturation_correction=False):
    npeaks_list = []
    corresponding_files = []
    pulse_widths = []
    time, signal = oct_file_generator.read_csv(file_path, selected_channel)
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
    apparent_signal = w
    signal_mean = np.mean(w)
    signal_min = np.min(apparent_signal)
    signal_min_idx = np.where(apparent_signal == signal_min)[0]
    print(signal_min_idx)
    signal_max = np.max(apparent_signal)
    signal_range = signal_max-signal_min
    shifted_min  = signal_min+0.8*signal_range
    print("signal mean", signal_mean)
    print("signal range", signal_range)
    threshold = -(0.4 * signal_range) + signal_mean
    save_this_threshold = threshold
    save_this_mean = signal_mean
    currmin = signal_max
    currmin_i = 0
    if config.verbose>1: print("threshold", threshold)
    if config.verbose>1: print(f"time\n {time[0:10]}")
    if config.verbose>1: print(f"signal\n {signal[0:10]}")
    for i in range(5, len(apparent_signal)):
        req_width = 10
        if (apparent_signal[i]) < (threshold): #  - apparent_signal[i-5]
            fluctuation_indices.append(i)
            if (apparent_signal[i] < currmin):
                currmin = apparent_signal[i]
                currmin_i = len(fluctuation_indices)-1
    if config.verbose>1: print(f"fluctuation indices: {fluctuation_indices[0:10]}")
    # Identify the start of fluctuations
    found = False
    i = 0
    if len(fluctuation_indices) > 1:
        fluctuation_indices.sort(reverse=False)
        fluctuation_indices = np.array(fluctuation_indices)
        print(fluctuation_indices)
        condition = fluctuation_indices >= signal_min_idx
        valid_fluctuations = np.argmax(condition)
        first_start_of_fluctuations = fluctuation_indices[valid_fluctuations]
        start_of_fluctuations = backtrack_to_local_max(time, apparent_signal, first_start_of_fluctuations)
        if config.verbose>1: print(f"start of fluctuations: {start_of_fluctuations}")
        # max_interval, max_width = oct_file_generator.find_widest_interval_in_numbers(fluctuation_indices)
        # start_of_fluctuations = max_interval[0]-1
        end_of_fluctuations = fluctuation_indices[-1]
        if config.verbose>1: print(f"end of fluctuations: {end_of_fluctuations}")
        print(f"Signal starts fluctuating beyond 5% of max range at index {start_of_fluctuations}")
        print(f"Signal starts fluctuating beyond 5% of max range at index {end_of_fluctuations}")
    else:
        print("No significant fluctuation found.")
        start_of_fluctuations = 0

    # first_good_bin = start_of_fluctuations+int(5e-7/time_step)
    first_good_bin = 1
    last_good_bin = 1

    sigma = 10  # Standard deviation of the Gaussian kernel
    # Loop backwards through the array
    '''for i in range(len(signal) - 1, lookback_points - 1, -1):
        # Calculate the signal difference over the lookback window
        if signal[i] > shifted_min: continue
        if i < first_good_bin: continue
        signal_difference = signal[max(0,i - lookback_points)] - signal[i]
        # Update if the current difference is the largest increase
        if signal_difference > largest_increase:
            largest_increase = signal_difference
            start_index = start_of_fluctuations - lookback_points
            # start_index = i - lookback_points
            end_index = i # + lookback_points'''
    print(f"start of fluctuations: {start_of_fluctuations}")
    start_index = start_of_fluctuations - lookback_points

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
    if config.verbose>1:
        print(f"Largest increase: {largest_increase:.2f}")
        print(f"Start index: {start_index}, Time: {time[start_index]:.6e} s signal {signal[start_index]}")
        print(f"End index: {end_index}, Time: {time[end_index]:.6e} s signal {signal[end_index]}")


    # Shift the starting point by 0.1 microseconds
    shift_points = int(0.2e-6 / time_step)  # Number of points to shift earlier
    shifted_start_index = max(0, start_index)  # Ensure index is not negative  - shift_points
    if pulse >= 0:
        print("pulse over 1")
        print("shifted start index") 
        print(shifted_start_index)
        shifted_end_index, time, w = oct_file_generator.straight_line_across(time, w, shifted_start_index, signal_range)
        print(f"length of time {len(time)}, length of signal {len(signal)}")
        if shifted_start_index < np.max(signal[shifted_start_index:shifted_start_index+100]):
            too_narrow = True
            counter = 1
            while too_narrow:
                new_end, time, signal = straight_line_across(time, signal, shifted_end_index, signal_range)
                shifted_start_index = shifted_end_index
                shifted_end_index = new_end
                counter += 1
                if shifted_start_index >= np.max(signal[current_point_idx:current_point_idx+100]):
                    too_narrow = False
    else:
        # the line commented out below is an alternate method which assumes a fixed pulse width
        # shifted_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))
        shifted_end_index, time, signal = straight_line_across(time, signal, shifted_start_index, signal_range)
    print("shifted start index", shifted_start_index)
    print("shifted end index", shifted_end_index)
    # Determine the region to remove
    remove_start_index = shifted_start_index
    remove_end_index = shifted_end_index-1

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
                if config.verbose>1: print("stopped oscillation count")
                break
    region_min = signal[remove_start_index:remove_end_index + 1].min()
    region_max = signal[remove_start_index:remove_end_index + 1].max()
    region_range = region_max-region_min
    large_osc_threshold = 0.3*region_range
    large_oscs = 0
    for i in range(remove_start_index+10, remove_end_index+10):
        if abs(w[i] - signal[i]) > large_osc_threshold:
            large_oscs += 1
    osc_percentage = large_oscs/(remove_end_index-remove_start_index)
    if config.verbose>1: print(f"OSC COUNT: {large_oscs}")
    if config.verbose>1: print(f"LARGE OSC PERCENTAGE: {osc_percentage}")
    # Remove the area by setting the signal to NaN in this region
    signal_removed = np.copy(signal)
    print("start and end index", remove_start_index, remove_end_index)
    print(f"start value: {signal_removed[remove_start_index]} end value: {signal_removed[remove_end_index+1]}")
    signal_removed[remove_start_index:remove_end_index] = np.nan
    
    print(signal_removed[remove_end_index-5:remove_end_index+5])
    print(signal_removed[remove_start_index-5:remove_start_index+5])
    if math.isnan(signal_removed[0]):
        signal_removed[0] = signal[0]
    if math.isnan(signal_removed[-1]):
        signal_removed[-1] = signal[-1]
        # signal_removed[-1] =

    # Interpolate the missing values
    # if there is no signal: select 2 signal points to remove
    valid_mask = ~np.isnan(signal_removed)  # Mask to keep valid values
    if np.all(valid_mask == False):
        valid_mask[-1] = False
        valid_mask[0] = True
    print("interpolation inputs")
    print(signal_removed[valid_mask][remove_end_index-5:remove_end_index+5])
    temp_interpolated_baseline = np.interp(time, time[valid_mask], signal_removed[valid_mask])

    baseline_smoothed = gaussian_filter(temp_interpolated_baseline, sigma=sigma)

    #  Interpolate the missing values after smoothing
    drop_value = signal[remove_start_index]
    interpolated_baseline = np.interp(time, time[valid_mask], baseline_smoothed[valid_mask])
    interpolated_baseline[~valid_mask] = drop_value
    current_time = time[remove_end_index]
    
    time = np.insert(time,remove_end_index,current_time)
    signal = np.insert(signal,remove_end_index,signal[remove_end_index])
    interpolated_baseline = np.insert(interpolated_baseline,remove_end_index,drop_value)
    corrected_signal = signal-interpolated_baseline

    # adds saturation correction
    if show_saturation_correction:
        saturation_corrected = saturation_correction(remove_start_index+savgol_peak_i, remove_start_index+savgol_neg_peak_i, time, corrected_signal, w, exp_mode=True)
        saturation_corrected_linear = saturation_correction(remove_start_index+savgol_peak_i, remove_start_index+savgol_neg_peak_i, time, corrected_signal, w, exp_mode=False)
    else:
        saturation_corrected = []
        saturation_corrected_linear = []
    # calculate the area under the interpolated curve in the removed region
    removed_time = time[remove_start_index:remove_end_index]  # Time in the removed region
    corrected_signal_removed_values = corrected_signal[remove_start_index:remove_end_index]
    signal_removed_values = signal[remove_start_index:remove_end_index]
    signal_area   = abs(simpson(y=corrected_signal_removed_values, x=removed_time))
    w = savgol_filter(signal, 200, 2)

    bins_around=30
    baseline_offset=0
    ch1=corrected_signal
    selected_ch1=corrected_signal_removed_values
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
    if config.verbose>1:
        print(f"Shifted start index: {shifted_start_index}, Time: {time[shifted_start_index]:.6e} s")
        print(f"Removed area start time: {time[remove_start_index]:.6e} s")
        print(f"Removed area end time: {time[remove_end_index]:.6e} s")
        print(f"Area under interpolated curve in signal region: {signal_area:.3e} ")

    print(interpolated_baseline[remove_start_index-5:remove_start_index+5])
    print(interpolated_baseline[remove_end_index-5:remove_end_index+5])

    # Visualization
    if savePlot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, signal, label="Original Signal", color="blue")
        plt.plot(time, corrected_signal, label="Corrected Signal", color="green")
        plt.plot(time, interpolated_baseline, label="Interpolated Baseline", color="red")
        plt.plot(time[remove_start_index],interpolated_baseline[remove_start_index], marker='o', color='purple')
        plt.plot(time[remove_end_index],interpolated_baseline[remove_end_index], marker='o', color='purple')
        plt.plot(time, w, 'black', label="smoothed signal (savgol)")  # high frequency noise removed
        if show_saturation_correction:
            plt.plot(time, saturation_corrected, 'purple', label="saturation corrected - exponential")
            plt.plot(time, saturation_corrected_linear, 'orange', label="saturation corrected - simple linear")
        plt.axvspan(time[remove_start_index], time[remove_end_index], color="yellow", alpha=0.3, label="Removed Region")
        plt.axvline(time[shifted_start_index], color="green", linestyle="--", label="Shifted Start")
        plt.scatter(time[remove_start_index+savgol_peak_i], corrected_signal[remove_start_index+savgol_peak_i], color='purple', marker='x', label="oscillation peaks", zorder=2)
        plt.scatter(time[remove_start_index+savgol_neg_peak_i], corrected_signal[remove_start_index+savgol_neg_peak_i], color='purple', marker='x', zorder=2)
        plt.hlines(y=save_this_mean, xmin=time[0],xmax=time[-1], color='purple',linestyle='--',label="signal mean")
        plt.hlines(y=save_this_threshold, xmin=time[0], xmax=time[-1],color='yellow', linestyle='--',label="signal threshold")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal Suppression and Interpolation")
        plt.text(
            0.6 * max(time),  # X-coordinate (adjust based on your plot range)
            min(signal)+0.05 * (max(signal)-min(signal)),  # Y-coordinate (adjust based on your plot range)
            f"Signal Area: {signal_area:.3e}\nBeam: {beam}\nPulse: {pulse}\nZ: {Z}\nHV: {HV}\nosc_count: {osc_count}\nfile: {file_path[-14:-1]}v\nchannel: {selected_channel}",
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
        if config.verbose>2: print("graphicFile ", graphicFile)
        plt.savefig(graphicFile, format="jpeg", dpi=300)
        if showPlot:
            print("SHOW")
            plt.show()
        plt.close()

    return signal_area, nPeaks, osc_count