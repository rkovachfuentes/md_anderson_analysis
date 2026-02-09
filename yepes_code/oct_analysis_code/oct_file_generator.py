import config
from alive_progress import alive_bar; import time
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
import area_testing

showPlot = config.showPlot 
savePlot = config.savePlot

# dose_file includes scaling information for doses for different beams, distances
dose_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/dose_scaling.csv"
dose_scale_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/dose_scale_factors.csv"

pulses=np.array([0.5,1.0,2.0,3.0],dtype=float)

range_dict = {"0.1":[699,np.inf],"0.5":[551, 1105],"1":[-np.inf,1121],"2":[-np.inf,1134],"3":[253,1103]}

def find_widest_interval_in_numbers(data):
    if len(data) < 2:
        return None, None # Not enough elements to form an interval
    
    sorted_data = sorted(data)
    max_width = 0
    widest_interval = (None, None)
    
    for i in range(len(sorted_data) - 1):
        # Calculate the width between adjacent elements
        width = sorted_data[i+1] - sorted_data[i]
        
        if width > max_width:
            max_width = width
            widest_interval = (sorted_data[i], sorted_data[i+1])
            
    return widest_interval, max_width
#==================================================================================================
#
#==================================================================================================
# reads scope csv files and extracts time and channel data
# returns either ch1 or ch2 data depending on selected channel
def read_csv(file_path_init, selected_channel="CH1"):
    time = []
    ch1 = []
    ch2 = []
    num_columns = 0
    file_path_check = Path(str(file_path_init))
    if file_path_check.is_file():
        if config.verbose > 1: print("filepath found")
        file_path = file_path_init
        pass
    else:
        file_path = str(file_path_init)
        file_path = file_path.replace("/home/lgad/data","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code")
        if config.verbose > 1: print(file_path)
    try:
        with open(file_path, mode='r') as file:
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

# takes in a list of peak values, both positive and negative, time, and signal and connects them either in linear or exponential mode
# returns a corrected array representing the saturation-corrected signal
def saturation_correction(positive_peaks_i, negative_peaks_i, time, signal, w, exp_mode=False):
    corrected_signal = []
    interval_lengths = []
    inner_interval_lengths = []
    interval_lengths.append(negative_peaks_i[0]+1)
    if exp_mode:
            x_points = time[negative_peaks_i]
            y_points = signal[negative_peaks_i]
            ylog_points = np.log(-y_points)
            if config.verbose>1: print(f"log data points: {ylog_points}")
            coeffs = np.polyfit(x_points, ylog_points, 1)
            corrected_signal = -(np.exp(coeffs[1]) * np.exp(coeffs[0]*(time[negative_peaks_i[0]:negative_peaks_i[-1]])))
            corrected_signal = list(corrected_signal)
    else:
        for segment in range(0,len(negative_peaks_i)-1):
            x_points = [time[negative_peaks_i[segment]], time[negative_peaks_i[segment+1]]]
            y_points = [signal[negative_peaks_i[segment]], signal[negative_peaks_i[segment+1]]]
            coeffs = np.polyfit(x_points, y_points, 1)
            p = np.poly1d(coeffs)
            polyfit_data = p(time[negative_peaks_i[segment]:negative_peaks_i[segment+1]])
            if config.verbose>1: print(f"polyfit data shape: {polyfit_data.shape}")
            corrected_signal = corrected_signal + list(polyfit_data)
            if config.verbose>1: print(f"interval length: {(negative_peaks_i[segment+1]-negative_peaks_i[segment])}")
            interval_lengths.append(negative_peaks_i[segment+1]-negative_peaks_i[segment])
            inner_interval_lengths.append(negative_peaks_i[segment+1]-negative_peaks_i[segment])
            if config.verbose>1: print(f"polyfit data length: {len(polyfit_data)}")
            polyfit_data = []
    if config.verbose>1:
        print(f"inner interval corrected signal length: {len(corrected_signal)}")
        print(f"pre-interval data length: {negative_peaks_i[0]}")
        print(f"post-interval data length: {len(signal)-negative_peaks_i[-1]}")
    interval_lengths.append(len(signal)-negative_peaks_i[-1])
    corrected_signal = list(signal[0:negative_peaks_i[0]+1]) + corrected_signal + list(signal[negative_peaks_i[-1]:-1])
    if config.verbose>1:
        print(f"interval lengths list: {interval_lengths}")
        print(f"total length: {sum(interval_lengths)}")
        print(f"pre and post interval total length: {negative_peaks_i[0]+len(signal)-negative_peaks_i[-1]}")
        print(f"inner interval total length: {sum(inner_interval_lengths)}")
    return corrected_signal

# cleanup files corrects files which aren't separated between ch1 and ch2, or files in a dictionary designated to be removed
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

# straight_line_across takes in time data, channel data, and the start and end points of the signal range and draws a straight line
# between the y values at the ends of the interval window
def straight_line_across(time_data, channel_data, drop_idx, signal_range):
    concat_channel_data = channel_data[(drop_idx):]
    drop_value = channel_data[drop_idx]
    return_value = 0
    # if config.verbose > 1: print(f"drop value: {drop_value}")
    indices = [i for i, x in enumerate(concat_channel_data) if x >= drop_value]
    prev = 0
    endpoints, width = find_widest_interval_in_numbers(indices)
    return_value = endpoints[1]+drop_idx
    if config.verbose > 1:
        print("returning value",channel_data[return_value])
        print("drop value", channel_data[drop_idx])
    return return_value, time_data, channel_data

# separate_comment filters through the "comment" column of csv files and extracts the HV value from them, saving it to a separate
# column in the file
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
# do not plot 85V data for SiC
# 

# plot_dose_vs_area plots dose vs area across a log file - it's a test function only, doesn't filter by distance or other values
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

# similar to the above, another test function that plots area vs dist
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

######## GENERATOR
# the generator function automatically runs get_area over all log and scope files for a specified date saving the results to an outfile
def generator(log_file, output_file, date, sensor):
    log_df = pd.read_csv(log_file)
    print(log_df.shape[0])
    Detector = sensor
    if not os.path.exists(output_file):
        print("does not exist")
        df = pd.DataFrame(columns=['Detector','Channel','Beam','Pulse','Dose','X','Z','File','ch1_area','ch2_area','ch1_peaks','ch2_peaks', 'ch1_osc_count', 'ch2_osc_count'])
        df.to_csv(output_file)
    output_file_df = pd.read_csv(output_file)

    for pulse in pulses:
        matching_rows = log_df[
            (log_df["Detector"] == Detector) &
            (log_df["Pulse"] == pulse)
        ]
        if config.verbose>1:
            print(f"MATCHING ROWS for {Detector} and {pulse}")
            print(matching_rows)
            # print("matching rows ", matching_rows["Detector", "Beam", "Z", "X", "Pulse", "Dose"])
        for _, row in matching_rows.iterrows():
            file_min = str(row["FileMin"])
            file_max = str(row["FileMax"])
            file_min_num = file_min.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_min_num = int(file_min_num.replace(".csv",""))
            file_max_num = file_max.replace(f"/home/lgad/data/{date}/scope-results-{date}-", "")
            file_max_num = int(file_max_num.replace(".csv",""))
            for i in range(file_min_num, file_max_num + 1):
                file_path = re.sub(r'\d{4}(?=\.csv)', f"{i:04}", file_min)
                file_path = row["FileMin"].replace(f"{file_min_num:04}",f"{i:04}")
                file_path = file_path.replace("/home/lgad/data/","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/")
                if not os.path.exists(file_path):
                    file_path = file_path.replace("CH1","CH2")
                    if not os.path.exists(file_path):
                        if config.verbose > 1: print("{file_path} does not exist, skip.")
                        continue
                # get area for both ch1 and ch2
                try:
                    yield
                    print("ch1 computation")
                    ch1_signal_area, ch1_signal_peak, ch1_osc = area_testing.combined_get_area(file_path, range_dict, date, pulse=pulse,ifile=i, selected_channel="CH1")
                    print("ch2 computation")
                    ch2_signal_area, ch2_signal_peak, ch2_osc = area_testing.combined_get_area(file_path, range_dict, date, pulse=pulse,ifile=i, selected_channel="CH2")
                except Exception as e:
                    print(f"Error reading CSV file: {e}")
                    print("couldn't determine area")
                    break
                if ((ch1_signal_area > 0.0)):
                    # new_dose = convert_dose_c_to_gy(dose_file, 0, row['Z'], row['Pulse'], 2, dist_from_col=True)
                    new_dose = 0
                    new_row_data = {'Detector': row['Detector'], 'Channel': row['Channel'], 'Beam': row['Beam'], 'Pulse': row['Pulse'], 'Dose': new_dose, 'HV': row['Comment'], 'X': row['X'], 'Z': row['Z'], 'File': file_path, 'ch1_area': ch1_signal_area, 'ch2_area': ch2_signal_area, 'ch1_peaks': ch1_signal_peak, 'ch2_peaks': ch2_signal_peak, 'ch1_osc_count': ch1_osc, 'ch2_osc_count': ch2_osc}
                    new_row_df = pd.DataFrame([new_row_data])
                    # Concatenate the DataFrames
                    output_file_df = pd.concat([output_file_df, new_row_df], ignore_index=True)
                    if config.verbose>-1:
                        print(f"ch1 i {i} pulse {pulse} signal_area {ch1_signal_area}", )
                else:
                    print("no signal area")
                    continue
    print(output_file_df.head())
    output_file_df.to_csv(output_file,index=False)
    # output_file_df = output_file_df.dropna(subset=['ch1_osc_count', 'ch1_osc_count'])
    return(output_file)

def convert_dose(dose_ref_csv, dose_scale_factors_csv, beam, dist_cm, pulse_width, collimator_length_cm, dist_from_col=True):
    # convert cm to m for lookup table
    dist_cm = dist_cm/100
    # read first csv file containing 85v pulse info
    dose_df = pd.read_csv(dose_ref_csv,skiprows=2)
    # filter by selected collimator length
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] == str(collimator_length_cm)]
    # dist_from_col is a bool which pulls from the appropriate column depending on where dist_cm is measured from
    # the resulting points are used in an exponential regression to fit the inputted distance and extrapolate its dose
    if dist_from_col:
        x_points = dose_df["dist. collimator exit (m)"]
    else:
        x_points = dose_df["dist. beam exit (m)"]
    # exponential fit
    y_points = (dose_df["Gy/P"]).astype(float)/(dose_df["PW (electron pulse, us, FWHM)"]).astype(float)
    ylog_points = np.log(y_points)
    if config.verbose>1: print(f"log data points: {ylog_points}")
    coeffs = np.polyfit(x_points, ylog_points, 1)

    # determine dose in gy for 85V beam based on this extrapolated fit and the given distance in m
    dose_gy = (np.exp(coeffs[1]) * np.exp(coeffs[0]*(float(dist_cm))))
    if beam == '85':
        if config.verbose > 1: print(f"final dose: {dose_gy}")
        return dose_gy
    # if beam is not 85, multiply by the appropriate scaling factor stored in dose_scale_factors_csv
    dose_scale_factors = pd.read_csv(dose_scale_factors_csv)
    dose_scale_factors = dose_scale_factors[dose_scale_factors["Beam (V)"] == beam]
    dose_scale_factors = dose_scale_factors[dose_scale_factors["Nominal PW (us)"] == pulse_width]
    scale_factor = dose_scale_factors['Relative output (relative to 85V beam, 0.5us)'].values[0]
    if config.verbose > 1: print(f"final dose: {dose_gy*scale_factor}")
    return(dose_gy*scale_factor)

# generator("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-2025-11-20-log.csv", "november_output.csv", "2025-11-20")
if __name__ == "__main__":
    sensor = "BNL"
    month, day = 10, 15
    convert_dose(dose_file, dose_scale_file, 110, 60, 0.5, 2, dist_from_col=True)
    '''with alive_bar(1000) as bar:
        for i in generator(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-2025-{month}-{day}-log.csv", f"retried-{month}_{day}_data.csv", f"2025-{month}-{day}", sensor):
            bar()'''


    
    '''sensor = "BNL"
    month, day = 10, 15
    generator(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-2025-{month}-{day}-log.csv", f"{month}_{day}_data.csv", f"2025-{month}-{day}", sensor)'''
    # generator("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-2025-11-20-log.csv", "nov_data.csv", "2025-11-20")

##### SOME USEFUL TEST FILES FOR AREA TESTING
# testing on a single file with a lot of oscillations
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0339.csv", range_dict, "2025-10-15", pulse=3, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH1", show_saturation_correction=True)
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0339.csv", range_dict, "2025-10-15", pulse=3, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH2", show_saturation_correction=True)

# a single file with very few oscillations
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-0897.csv", range_dict, "2025-10-15", pulse=1, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH1" )

# a weird one
# get_area("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/2025-10-15/scope-results-2025-10-15-1272.csv", range_dict, "2025-10-15", pulse=3, Z=50, HV=100, beam="Electrons 85V", ifile=1560, selected_channel="CH1" )