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
# from generate_all_files import plot_area_vs_distance

output_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas.csv"
lower_limits = {"0.1":1e-11, "0.5":1e-8,"1.0":1e-8,"2.0":1e-8,"3.0":1e-8}

def percent_diff_removed(original_dir, current_dir):
    original_lst = os.listdir(original_dir)
    current_lst = os.listdir(current_dir)
    # len_or = len(original_lst)
    len_or = 1173
    len_curr = len(current_lst)
    print((len_curr-len_or)/len_or)
    return (len_curr-len_or)/len_or

def filter_small_values(output_file, lower_limits, dropped_file):
    output_file_df = pd.read_csv(output_file)
    channels = ["CH1","CH2"]
    for index, row in output_file_df.iterrows():
        pulse = row["Pulse"]
        ifile = int((row["File"].split("-"))[-1].replace(".csv",""))
        if (float(row["ch1_area"]) < lower_limits[str(pulse)]) or (float(row["ch2_area"]) < lower_limits[str(pulse)]):
            graphicDir = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-07-30/all_files"
            for channel in channels:
                graphicFile=f"{graphicDir}/dose-calc-pulse={int(pulse)}-{ifile}-{channel}.jpg"
                print(graphicFile)
                if os.path.exists(graphicFile):
                    os.remove(graphicFile)
            output_file_df.drop(index, inplace=True)
    output_file_df.to_csv(dropped_file)
    return

def cleanup_output_file(output_file):
    output_file_df = pd.read_csv(output_file)
    column_list_tolist = output_file_df.columns.tolist()
    for column in column_list_tolist:
        if ("Unnamed" in column):
            output_file_df = output_file_df.drop(column, axis=1)
    output_file_df = output_file_df[output_file_df['ch1_area'] > 0]
    output_file_df = output_file_df[output_file_df['ch2_area'] > 0]
    print(output_file_df.head)
    output_file_df.to_csv(output_file)
    return

def plot_a_vs_b(log_file, a, b, filter_var, filter_val, group_var, savePlot = False, no_3 = False):
    log_df = pd.read_csv(log_file)
    log_df = log_df[[a, b, filter_var, group_var]]
    log_df = log_df[log_df[filter_var] == filter_val]
    log_df = log_df.drop(filter_var, axis=1)
    print(log_df.head())
    color_dict = {"0.1":"red","0.5":"black","1.0":"blue","2.0":"green","3.0":"yellow"}
    pulse_list = ["0.1","0.5","1","2","3"]

    grouped_mean_multi = log_df.groupby([group_var, a])
    print(grouped_mean_multi.head())
    print(grouped_mean_multi)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group_tuple, val in grouped_mean_multi:
        initial_grp = group_tuple[0]
        break
    print(initial_grp)
    grp_means = []
    grp_a = []
    grp_errs = []
    for group_tuple, val in grouped_mean_multi:
        if no_3 and (str(group_tuple[0]) == "3.0"):
            break
        else:
            if group_tuple[0] == initial_grp:
                grp_means.append(val[b].mean())
                grp_errs.append(val[b].std())
                grp_a.append(group_tuple[1])
            else:
                plt.errorbar(grp_a, grp_means, yerr=grp_errs, color=color_dict[str(initial_grp)], label=f"{initial_grp} us", fmt='-o')
                grp_means = []
                grp_a = []
                grp_errs = []
                initial_grp = group_tuple[0]
                print(initial_grp)

    ax.set_title(f"{a} vs Mean {b}, Grouped By {group_var}, HV={filter_val}")
    ax.set_xlabel(f"{a}")
    ax.set_ylabel(f"{b}")
    ax.legend()
    plt.show()
    
    return

def plot_a_vs_b_both_mean(log_file, a, b, filter_var, filter_val, group_var, savePlot = False, no_3 = False):
    log_df = pd.read_csv(log_file)
    log_df = log_df[[a, b, filter_var, group_var]]
    log_df = log_df[log_df[filter_var] == filter_val]
    log_df = log_df.drop(filter_var, axis=1)
    print(log_df.head())
    color_dict = {"0.1":"red","0.5":"black","1.0":"blue","2.0":"green","3.0":"yellow"}
    pulse_list = ["0.1","0.5","1","2","3"]

    grouped_mean_multi = log_df.groupby([group_var])
    print(grouped_mean_multi.head())
    print(grouped_mean_multi)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group, val in grouped_mean_multi:
        print(group)
        print(val)
        if no_3 and (str(group[0]) == "3.0"):
            break
        ax.errorbar(val[a].mean(), val[b].mean(), xerr=val[a].std(), yerr=val[b].std(), color=color_dict[str(group[0])], fmt='-^')
    ax.set_title(f"Mean {a} vs Mean {b}, Grouped By {group_var}, HV={filter_val}")
    ax.set_xlabel(f"{a}")
    ax.set_ylabel(f"{b}")
    plt.show()
    
    return

# percent_diff_removed("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-07-30/Beam=Electrons 85V-Z=60-HV=0","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-07-30/all_files")
plot_a_vs_b("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv", "Z", "ch2_area", "HV", 50, "Pulse", savePlot=False, no_3 = False)
# plot_a_vs_b_both_mean("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv", "ch1_peaks", "ch2_peaks", "HV", 20, "Pulse", savePlot=False, no_3 = True)

# filter_small_values(output_file, lower_limits, "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv")