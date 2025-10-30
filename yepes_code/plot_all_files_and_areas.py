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

output_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_06_all_files_and_areas.csv"
lower_limits = {"0.1":1e-11, "0.5":1e-8,"1.0":1e-8,"2.0":1e-8,"3.0":1e-8}

def count_file_no_dif(file1, file2):
    file_path = 'your_file.csv'
    with open(file1, 'r') as f1:
        reader1 = csv.reader(f1)
        row_count1 = len(list(reader1))
    f1.close()
    with open(file2, 'r') as f2:
        reader = csv.reader(f2)
        row_count2 = len(list(reader1))
    f2.close()
    print((row_count_1-row_count_2)/row_count_2)
    return (row_count_1-row_count_2)/row_count_2

def percent_diff_removed(original_dir, current_dir1, current_dir2):
    original_lst = os.listdir(original_dir)
    current_lst1 = os.listdir(current_dir1)
    current_lst2 = os.listdir(current_dir2)
    len_or = len(original_lst)
    len_curr = len(current_lst1) + len (current_lst2)
    print((len_curr-len_or)/len_or)
    return (len_curr-len_or)/len_or

def filter_small_values(output_file, lower_limits, dropped_file):
    output_file_df = pd.read_csv(output_file)
    channels = ["CH1"]
    for index, row in output_file_df.iterrows():
        pulse = row["Pulse"]
        ifile = int((row["File"].split("-"))[-1].replace(".csv",""))
        if (float(row["ch1_area"]) < lower_limits[str(pulse)]):
            graphicDir = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-05-06/all_files"
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

def plot_a_vs_b(log_file, a, b, a_unit, b_unit, filter_var, filter_val, group_var, savePlot = False, no_3 = False):
    log_df = pd.read_csv(log_file)
    log_df = log_df[["Beam", a, b, filter_var, group_var]]
    log_df = log_df[log_df["Beam"] == "Electrons 85V"]
    log_df = log_df[log_df[filter_var] == filter_val]
    log_df = log_df.drop(filter_var, axis=1)
    log_df = log_df.drop("Beam", axis=1)
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

    ax.set_title(f"{a} ({a_unit}) vs Mean {b} ({b_unit}), Grouped By {group_var}, HV={filter_val}")
    ax.set_xlabel(f"{a}")
    ax.set_ylabel(f"{b}")
    ax.legend()
    if savePlot:
        plt.savefig(f"plot_graveyard/{a}_vs_{b}_grouped_by_{group_var}_{filter_var}={filter_val}.png")
    
    return

def plot_a_vs_b_fixed_Z(log_file, a, b, a_unit, b_unit, Z_value, filter_var, filter_val, group_var, savePlot = False, no_3 = False):
    log_df = pd.read_csv(log_file)
    log_df = log_df[[a, b, filter_var, group_var, "Z"]]
    log_df = log_df[log_df[filter_var] == filter_val]
    log_df = log_df[log_df["Z"] == Z_value]
    log_df = log_df.drop(filter_var, axis=1)
    log_df = log_df.drop("Z", axis=1)
    print(log_df.head())
    color_dict = {"0.1":"red","0.5":"black","1.0":"blue","2.0":"green","3.0":"yellow"}
    pulse_list = ["0.1","0.5","1","2","3"]
    grouped_mean_multi = log_df.groupby([group_var])
    fig, ax = plt.subplots(figsize=(8, 5))
    for group_tuple, val in grouped_mean_multi:
        initial_grp = group_tuple[0]
        break
    grp_means = []
    grp_a = []
    grp_errs = []
    for group_tuple, val in grouped_mean_multi:
        if no_3 and (group_tuple[0] == 3):
            break
        plt.errorbar(val[a].mean(), val[b].mean(), yerr=val[b].std(), xerr = val[a].std(), color=color_dict[str(group_tuple[0])], label=f"{group_tuple[0]} us", fmt='-o')
        initial_grp = group_tuple[0]

    ax.set_title(f"{a} vs Mean {b}, Grouped By {group_var}, {filter_var}={filter_val} Z={Z_value}")
    ax.set_xlabel(f"{a}, {a_unit}")
    ax.set_ylabel(f"{b}, {b_unit}")
    ax.legend(loc='upper left')
    plt.show()
    return

def plot_a_vs_b_both_mean(log_file, a, b, a_unit, b_unit, filter_var, filter_val, group_var, savePlot = False, no_3 = False):
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
    ax.set_title(f"Mean {a} ({a_unit}) vs Mean {b} ({b_unit}), Grouped By {group_var}, {filter_var}={filter_val}")
    ax.set_xlabel(f"{a}")
    ax.set_ylabel(f"{b}")
    plt.show()
    
    return

# dropped_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_06_all_files_and_areas_dropped.csv"
# filter_small_values(output_file, lower_limits, dropped_file)
file_05 = pd.read_csv("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_05_all_files_and_areas_dropped.csv")
file_06 = pd.read_csv("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_06_all_files_and_areas_dropped.csv")
# combined_df = pd.concat([file_05, file_06], ignore_index=True)
# combined_df.to_csv('/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/combined_file.csv', index=False)
print("may 6")
# count_file_no_dif("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_06_all_files_and_areas_dropped.csv", "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_06_all_files_and_areas.csv")
print("may 5")
# count_file_no_dif("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_05_all_files_and_areas_dropped.csv", "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/05_05_all_files_and_areas.csv")
HV_list = [10, 20, 30, 40, 50, 60, 80, 100]
# dist_list = [50, 60, 70]
# plot_a_vs_b("combined_file.csv", "Z", "Dose", "HV", 60, "Pulse", savePlot=False, no_3 = False)
'''plots_to_gen = [["Dose", "ch1_area", "C*10^-8", ""], ["Z", "ch1_area", "cm from collimator edge", ""], ["Z", "Dose", "cm from collimator edge", "C*10^-8"]]
for lst in plots_to_gen:
    for HV in HV_list:
        plot_a_vs_b("combined_file.csv", lst[0], lst[1], lst[2], lst[3], "HV", HV, "Pulse", savePlot=True, no_3 = False)'''

plot_a_vs_b("combined_file.csv", "Dose", "ch1_area", "C*10^-8", "", "HV", "HV", "Pulse", savePlot=True, no_3 = False)

# percent_diff_removed("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-07-30/all_files","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-05-05/all_files","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-05-06/all_files")
# plot_a_vs_b_both_mean("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv", "ch1_peaks", "ch2_peaks", "HV", 20, "Pulse", savePlot=False, no_3 = True)
# filter_small_values(output_file, lower_limits, "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv")