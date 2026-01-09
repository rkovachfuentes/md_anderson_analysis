import csv
import config
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
from oct_file_generator import convert_dose_c_to_gy
from plot_with_linear_reg_by_pulse import plot_a_vs_b_with_linreg

# reference lists of beams and HV values
beam_list = [85, 110, 191]
HV_list = [20, 50, 100, 9, 10, 7, 0]
lower_limits = {"0.1":1e-11, "0.5":1e-8,"1.0":1e-8,"2.0":1e-8,"3.0":1e-8}
lower_limits = {"0.1":1e-11, "0.5":1e-8,"1.0":1e-8,"2.0":1e-8,"3.0":1e-8}

# helper function to count difference in # of rows between files
def count_file_no_dif(file1, file2):
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

# helper function to count the percent difference in # of files between 2 directories
def percent_diff_removed(original_dir, current_dir1, current_dir2):
    original_lst = os.listdir(original_dir)
    current_lst1 = os.listdir(current_dir1)
    current_lst2 = os.listdir(current_dir2)
    len_or = len(original_lst)
    len_curr = len(current_lst1) + len (current_lst2)
    print((len_curr-len_or)/len_or)
    return (len_curr-len_or)/len_or

# helper function to filter out files for plots with areas below an area threshold defined in the lower_limits dictionary
def filter_small_values(output_file, lower_limits, dropped_file):
    output_file_df = pd.read_csv(output_file)
    channels = ["CH1"]
    for index, row in output_file_df.iterrows():
        pulse = row["Pulse"]
        ifile = int((row["File"].split("-"))[-1].replace(".csv",""))
        if (float(row["ch1_area"]) <= 0.000001): # lower_limits[str(pulse)])
            graphicDir = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/new-plot-dose-2025-05-06/all_files"
            for channel in channels:
                graphicFile=f"{graphicDir}/dose-calc-pulse={int(pulse)}-{ifile}-{channel}.jpg"
                if os.path.exists(graphicFile):
                    os.remove(graphicFile)
            output_file_df.drop(index, inplace=True)
    output_file_df.to_csv(dropped_file)
    return

# helper function to drop unnamed files, files with negative areas, or with NaN column values
# also reformats "comment" column into an HV column
def cleanup_output_file(output_file):
    output_file_df = pd.read_csv(output_file)
    column_list_tolist = output_file_df.columns.tolist()
    for column in column_list_tolist:
        if ("Unnamed" in column):
            output_file_df = output_file_df.drop(column, axis=1)
    output_file_df = output_file_df[output_file_df['ch1_area'] > 0]
    output_file_df = output_file_df[output_file_df['ch2_area'] > 0]
    output_file_df = output_file_df.drop_duplicates()
    comments_new = []
    '''for comment in output_file_df["Comment"]:
        comment_clean = comment.replace("HV ", "")
        comments_new.append(comment_clean)
    output_file_df["HV"] = comments_new'''
    output_file_df.to_csv(output_file)
    return

###### PLOTTER FUNCTIONS
# plots column a vs column b, organized by groups and with filters as defined in filter_dict: {column: value}
# does not include regression feature; this function with regression is in plot_with_linear_reg_by_pulse
def plot_a_vs_b(log_file, a, b, group_var, filter_dict={None:None}, savePlot = False):
    log_df = pd.read_csv(log_file)
    print("unique HVs before slicing")
    print(log_df["HV"])
    print(log_df['HV'].unique())
    df_10 = log_df[(log_df["HV"] == 20)]
    unit_dict = {"Detector":None, "Channel":None, "Beam":None, "Pulse":"us", "Dose":"C*10^-8", "X":"cm", "Z":"cm", "File":None, "ch1_area":"unitless", "ch2_area": "unitless", "HV":"V", "ch1_osc_count":"counts", "ch2_osc_count":"counts", "ch1_osc_perc":"%", "ch2_osc_perc":"%"}
    for filter in filter_dict.keys():
        if filter is None:
            break
        else:
            log_df = log_df[log_df[filter] == filter_dict[filter]]
    color_dict = {"0.1":"red","0.5":"black","1.0":"blue","2.0":"green","3.0":"yellow"}
    color_list = ["red","black","blue","green","yellow"]
    pulse_list = ["0.1","0.5","1","2","3"]
    grouped_mean_multi = log_df.groupby([group_var, a])

    print("unique HVs after slicing")
    print(log_df['HV'].unique())
    df_10 = log_df[(log_df["HV"] == 20)]
    print(df_10["Z"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for group_tuple, val in grouped_mean_multi:
        initial_grp = group_tuple[0]
        break
    grp_means = []
    grp_a = []
    grp_errs = []
    color_idx = 0
    for group_tuple, val in grouped_mean_multi:
        print(group_tuple)
        if group_tuple[0] == initial_grp:
            grp_means.append(val[b].mean())
            grp_errs.append(val[b].std())
            grp_a.append(group_tuple[1])
        else:
            plt.errorbar(grp_a, grp_means, yerr=grp_errs, color=color_list[color_idx], label=f"{initial_grp}", fmt='-o')
            grp_means = []
            grp_a = []
            grp_errs = []
            initial_grp = group_tuple[0]
            color_idx += 1
            grp_means.append(val[b].mean())
            grp_errs.append(val[b].std())
            grp_a.append(group_tuple[1])
    plt.errorbar(grp_a, grp_means, yerr=grp_errs, color=color_list[color_idx], label=f"{initial_grp}", fmt='-o')
    # generate all filter strings
    filter_keys = list(filter_dict.keys())
    filter_vals = list(filter_dict.values())
    filter_str_list = []
    for i in (0,len(filter_keys)-1):
        new_str = f"{filter_keys[i]} = {filter_vals[i]}\n"
        filter_str_list.append(new_str)
    mega_string = "".join(filter_str_list)
    ax.set_title(f"{a} vs {b}, Grouped By {group_var} ({unit_dict[group_var]})")
    plt.text(
            0.7 * max(log_df[a]),  # X-coordinate (adjust based on your plot range)
            min(log_df[b])+0.2 * (max(log_df[b])-min(log_df[b])),  # Y-coordinate (adjust based on your plot range)
            f"Filters:\n{mega_string}",
            fontsize=10,
            color="purple",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple')  # Optional styling
        )
    ax.set_xlabel(f"{a} ({unit_dict[a]})")
    ax.set_ylabel(f"{b} ({unit_dict[b]})")
    ax.legend()
    plt.show()
    if savePlot:
        plt.savefig(f"plot_graveyard/{a}_vs_{b}_grouped_by_{group_var}_filters.png")
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

# generate plots here
# cleanup_output_file("final_oct_combined_file.csv")

log_file = "oct_combined_file.csv"
log_df = pd.read_csv(log_file)
print(log_df.head)
unique_beams = log_df['Beam'].unique()
unique_hv = log_df['HV'].unique()
print(unique_beams)
print(unique_hv)
unique_hv = np.delete(unique_hv, 0)
for hv in unique_hv:
    hv.replace("V","")
    hv = int(hv)

grouped_by_beam_hv = log_df.groupby(['Beam', 'HV'])
for group_tuple, val in grouped_by_beam_hv:
    print(group_tuple)
    beam = group_tuple[0]
    hv = group_tuple[1]
    plot_a_vs_b_with_linreg(log_file, "Dose", "ch1_area", "Pulse", filter_dict={"Beam":f"{beam}","HV":f"{hv}"}, savePlot=True, no_3=False, no_means=True)
    plot_a_vs_b_with_linreg(log_file, "Dose", "ch1_area", "Pulse", filter_dict={"Beam":f"{beam}","HV":f"{hv}"}, savePlot=True, no_3=False, no_means=False)
    plot_a_vs_b_with_linreg(log_file, "Dose", "ch2_area", "Pulse", filter_dict={"Beam":f"{beam}","HV":f"{hv}"}, savePlot=True, no_3=False, no_means=True)
    plot_a_vs_b_with_linreg(log_file, "Dose", "ch2_area", "Pulse", filter_dict={"Beam":f"{beam}","HV":f"{hv}"}, savePlot=True, no_3=False, no_means=False)