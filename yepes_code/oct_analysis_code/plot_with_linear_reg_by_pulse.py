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
from scipy.stats import chisquare
from scipy.optimize import differential_evolution
import warnings
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# from generate_all_files import plot_area_vs_distance

output_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas.csv"
lower_limits = {"0.1":1e-11, "0.5":1e-8,"1.0":1e-8,"2.0":1e-8,"3.0":1e-8}
parameterTuple = (0,0,0,0)
point_counter = 0

# The function to fit: y = a * sqrt(x - b) + c
def sqrt_func(x, a, b, c):
    # Ensure non-negative argument to sqrt during fitting iterations
    # If the optimizer tries a 'b' that makes x-b negative, return a large error
    # value so the optimizer avoids that parameter space.
    if np.any(x - b < 0):
        return 1e10 * np.ones_like(x) # Return large values for a bad fit
    return a * np.sqrt(x - b) + c

def sigmoid(x, L, x0, k, b):
    return L / (1. + np.exp(-k*(x-x0))) + b

def generate_polynomial_fit(x_data, y_data, degree):
    poly = PolynomialFeatures(degree=degree)
    y_pred = np.poly1d(np.polyfit(x_data,y_data,degree))
    return y_pred

# functions for generating initial guess parameters for logistic regression
# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(xData, yData):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return np.sum((yData - val) ** 2.0)

def generate_Initial_Parameters(xData, yData):
    parameterBounds = []
    parameterBounds.append([0.0, 100.0]) # search bounds for a
    parameterBounds.append([-10.0, 0.0]) # search bounds for b
    parameterBounds.append([0.0, 10.0]) # search bounds for c
    parameterBounds.append([0.0, 10.0]) # search bounds for d

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError(xData, yData), parameterBounds, seed=3)
    return result.x

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

def plot_a_vs_b_with_linreg(log_file, a, b, group_var, filter_dict={None:None}, savePlot = False, no_3 = False):
    point_counter = 0
    log_df = pd.read_csv(log_file)
    unit_dict = {"Detector":None, "Channel":None, "Beam":None, "Pulse":"us", "Dose":"Gy", "X":"cm", "Z":"cm", "File":None, "ch1_area":"unitless", "ch2_area": "unitless", "HV":"V", "ch1_osc_count":"counts", "ch2_osc_count":"counts", "ch1_osc_perc":"%", "ch2_osc_perc":"%"}
    # log_df = log_df[[a, b, filter_var, group_var]]
    for filter in filter_dict.keys():
        if filter is None:
            break
        else:
            log_df = log_df[log_df[filter] == filter_dict[filter]]
    # log_df = log_df[log_df["Z"] > 30]
    color_dict = {"0.1":"yellow","0.5":"black","1.0":"blue","2.0":"green","3.0":"red"}
    color_list = ["yellow","black","blue","green","red"]
    pulse_list = ["0.1","0.5","1","2","3"]
    grouped_mean_multi = log_df.groupby([group_var, a])
    fig, ax = plt.subplots(figsize=(8, 5))
    for group_tuple, val in grouped_mean_multi:
        initial_grp = group_tuple[0]
        break
    grp_means = []
    grp_a = []
    grp_errs = []
    z_labels = []
    regression_dict = {}
    for group_tuple, val in grouped_mean_multi:
        if no_3 and (str(group_tuple[0]) == "3.0"):
            pass
        else:
            if group_tuple[0] == initial_grp:
                grp_means.append(val[b].mean())
                if val[b].std() > 0:
                    grp_errs.append(val[b].std())
                else:
                    grp_errs.append(0.4)
                grp_a.append(group_tuple[1])
                z_labels.append(val["Z"].mean())
            else:
                plt.errorbar(grp_a/initial_grp, grp_means/initial_grp, yerr=grp_errs, color=color_dict[str(initial_grp)], label=f"{initial_grp} us", fmt='o')
                fitted_values = generate_polynomial_fit(grp_a,grp_means,1)
                smooth_range = np.linspace(np.min(grp_a),np.max(grp_a),100)
                # y_pred = sigmoid(grp_a, *popt)
                # plt.plot(grp_a, y_pred, color=color_dict[str(initial_grp)], linestyle='dashed')
                if initial_grp in regression_dict.keys():
                    regression_dict[initial_grp][0].append(grp_a)
                    regression_dict[initial_grp][1].append(grp_means)
                    print("added to existing")
                    print(regression_dict)
                else:
                    regression_dict[initial_grp] = [[],[]]
                    regression_dict[initial_grp][0].append(grp_a)
                    regression_dict[initial_grp][1].append(grp_means)
                    print("added new")
                grp_means = []
                grp_a = []
                grp_errs = []
                z_labels = []
                initial_grp = group_tuple[0]
                grp_means.append(val[b].mean())
                if val[b].std() > 0:
                    grp_errs.append(val[b].std())
                else:
                    grp_errs.append(0.4)
                grp_a.append(group_tuple[1])
    plt.errorbar(grp_a/initial_grp, grp_means/initial_grp, yerr=0.2*np.array(grp_means), color=color_dict[str(initial_grp)], label=f"{initial_grp} us", fmt='o')
    fitted_values = generate_polynomial_fit(grp_a,grp_means,1)
    smooth_range = np.linspace(np.min(grp_a),np.max(grp_a),100)
    # plt.errorbar(smooth_range/initial_grp, fitted_values(smooth_range), color=color_dict[str(initial_grp)], label=f"{initial_grp} best poly fit", fmt='-')
    # fitted_values = generate_polynomial_fit(grp_a,grp_means,2)
    # plt.errorbar(np.linspace(np.min(grp_a),np.max(grp_a),100), fitted_values, color=color_dict[str(initial_grp)], label=f"{initial_grp} best poly fit", fmt='-')
    # plot the linear regressions for all sets
    # generate all filter strings
    filter_keys = list(filter_dict.keys())
    filter_vals = list(filter_dict.values())
    filter_str_list = []
    for i in (0,len(filter_keys)-1):
        new_str = f"{filter_keys[i]} = {filter_vals[i]}\n"
        filter_str_list.append(new_str)
    mega_string = "".join(filter_str_list)
    # plt.plot(grp_a, modelPredictions, color=color_dict[str(initial_grp)], linestyle='dashed')
    plt.text(
            0.7 * max(log_df[a]),  # X-coordinate (adjust based on your plot range)
            min(log_df[b])+0.2 * (max(log_df[b])-min(log_df[b])),  # Y-coordinate (adjust based on your plot range)
            f"Filters:\n{mega_string}",
            fontsize=10,
            color="purple",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple')  # Optional styling
        )
    ax.set_title(f"{a} vs {b}, Grouped By {group_var} ({unit_dict[group_var]})")
    ax.set_xlabel(f"{a}/us pulse (Gy/us)")
    ax.set_ylabel(f"{b}/us pulse ({unit_dict[b]})")
    print("final point count")
    print(point_counter)
    ax.legend()
    plt.show()
    return

def alt_plot_a_vs_b(log_file, a, b, group_var, filter_dict={None:None}, savePlot = False):
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
    initial_grp = ""
    grp_a = []
    grp_errs = []
    color_idx = 0
    for group_tuple, val in grouped_mean_multi:
        print(group_tuple)
        if group_tuple[0] == initial_grp:
            grp_means.append(val[b].mean())
            if val[b].std() > 0:
                grp_errs.append(val[b].std())
            else:
                grp_errs.append(val[b]*0.2)
            grp_a.append(group_tuple[1])
        else:
            plt.errorbar(grp_a, grp_means, yerr=grp_errs, color=color_list[color_idx], label=f"{initial_grp}", fmt='-o')
            grp_means = []
            grp_a = []
            grp_errs = []
            initial_grp = group_tuple[0]
            color_idx += 1
            grp_means.append(val[b].mean())
            if val[b].std() > 0:
                    grp_errs.append(val[b].std())
            else:
                grp_errs.append(1)
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
# alt_plot_a_vs_b("rachels_data_final.csv", "Dose", "ch1_area", "HV", filter_dict={"HV":20, "Beam":"Electrons 85V"}, savePlot=True)
#plot_a_vs_b("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/rachels_data_final.csv", "Dose", "ch2_area", "HV", 20, "Pulse", savePlot=False, no_3 = True)
#plot_a_vs_b("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/rachels_data_final.csv", "Dose", "ch1_area", "HV", 20, "Pulse", savePlot=False, no_3 = True)

# plot_a_vs_b_both_mean("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv", "ch1_peaks", "ch2_peaks", "HV", 20, "Pulse", savePlot=False, no_3 = True)

# filter_small_values(output_file, lower_limits, "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/all_files_and_areas_dropped.csv")