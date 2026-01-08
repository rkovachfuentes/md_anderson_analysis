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
from scipy.stats import chisquare
from scipy.optimize import differential_evolution
import warnings
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# reference dictionaries for units, column labels, pulse lengths and colors for plotting
color_dict = {"0.1":"yellow","0.5":"black","1.0":"blue","2.0":"green","3.0":"red"}
color_list = ["yellow","black","blue","green","red"]
pulse_list = ["0.1","0.5","1","2","3"]
unit_dict = {"Detector":None, "Channel":None, "Beam":None, "Pulse":"us", "Dose":"Gy", "X":"cm", "Z":"cm", "File":None, "ch1_area":"unitless", "ch2_area": "unitless", "HV":"V", "ch1_osc_count":"counts", "ch2_osc_count":"counts", "ch1_osc_perc":"%", "ch2_osc_perc":"%"}

##### FUNCTIONS FOR REGRESSION
def sqrt_func(x, a, b, c):
    # Ensure non-negative argument to sqrt during fitting iterations
    # If the optimizer tries a 'b' that makes x-b negative, return a large error
    # value so the optimizer avoids that parameter space.
    if np.any(x - b < 0):
        return 1e10 * np.ones_like(x) # Return large values for a bad fit
    return a * np.sqrt(x - b) + c

def sigmoid(x, L, x0, k, b):
    return L / (1. + np.exp(-k*(x-x0))) + b

# generates a polynomial fit for the specified degree, returns a polynomial (NOT a set of data)
def generate_polynomial_fit(x_data, y_data, degree):
    poly = PolynomialFeatures(degree=degree)
    y_pred = np.poly1d(np.polyfit(x_data,y_data,degree))
    if config.verbose > 0: print(f"polynomial generated: {y_pred}")
    return y_pred

##### REGRESSION PLOTTING FUNCTIONS

# plots a variable a vs variable b, grouped by group_var, with filters applied in filter_dict. filter format: {column name: filter value}
# no_3 is an optional argument which, when true, excludes 3.0 us pulses from the plot
def plot_a_vs_b_with_linreg(log_file, a, b, group_var, filter_dict={None:None}, savePlot = False, no_3 = False):
    ##### generates dataframe of log file, drops all rows that don't satisfy filters, groups by groupvar and a (in that order)
    log_df = pd.read_csv(log_file)
    # removes filtered values
    for filter in filter_dict.keys():
        if filter is None:
            break
        else:
            log_df = log_df[log_df[filter] == filter_dict[filter]]
    # groups values, selects first group (pulse length)
    grouped_mean_multi = log_df.groupby([group_var, a])
    for group_tuple, val in grouped_mean_multi:
        initial_grp = group_tuple[0]
        break
    # initializing empty lists to store the a, mean b, error values
    grp_means = []
    grp_a = []
    grp_errs = []
    z_labels = []
    # regression dict will be used later for the polynomial fits
    regression_dict = {}
    # creates plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # loops over all groups, generates mean b and error values
    for group_tuple, val in grouped_mean_multi:
        if no_3 and (str(group_tuple[0]) == "3.0"):
            pass
        else:
            # case 1: still in the same group, add to lists and continue
            if group_tuple[0] == initial_grp:
                grp_means.append(val[b].mean())
                if val[b].std() > 0:
                    grp_errs.append(val[b].std())
                else:
                    grp_errs.append(0.4)
                grp_a.append(group_tuple[1])
                z_labels.append(val["Z"].mean())
            # case 2: moved to a new group, aggregate the previous data and generate a plot
            else:
                plt.errorbar(grp_a, grp_means, yerr=grp_errs, color=color_dict[str(initial_grp)], label=f"{initial_grp} us", fmt='o')
                fitted_values = generate_polynomial_fit(grp_a,grp_means,2)
                smooth_range = np.linspace(np.min(grp_a),np.max(grp_a),100)
                # add values needed for regression to regression_dict
                if initial_grp in regression_dict.keys():
                    regression_dict[initial_grp][0].append(grp_a)
                    regression_dict[initial_grp][1].append(grp_means)
                else:
                    regression_dict[initial_grp] = [[],[]]
                    regression_dict[initial_grp][0].append(grp_a)
                    regression_dict[initial_grp][1].append(grp_means)
                # reset lists for the next group
                grp_means = []
                grp_a = []
                grp_errs = []
                z_labels = []
                # reset initial_grp to the new current group
                initial_grp = group_tuple[0]
                grp_means.append(val[b].mean())
                grp_errs.append(val[b].std())
                grp_a.append(group_tuple[1])
    # add the final group to the regression dict
    if initial_grp in regression_dict.keys():
        regression_dict[initial_grp][0].append(grp_a)
        regression_dict[initial_grp][1].append(grp_means)
    else:
        regression_dict[initial_grp] = [[],[]]
        regression_dict[initial_grp][0].append(grp_a)
        regression_dict[initial_grp][1].append(grp_means)
    # generate final plot
    plt.errorbar(grp_a, grp_means, yerr=0.2*np.array(grp_means), color=color_dict[str(initial_grp)], label=f"{initial_grp} us", fmt='o')
    # generate polynomial fits and plot them
    print("REGRESSION PLOTS")
    for initial_grp in regression_dict.keys():
        grp_a = regression_dict[initial_grp][0][0]
        print(f"initial grp: {initial_grp}")
        grp_means = regression_dict[initial_grp][1][0]
        fitted_values = generate_polynomial_fit(grp_a,grp_means,2)
        smooth_range = np.linspace(np.min(grp_a),np.max(grp_a),100)
        plt.plot(smooth_range, fitted_values(smooth_range), color_dict[str(initial_grp)])
    # create text box with filter names and values
    filter_keys = list(filter_dict.keys())
    filter_vals = list(filter_dict.values())
    filter_str_list = []
    for i in (0,len(filter_keys)-1):
        new_str = f"{filter_keys[i]} = {filter_vals[i]}\n"
        filter_str_list.append(new_str)
    mega_string = "".join(filter_str_list)
    # draw plot features
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
    ax.legend()
    plt.grid(True)
    plt.show()
    return