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
import oct_file_generator
import oct_plotter
import area_testing

range_dict = {}
date = "2025-10-15"

showPlot = config.showPlot 
savePlot = config.savePlot

if __name__ == "__main__":
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0775.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    signal_area_list = area_testing.signal_region_finder(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0775.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False, mode='old')
    area_testing.plot_signal_region(signal_area_list, date, pulse=1.0, ifile=1560, selected_channel="CH2")
    

    # open figure window and clear
    '''plt.figure(1)
    plt.clf()

    # first subplot: plots sinfunc and cosfunc on the same plot, with legend and gridlines
    plt.subplot(2,2,1)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    
    # second subplot: plots expsin function with shown polynomial fits to both the local maxima and minima
    plt.subplot(2,2,2)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)

    # third subplot: scatter plot of expsinsquare function
    plt.subplot(2,2,3)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0775.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)

    # fourth subplot: polar plot of cos()
    plt.subplot(2,2,4)
    signal_area_list = area_testing.signal_region_finder(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0775.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False, mode='old')
    area_testing.plot_signal_region(signal_area_list, date, pulse=1.0, ifile=1560, selected_channel="CH2")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.show()'''


