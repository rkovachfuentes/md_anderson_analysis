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
date = "2025-10-14"

showPlot = config.showPlot 
savePlot = config.savePlot

if __name__ == "__main__":
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-1480.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    area_testing.combined_get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-1020.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)
    # area_testing.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    # area_testing.get_area_old(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-1019.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    #area_testing.get_area_old(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0775.csv", range_dict, f"{date}", pulse=3.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    # LGAD_data = area_testing.signal_region_finder_new(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    # diode_data = area_testing.signal_region_finder_old(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0745.csv", range_dict, f"{date}", pulse=1.0, Z=20, HV=7, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)
    # area_testing.plot_signals_only(LGAD_data, diode_data, "LGAD", "diode")
    
    '''area_testing.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    area_testing.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0718.csv", range_dict, f"{date}", pulse=3.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    area_testing.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)
    area_testing.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)'''




