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

range_dict = {}
date = "2025-10-15"

if __name__ == "__main__":
    oct_file_generator.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    oct_file_generator.get_area(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0718.csv", range_dict, f"{date}", pulse=3.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH1", show_saturation_correction=False)
    oct_file_generator.get_area_old(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)
    oct_file_generator.get_area_old(f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-0685.csv", range_dict, f"{date}", pulse=1.0, Z=30, HV=20, beam="Electrons 85", ifile=1560, selected_channel="CH2", show_saturation_correction=False)


