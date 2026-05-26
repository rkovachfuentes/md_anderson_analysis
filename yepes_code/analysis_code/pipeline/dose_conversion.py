import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
from tqdm import tqdm
import config
import area_testing
from scipy.signal import savgol_filter
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit
from PIL import Image

dose_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/dose_scaling.csv"
dose_scale_file = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/dose_scale_factors.csv"

def find_noise_threshold(signal, idx=500):
    first_noise = signal[:idx]
    sigma = np.std(first_noise)
    return max(10 * sigma, 0.001)

def dose_model(d, A, z0, b):
    # A is scale, z0 is the source offset, b is the power (usually near 2)
    return A / ((d + z0)**b)

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import config

# Global cache to store fitted parameters and scale factors
_MODEL_CACHE = {}

def dose_model(d, A, z0, b):
    return A / ((d + z0)**b)

def get_fitted_params(dose_ref_csv, collimator_length_cm, dist_from_col):
    """
    Computes the curve_fit parameters once per collimator setting.
    """
    cache_key = (dose_ref_csv, collimator_length_cm, dist_from_col)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    dose_df = pd.read_csv(dose_ref_csv, skiprows=2)
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] != "Uncollimated"]
    dose_df["Collimation (cm, diameter)"] = dose_df["Collimation (cm, diameter)"].astype(float)
    
    # Filter for specific collimator and reference pulse width
    subset = dose_df[
        (dose_df["Collimation (cm, diameter)"] == float(collimator_length_cm)) & 
        (dose_df["PW (electron pulse, us, FWHM)"] == 1.01)
    ].copy()

    if subset.empty:
        raise ValueError(f"No reference data found for Collimator: {collimator_length_cm}")

    x_points = subset["dist. collimator exit (m)"] if dist_from_col else subset["dist. beam exit (m)"]
    y_points = subset["Gy/P"].astype(float)

    initial_guess = [max(y_points), 0.05, 2.0]
    params, _ = curve_fit(dose_model, x_points, y_points, p0=initial_guess, 
                          bounds=(0, [np.inf, 1.0, 5.0]))
    
    _MODEL_CACHE[cache_key] = params
    return params

def get_scale_factors_map(dose_scale_factors_csv):
    """
    Loads the scale factor CSV once and returns a dictionary for O(1) lookup.
    """
    if dose_scale_factors_csv in _MODEL_CACHE:
        return _MODEL_CACHE[dose_scale_factors_csv]

    dsf = pd.read_csv(dose_scale_factors_csv)
    dsf["Beam (V)"] = dsf["Beam (V)"].astype(str).str.extract('(\d+)')[0] # Ensure numeric string
    
    # Create a lookup dictionary: (Beam, PW) -> Factor
    sf_map = dsf.set_index(['Beam (V)', 'Nominal PW (us)'])['Relative output (relative to 85V beam, 0.5us)'].to_dict()
    
    _MODEL_CACHE[dose_scale_factors_csv] = sf_map
    return sf_map

def convert_dose_optimized(beam, dist_m, pulse_width, params, sf_map):
    """
    Performs only the math, no file I/O or fitting.
    """
    try:
        beam_id = ''.join(filter(str.isdigit, str(beam)))
        
        # 1. Calculate base dose from pre-fitted model
        dose_gy = dose_model(dist_m, *params)
        
        # 2. Apply the 1.3 constant (from your original logic)
        dose_gy = dose_gy / 1.3

        # 3. Lookup scale factor
        scale_factor = sf_map.get((beam_id, float(pulse_width)))

        if scale_factor is None:
            return np.nan
        
        return dose_gy * scale_factor
    except Exception as e:
        return 0

def file_dose_converter(input_file, output_file, dose_ref_csv, dose_scale_factors_csv, collimator_length_cm=2.0, dist_from_col=True):
    # PRE-COMPUTE: Do these ONCE per file
    params = get_fitted_params(dose_ref_csv, collimator_length_cm, dist_from_col)
    sf_map = get_scale_factors_map(dose_scale_factors_csv)
    
    df = pd.read_csv(input_file)
    
    # Use .apply() for speed over row iteration
    df['Dose (Gy)'] = df.apply(
        lambda row: convert_dose_optimized(
            row['Beam'], 
            float(row['Z'])/100, 
            row['Pulse'], 
            params, 
            sf_map
        ), axis=1
    )
    
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} successfully.")

'''def convert_dose(dose_ref_csv, dose_scale_factors_csv, beam, dist_m, pulse_width, collimator_length_cm, dist_from_col=True):
    beam = ''.join(filter(str.isdigit, beam))
    # read first csv file containing 85v pulse info
    dose_df = pd.read_csv(dose_ref_csv,skiprows=2)
    # filter by selected collimator length for a 1.0 length pulse
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] != "Uncollimated"]
    dose_df["Collimation (cm, diameter)"] = dose_df["Collimation (cm, diameter)"].astype(float)
    dose_df = dose_df[dose_df["Collimation (cm, diameter)"] == float(collimator_length_cm)]
    dose_df = dose_df[dose_df["PW (electron pulse, us, FWHM)"] == 1.01]
    # dist_from_col is a bool which pulls from the appropriate column depending on where dist_cm is measured from
    # the resulting points are used in an exponential regression to fit the inputted distance and extrapolate its dose
    if dist_from_col:
        x_points = dose_df["dist. collimator exit (m)"]
    else:
        x_points = dose_df["dist. beam exit (m)"]
    try:
        # modified inverse square law fit (shifted by an offset to accommodate points near the x minimum and maximum)
        y_points = (dose_df["Gy/P"]).astype(float) #/(dose_df["PW (electron pulse, us, FWHM)"]).astype(float)
        ylog_points = np.log(y_points)
        if config.verbose>1: print(f"log data points: {ylog_points}")
        initial_guess = [max(y_points), 0.05, 2.0]
        # we constrain z0 and b to stay physically realistic
        params, _ = curve_fit(dose_model, x_points, y_points, p0=initial_guess, 
                            bounds=(0, [np.inf, 1.0, 5.0]))
        A_fit, z0_fit, b_fit = params
        # calculate the specific desired dose - note this result is a rate (Gy/P, P=1 us)
        dose_gy = dose_model(dist_m, A_fit, z0_fit, b_fit) # this will give dose in Gy
        smooth_range = np.linspace(min(x_points),max(x_points),100)
        # determine dose in gy for 85V beam based on this extrapolated fit and the given distance in m
        # dose_gy = (np.exp(coeffs[1]) * np.exp(coeffs[0]*(float(dist_m))))
        # to change from 1.0 beam (table 1) to 0.5 beam (table 2) divide by the appropriate scaling factor
        dose_gy = dose_gy/1.3

        # Load scale factors
        dsf = pd.read_csv(dose_scale_factors_csv)
        
        # Clean dtypes to ensure the match works regardless of CSV formatting
        dsf["Beam (V)"] = dsf["Beam (V)"].astype(str)
        dsf["Nominal PW (us)"] = dsf["Nominal PW (us)"].astype(float)
        
        # Single robust filter
        matched_factor = dsf[
            (dsf["Beam (V)"] == str(beam)) & 
            (dsf["Nominal PW (us)"] == float(pulse_width))
        ]['Relative output (relative to 85V beam, 0.5us)']

        if matched_factor.empty:
            print(f"!!! MISSING SCALE FACTOR for Beam: {beam}, Pulse: {pulse_width} !!!")
            return np.nan # Using NaN is better than 0 to distinguish "Error" from "No Signal"
        
        scale_factor = matched_factor.values[0]
        if config.verbose > 1: print(f"final dose: {dose_gy * scale_factor}")
        return dose_gy * scale_factor
    except:
        print("Error, returning 0 dose")
        return 0
    
def file_dose_converter(input_file, output_file, dose_ref_csv, dose_scale_factors_csv, collimator_length_cm=2.0, dist_from_col=True):
    df = pd.read_csv(input_file)
    doses = []
    for _, row in df.iterrows():
        beam = row['Beam']
        dist_m = float(row['Z'])/100 # convert from cm to m
        pulse_width = float(row['Pulse'])
        converted_dose = convert_dose(dose_ref_csv, dose_scale_factors_csv, beam, dist_m, pulse_width, collimator_length_cm, dist_from_col)
        doses.append(converted_dose)
    df['Dose (Gy)'] = doses
    df.to_csv(output_file)
    print("doses converted successfully")'''

if __name__ == "__main__":
    file_dose_converter("total_10_15_CH1.csv","total_10_15_CH1_with_dose.csv",dose_file, dose_scale_file)
    file_dose_converter("total_10_15_CH2.csv","total_10_15_CH2_with_dose.csv",dose_file, dose_scale_file)