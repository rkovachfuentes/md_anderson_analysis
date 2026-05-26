import config
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import re
from scipy import stats
from pathlib import Path
import oct_file_generator
import oct_plotter
import area_testing
from PIL import Image

R_OHM = 50.0
beam_filter = 'Electrons 85V'
HV_filter = 'HV 100'

import pandas as pd

def sync_csv_files(file1_path, file2_path, match_col='b', output_path='updated_file2.csv'):
    """
    If match_col in file1 has a matching value to match_col in file2,
    overwrites the values in file2 with the values from file1.
    """
    # 1. Load the data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    # clean duplicates
    df1 = df1.drop_duplicates(subset=match_col)
    df2 = df2.drop_duplicates(subset=match_col)

    # 2. Set the 'match_col' as the index for both. 
    # This is required for .update() to know which rows correspond.
    df1_indexed = df1.set_index(match_col)
    df2_indexed = df2.set_index(match_col)

    # 3. Update df2 with df1. 
    # This overwrites common columns in df2 with df1's values where the index matches.
    df2_indexed.update(df1_indexed)

    # 4. Reset index to return 'match_col' to a normal column and save
    df_final = df2_indexed.reset_index()
    df_final.to_csv(output_path, index=False)
    
    return df_final

# Usage
# updated_df = sync_csv_files('file1.csv', 'file2.csv', match_col='b')

def plottable_format(input_csv, outfile_name):
    df = pd.read_csv(input_csv)
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    # rename the detector based on channel indicated
    df = df.rename(columns={"Pulse": "pulsewidth", "File": "filename_scope", "Detector":"detector", "ch1_area":"Area_Vs", "Dose (Gy)": "Dose"})
    #df = df.drop(columns=['Detector'])
    # df = df.drop(columns=['ch2_area'])
    # delete old detector column and replace with Si and LGAD based on channel column
    # append relevant area to area column and rename
    if "CH2" in input_csv:
        df['detector'] = "LGAD"
    elif "CH1" in input_csv:
        df['detector'] = "Si Diode"
    else:
        print("ERROR: unable to assign detector name, aborting")
        return
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    colnames = df.columns
    df = pd.DataFrame(np.repeat(df.values, 2, axis=0),columns=colnames)
    # computes charge for corresponding areas in nC
    charges_nc = (df['Area_Vs'] / R_OHM) * 1e9
    df["Q_nC"] = charges_nc
    # filters to desired beam and HV
    #df = df[df['Beam'] == beam_filter]
    #df = df[df['HV'] == HV_filter]
    new_hvs = []
    for HV in df["HV"]:
        new_hv = HV.replace("HV ", "")
        new_hvs.append(new_hv)
    df["HV"] = new_hvs
    print(df["Z"])
    print(df["Z"])
    df.to_csv(outfile_name)

def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)

    mask = (series < lower) | (series > upper)

    return series[mask], series[mask].index

def open_img(image):
    start = np.nan
    try:
        with Image.open(image) as img:
            img.resize((100,100))
            img.show()

            start = get_user_input("enter new start, or nothing if no changes needed: <nan> ",input_type=float)
            # Remember to close the image file when done with processing (optional for single frame images)

    except OSError as e:
        print(f"Error opening image: {e}")
    return start

def get_user_input(prompt, input_type=int, valid_value=None, default=np.nan):
    """
    Helper function to get user input
    """
    while True:
        try:
            user_input = input(prompt)
            # If the input is empty, use the default value
            if user_input == "":
                print('returning default value of', default)
                return default
            user_input = input_type(user_input)
            if valid_value is None or user_input in valid_value:
                return user_input
            else:
                print("Invalid input. Please enter a valid value: "+str(valid_value))
        except ValueError:
            print("Invalid input type. Please enter a valid value.")

def find_outliers_detector(filename, detector):
    df = pd.read_csv(filename)
    print(df.head())
    print(df.columns)
    df_filtered = df[df["detector"] == detector]
    groups = df_filtered.groupby(["Z","pulsewidth"])
    idx_set = set()
    for group, values in groups:
        print("values")
        print(values)
        outliers, indices = find_outliers_iqr(values["Q_nC"])
        zscores = stats.zscore(values["Q_nC"])
        indices_z = np.where(abs(zscores) > 1)[0]
        idx_set.update(indices)                    # IQR indices (label-based)
        idx_set.update(values.index[indices_z])    # Z-score indices (converted to labels)
    outlier_df = df_filtered.loc[list(idx_set)]
    print(outlier_df.head)
    filenames_only = outlier_df[["filename_scope","pulsewidth"]]
    filenames_only.to_csv(f"outliers_{detector}.csv", index=False)
    print("filenames saved successfully")

if __name__ == "__main__":
    #oct_plotter.combine_files("processed_BNL_2025-10-14_CH1.csv","processed_BNL_2025-10-15_CH1.csv","total_10_15_CH1.csv")
    #oct_plotter.combine_files("processed_BNL_2025-10-14_CH2.csv","processed_BNL_2025-10-15_CH2.csv","total_10_15_CH2.csv")
    # find_outliers_detector("/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/output_combined_oct_processed_data.csv", "LGAD")
    # filename = "outliers_LGAD.csv"
    sync_csv_files("isolated_CH2.csv","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/total_10_15_CH2_plottable.csv","filename_scope")
    '''filename = "cleaning.csv"
    detector = "LGAD"
    if detector == "LGAD":
        selected_channel = "CH2"
    else:
        selected_channel = "CH1"
    df = pd.read_csv(filename)
    df = df[(df["Z"] == 150) | (df["Z"] == 200)]
    print(df["HV"])
    new_starts = []
    for index, row in df.iterrows():
        item = row["filename_scope"]
        pulse = row["pulsewidth"]
        date = item[-19:-9]
        ifile = item[-8:-4]
        ifile = ifile.lstrip('0')
        print(date)
        print(item)
        if pulse > 0.5:
            graphicDir = f"new-plot-dose-{date}/all_files"
            corr_img = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/{graphicDir}/dose-calc-pulse={pulse}-{ifile}-{selected_channel}.jpg"
            start = open_img(corr_img)
        else:
            start = np.nan
        new_starts.append(start)
    df['new_start'] = new_starts
    df.to_csv("third_round_cleaning.csv")'''

