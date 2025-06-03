#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.integrate import simpson  # Use `simpson` instead of `simps`
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

verbose=1
savePlot=True
showPlot=False

#==================================================================================================
#
#==================================================================================================
def read_csv(file_path, selected_channel="CH1"):
    """Reads the oscilloscope CSV file and extracts time and channel data."""
    time = []
    ch1 = []
    ch2 = []
    num_columns = 0
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data_started = False
            for row in reader:
                if row and row[0] == "TIME":
                    data_started = True
                    continue
                if data_started and row:
                    try:
                        num_columns = len(row)
                        time.append(float(row[0]))
                        ch1.append(float(row[1]))
                        if num_columns == 3:
                            ch2.append(float(row[2]))  # Read third column if it exists
                        elif num_columns == 2:
                            ch2.append(float(0))
                    except ValueError:
                        print(f"Skipping invalid data row: {row}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return np.array([]), np.array([])

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([]), np.array([])

    if num_columns == 2:
        return np.array(time), np.array(ch1)
    else:
        #return np.array(time), np.array(ch1) if selected_channel == "CH1" else np.array(time), np.array(ch2)
        if selected_channel == "CH1":
            #print("selected channel CH1")
            return np.array(time), np.array(ch1) 
        else:
            return np.array(time), np.array(ch2)
 
#==================================================================================================
#
#==================================================================================================
#def get_area(file_path, pulse=2, Z=60, X=0, ifile=1560 ):
def get_area(file_path, pulse=2, Z=60, HV=0, beam="Electrons 85V", ifile=1560 ):

   time, signal = read_csv(file_path)
   # Determine the number of points corresponding to 0.1 microseconds
   time_step = time[1] - time[0]  # Time difference between consecutive points
   lookback_points = int(0.3e-6 / time_step)  # Number of points in 0.1 microseconds

   # Initialize variables to track the largest increase
   largest_increase = -np.inf  # Start with the smallest possible number
   start_index = None
   end_index = None

   signal_min = np.min(signal)
   signal_max = np.max(signal)
   signal_range = signal_max-signal_min
   shifted_min  = signal_min+0.8*signal_range

   threshold = 0.05 * signal_range
# Find indices where fluctuation exceeds the threshold
   fluctuation_indices = []
   for i in range(1, len(signal)):
       if abs(signal[i] - signal[i-1]) > threshold:
           fluctuation_indices.append(i)

# Identify the start of fluctuations
   if fluctuation_indices:
      start_of_fluctuations = fluctuation_indices[0]
      print(f"Signal starts fluctuating beyond 5% of max range at index {start_of_fluctuations}")
   else:
      print("No significant fluctuation found.")
      start_of_fluctuations = 0

   first_good_bin = start_of_fluctuations+int(5e-7/time_step)
   first_good_bin = 1
   print("first_good_bin ", first_good_bin)

   sigma = 10  # Standard deviation of the Gaussian kernel
   # Loop backwards through the array
   for i in range(len(signal) - 1, lookback_points - 1, -1):
       # Calculate the signal difference over the lookback window
       if signal[i] > shifted_min: continue
       if i < first_good_bin: continue
       signal_difference = signal[max(0,i - lookback_points)] - signal[i]

       #print("signal_difference ", signal_difference, " time ", time[i]," signal ", signal[i]) 
       #if time[i] < 0.0: continue
       # Update if the current difference is the largest increase
       if signal_difference > largest_increase:
           #print("**************** largest increase yet ")
          largest_increase = signal_difference
          start_index = i - lookback_points
          end_index = i

   # Output the results
   if verbose>1:
      print(f"Largest increase: {largest_increase:.2f}")
      print(f"Start index: {start_index}, Time: {time[start_index]:.6e} s signal {signal[start_index]}")
      print(f"End index: {end_index}, Time: {time[end_index]:.6e} s signal {signal[end_index]}")


   # Shift the starting point by 0.1 microseconds
   shift_points = int(0.2e-6 / time_step)  # Number of points to shift earlier
   shifted_start_index = max(0, start_index - shift_points)  # Ensure index is not negative

   # Determine the region to remove
   remove_start_index = shifted_start_index
   remove_end_index = min(len(time) - 1, shifted_start_index + 3 * shift_points + int(pulse * 1e-6 / time_step))

   # Remove the area by setting the signal to NaN in this region
   signal_removed = np.copy(signal)
   signal_removed[remove_start_index:remove_end_index + 1] = np.nan
   #if pulse>2:
   #   signal_removed[remove_start_index:len(signal)] = np.nan

   # Interpolate the missing values
   valid_mask = ~np.isnan(signal_removed)  # Mask to keep valid values
   temp_interpolated_baseline = np.interp(time, time[valid_mask], signal_removed[valid_mask])

   baseline_smoothed = gaussian_filter(temp_interpolated_baseline, sigma=sigma)

   #  Interpolate the missing values after smoothing
   interpolated_baseline = np.interp(time, time[valid_mask], baseline_smoothed[valid_mask])

   corrected_signal = signal-interpolated_baseline

   # Calculate the area under the interpolated curve in the removed region
   removed_time = time[remove_start_index:remove_end_index + 1]  # Time in the removed region
   corrected_signal_removed_values = corrected_signal[remove_start_index:remove_end_index + 1]
   signal_removed_values = signal[remove_start_index:remove_end_index + 1]
   signal_area   = abs(simpson(y=corrected_signal_removed_values, x=removed_time))

   bins_around=30
   baseline_offset=0
   ch1=corrected_signal
   selected_ch1=corrected_signal_removed_values
   #dynamic_baseline = np.array([
   #         np.mean(selected_ch1[max(0, i - bins_around):min(len(selected_ch1), i + bins_around)]) - baseline_offset
   #         for i in selected_ch1 
   #     ])
   dynamic_baseline = np.array([
       np.mean(selected_ch1[max(0, idx - bins_around):min(len(selected_ch1), idx + bins_around)]) - baseline_offset
        for idx in range(len(selected_ch1))
   ])
   

   selected_ch1 = np.nan_to_num(selected_ch1, nan=0.0, posinf=0.0, neginf=0.0)
   dynamic_baseline = np.nan_to_num(dynamic_baseline, nan=0.0, posinf=0.0, neginf=0.0)
        # Find peaks below the dynamic baseline
   inverted_ch1 = -(selected_ch1 - dynamic_baseline)
   peaks, _ = find_peaks(inverted_ch1, height=0)
   peak_times = removed_time[peaks]
   peak_values = selected_ch1[peaks]
   nPeaks = len(peaks)




   # Output the results
   if verbose>1:
      print(f"Shifted start index: {shifted_start_index}, Time: {time[shifted_start_index]:.6e} s")
      print(f"Removed area start time: {time[remove_start_index]:.6e} s")
      print(f"Removed area end time: {time[remove_end_index]:.6e} s")
      print(f"Area under interpolated curve in signal region: {signal_area:.3e} ")

   # Visualization
   if savePlot:
      plt.figure(figsize=(10, 6))
      plt.plot(time, signal, label="Original Signal", color="blue")
      plt.plot(time, corrected_signal, label="Corrected Signal", color="green")
      plt.plot(time, interpolated_baseline, label="Interpolated Baseline", color="red")
      plt.axvspan(time[remove_start_index], time[remove_end_index], color="yellow", alpha=0.3, label="Removed Region")
      plt.axvline(time[shifted_start_index], color="green", linestyle="--", label="Shifted Start")
      plt.scatter(peak_times, peak_values, color='purple', marker='x', label="Peaks in Signal Zone")
      plt.xlabel("Time (s)")
      plt.ylabel("Signal")
      plt.title("Signal Suppression and Interpolation")
      plt.text(
         0.6 * max(time),  # X-coordinate (adjust based on your plot range)
         min(signal)+0.05 * (max(signal)-min(signal)),  # Y-coordinate (adjust based on your plot range)
         f"Signal Area: {signal_area:.3e}",  # Text for the label
         fontsize=12,
         #color="purple",
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple')  # Optional styling
      )
      
      plt.legend()
      plt.grid()

      graphicDir=f"plot-dose-2025-05-05/Beam={beam}-Z={Z}-HV={HV}"
      if not os.path.exists(graphicDir):
         os.makedirs(graphicDir)
         print(f"Directory created: {graphicDir}")
      else:
         print(f"Directory already exists: {graphicDir}")
      
      graphicFile=f"{graphicDir}/dose-calc-pulse={pulse}-{ifile}.jpg"
      if verbose>2: print("graphicFile ", graphicFile)
      plt.savefig(graphicFile, format="jpeg", dpi=300)  
      if showPlot:
         plt.show()
      plt.close()


   return signal_area, nPeaks
#==============================================================================
log_file = "lgad-2025-05-06-analysis.csv"
if not os.path.exists(log_file):
    print(f"log file {log_file} not found")
    exit()

log_df = pd.read_csv(log_file)

Detector="BNL"
Channel=16
Beam="Electrons 110V"
Beam="Electrons 85V"
X=0
Z=135
Z=35
Z=-26
Z=60
Shield="No"
HV=30
LV="5.7"

verbose=1

pulses=[1]
pulses=[0.5,1,2,3]

mean_signal_areas=[]
mean_signal_peaks=[]
doses=[]


print(f"Detector {Detector}")
print(f"Beam     {Beam}")
print(f"X        {X}")
print(f"Z        {Z}")
print(f"HV       {HV}")
print(f"LV       {LV}")
 

for pulse in pulses:
    matching_rows = log_df[
        (log_df["Detector"] == Detector) &
        (log_df["Channel"] == Channel) &
        (log_df["Beam"] == Beam) &
        (log_df["X"] == X) &
        (log_df["Z"] == Z) &
        (log_df["Shield"] == Shield) & 
        (log_df["HV"] == HV) & 
        (log_df["LV"] == LV) & 
     #  (log_df["Comment"] == Comment) & 
        (log_df["Pulse"] == pulse)
    ]
    if verbose>0:
       print("matching rows ", matching_rows[["Detector","Dose", "Pulse","X","Z","HV","LV"]])
       #continue
    for _, row in matching_rows.iterrows():
        file_min = int(row["FileMin"].split("-")[-1].replace(".csv", ""))
        file_max = int(row["FileMax"].split("-")[-1].replace(".csv", ""))

        dose = row['Dose']
        signal_areas = []
        signal_peaks = []
        for i in range(file_min, file_max + 1):
            #if not i== 1422: continue
            #file_path = re.sub(r'\d{4}(?=\.csv)', str(i), f{row["FileMin"])
            file_path = re.sub(r'\d{4}(?=\.csv)', f"{i:04}", row["FileMin"])
            file_path = file_path.replace("/home/pyepes/data/","/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/")
            file_path = file_path.replace("2025-05-06","2025-05-05")
            print("NEW FILE PATH")
            print(file_path)

            if not os.path.exists(file_path):
               print("{file_path} does not exist, skip.")
               continue
            
            #signal_area, signal_peak=get_area(file_path, pulse=pulse,Z=Z,X=X, ifile=i)
            signal_area, signal_peak=get_area(file_path, pulse=pulse,Z=Z,HV=HV,beam=Beam,ifile=i)
            signal_areas.append(signal_area)
            signal_peaks.append(signal_peak)
            if verbose>-1:
               print(f"i {i} pulse {pulse} signal_area {signal_area}", )

        mean_signal_area = np.mean(signal_areas)
        mean_signal_peak = np.mean(signal_peaks)
        if verbose>1:
           print("signal_areas ", signal_areas)
           print(f"Pulse {pulse} Dose {dose} mean signal_area {mean_signal_area} ")
        mean_signal_areas.append(mean_signal_area)
        mean_signal_peaks.append(mean_signal_peak)
        doses.append(dose)

        matching_columns = ["Detector", "Channel", "Beam", "Z", "X", "HV", "LV", "Pulse"]
    # Find rows that match on all specified columns
        row['area']=mean_signal_area
        row['peaks']=mean_signal_peak

        row_copy = row.copy()  # Prevents modification issues
        print("row ", row_copy[matching_columns])

        df_row = pd.DataFrame([row_copy])
        
        mask = (log_df[matching_columns] == df_row[matching_columns].iloc[0]).all(axis=1)

        if mask.any():
        # Remove the matching rows before appending the new row
           #print("remove and add ")
           #print("df_row ", df_row)
           log_df = log_df[~mask]
           log_df = pd.concat([log_df, df_row], ignore_index=True)
           log_df.to_csv(log_file, index=False)
        else:
           #print("row not found")
           row_copy.to_csv(log_file, index=False)


print("mean_signal_areas ", mean_signal_areas," doses ", doses)
print("mean_signal_peaks ", mean_signal_peaks," doses ", doses)
plt.figure(figsize=(10, 6))
plt.plot(doses, mean_signal_areas, label="Dose vs Area", color="blue", alpha=0.6)
#plt.plot(doses, mean_signal_peaks, label="Dose vs Peaks", color="blue", alpha=0.6)
plt.xlabel("Dose (1e-8 C)")
plt.ylabel("Area")
plt.title(f"Dose vs Area")
plt.legend()
plt.grid()
plt.show()



