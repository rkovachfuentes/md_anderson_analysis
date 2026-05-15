import tqdm
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from class_area_functions import AreaDetector
from class_area_functions import SignalVisualizer
import dose_conversion

class ExperimentGenerator:
    def __init__(self, log_file, detector_type, date, output_path, channel):
        self.log_df = pd.read_csv(log_file)
        self.detector_type = detector_type
        self.date = date
        self.output_path = output_path
        self.results = []
        self.channel = channel

    def _generate_file_paths(self, row):
        """Extracts the file range and constructs absolute paths."""
        # Regex to find the 4-digit file number (e.g., 0775)
        match_min = re.search(r'(\d{4})\.csv', row["FileMin"])
        match_max = re.search(r'(\d{4})\.csv', row["FileMax"])
        
        if not match_min or not match_max:
            print("exiting with nothing")
            return []
            
        f_min = int(match_min.group(1))
        f_max = int(match_max.group(1))
        
        paths = []
        local_base_dir = "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data"

        for i in range(f_min, f_max + 1):
            # 1. Get just the filename (e.g., scope-results-2025-10-14-0775.csv)
            # Use os.path.basename to strip all the /home/lgad/data/ garbage
            filename = os.path.basename(row["FileMin"])
            
            # 2. Update the number in the filename
            current_filename = filename.replace(f"{f_min:04}", f"{i:04}")
            
            # 3. Join it to your ACTUAL local path structure
            # Result: /Users/.../data/2025-10-14/scope-results-2025-10-14-0775.csv
            full_path = os.path.join(local_base_dir, self.date, current_filename)
            
            paths.append((i, full_path))
        return paths

    def run(self, area_detector_instance):
        """Iterates through logs and processes signals with a progress bar."""
        matching = self.log_df[self.log_df["Detector"] == self.detector_type]
        # additional filtering for only desired Z values
        max_z = 150
        min_pulse = 1.0
        HV = 100
        print(self.log_df.columns)
        # Convert column to numeric, turning errors (like 'abc') into NaN
        self.log_df['Z'] = pd.to_numeric(self.log_df['Z'], errors='coerce')
        # Drop rows where the column is now NaN
        self.log_df = self.log_df.dropna(subset=['Z'])
        matching = self.log_df[(self.log_df["Z"].astype(float) >= max_z) & (self.log_df['Pulse'].astype(float) >= min_pulse) & (self.log_df['Comment'] == 'HV 100')]
        print(f"Found {len(matching)} rows matching detector: {self.detector_type}")
        
        # We wrap the dataframe iteration in tqdm for a visual progress bar
        print(f"Starting batch processing for {self.detector_type}...")
        for _, row in tqdm.tqdm(matching.iterrows(), total=matching.shape[0], desc="Log Rows"):
            file_info = self._generate_file_paths(row)
            
            for index, path in file_info:
                if not os.path.exists(path):
                    print(f"DEBUG: File not found at {os.path.abspath(path)}")
                    continue

                params = {
                    'pulse': row['Pulse'], 
                    'Z': row['Z'], 
                    'HV': row['Comment'],
                    'beam': row['Beam'], 
                    'channel': row['Channel'], # Or row['Channel'] if dynamic
                    'date': self.date,
                    'file_index': index,
                    'file_path': str(path)
                }

                # Use the logic from our AreaDetector class
                result = area_detector_instance.process_signal(path, params, channel=CHANNEL)
                if result:
                    SignalVisualizer.plot_result(result,f"isolated_150_or_less/{index}.png")
                
                if result:
                    self.results.append(self._format_row(result))

        self.save()
        print(f"Processing complete. Results saved to {self.output_path}")

    def _format_row(self, res):
        """Converts the SignalResult dataclass into a flat dictionary for CSV."""
        meta = res.metadata if res.metadata is not None else {}
        return {
            'Detector': self.detector_type,
            'Beam': res.metadata['beam'],
            'Pulse': res.metadata['pulse'],
            'HV': res.metadata['HV'],
            'Z': res.metadata['Z'],
            'File': str(meta.get('file_path', 'unknown')),
            'Area_Vs': res.area,
            'Timestamp': self.date
        }

    def save(self):
        if self.results:
            lengths = [len(r) for r in self.results]
        if len(set(lengths)) > 1:
            print("Warning: Inconsistent dictionary lengths found in results!")
        pd.DataFrame(self.results).to_csv(self.output_path, index=False)

if __name__ == "__main__":
    # --- Configuration ---
    DATE = "2025-10-14"
    SENSOR = "BNL"
    LOG_PATH = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/log_files/lgad-{DATE}-log.csv"
    CHANNEL = "CH2"
    OUTPUT_PATH = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/oct_analysis_code/isolated_150_{SENSOR}_{DATE}_{CHANNEL}.csv"

    # --- Initialization ---
    # 1. Initialize the math engine (math from previous steps)
    detector_engine = AreaDetector(verbose=0) 
    
    # 2. Initialize the file/log manager
    experiment = ExperimentGenerator(
        log_file=LOG_PATH, 
        detector_type=SENSOR, 
        date=DATE, 
        output_path=OUTPUT_PATH,
        channel = CHANNEL
    )

    # --- Execution ---
    experiment.run(detector_engine)

    # 3. (Optional) Post-Process Doses