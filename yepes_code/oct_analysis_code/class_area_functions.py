import numpy as np
import matplotlib.pyplot as plt
import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import medfilt

# External dependencies (assumed to be in your path)
import oct_file_generator
import config
import plot_with_linear_reg_by_pulse

@dataclass
class SignalResult:
    """Container for processing results to avoid long, unindexed lists."""
    time: np.ndarray
    raw_signal: np.ndarray
    corrected_signal: np.ndarray
    interpolated_baseline: np.ndarray
    area: float
    indices: dict  # Stores start_idx, end_idx, shifted_start
    metadata: dict # Stores pulse, Z, HV, beam, file_path, date

class AreaDetector:
    def __init__(self, verbose=1, sigma=10):
        self.verbose = verbose
        self.sigma = sigma

    def _calculate_slope(self, x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)

    def backtrack_to_local_max(self, time, signal, idx, step=1):
        """Converted from recursion to a while loop to prevent recursion depth errors."""
        curr = idx
        while curr > 100:
            if (signal[curr] >= np.max(signal[curr:curr+100])) and \
               (signal[curr] > np.min(signal[curr-100:curr])):
                return curr
            curr -= step
        return 0

    def get_signal_bounds(self, time, signal, mode='old'):
        # 1. Smooth to remove high-frequency noise that creates "fake" peaks
        clean_step = medfilt(signal, kernel_size=101) 
        w = savgol_filter(clean_step, 51, 1)
        
        # 2. Calculate the "Drop Intensity" (Negative Gradient)
        # This turns every downward drop into a positive peak
        drop_intensity = -np.gradient(w)
        
        # 3. Find all significant drops
        # Adjust 'height' or 'prominence' based on your noise floor
        peak_indices, properties = find_peaks(drop_intensity, 
                                            height=np.std(drop_intensity)*2, 
                                            distance=100)
        
        if len(peak_indices) == 0:
            return 0, len(signal) - 1

        # 4. Logic to ignore the first drop:
        # If there are multiple drops, we usually want the LATEST one 
        # before the signal hits its absolute minimum.
        absolute_min_idx = np.argmin(w)
        valid_drops = [idx for idx in peak_indices if idx <= absolute_min_idx]
        
        if not valid_drops:
            main_drop_idx = peak_indices[0] # Fallback
        else:
            # Pick the LAST drop that occurs before the minimum 
            # This effectively skips the "pre-drop"
            main_drop_idx = valid_drops[-1]

        # 5. Expand outward from this specific drop to find the corners
        # Search backward for the "Shoulder"
        first_deriv = np.gradient(w)
        second_deriv = np.gradient(first_deriv)
        
        # Look back 300 points for the start of the curve
        start_search = max(0, main_drop_idx - 300)
        start_fluct = start_search + np.argmin(second_deriv[start_search:main_drop_idx])
        
        # Look forward for the "Toe" (recovery start)
        end_search = min(len(w)-1, main_drop_idx + 500)
        end_fluct = main_drop_idx + np.argmax(second_deriv[main_drop_idx:end_search])

        # 6. Safety Buffer
        start_fluct = max(0, start_fluct - 5)
        end_fluct = min(len(signal) - 1, end_fluct + 10)

        if self.verbose > 0:
            print(f"All drops found at: {peak_indices}")
            print(f"Selected Main Drop: {main_drop_idx}")
            
        return start_fluct, end_fluct

    def process_signal(self, file_path, params: dict, mode='old', channel="CH1") -> Optional[SignalResult]:
        """Main pipeline for processing a single file."""
        params['channel'] = channel
        time, signal = oct_file_generator.read_csv(file_path, params['channel'])
        clean_step = medfilt(signal, kernel_size=101) 
        w = savgol_filter(clean_step, 51, 1)
        if signal.size == 0 or len(time) == 0:
            return None
        
        try:
            # 1. Detection
            start_fluct, end_fluct = self.get_signal_bounds(time, signal, mode)
            time_step = time[1] - time[0]
            lookback = int(0.3e-6 / time_step)
            
            start_idx = max(0, start_fluct - lookback)
            
            # 2. Straight line fitting (logic from your oct_file_generator)
            signal_range = np.max(signal) - np.min(signal)
            end_idx, time, signal = oct_file_generator.straight_line_across(time, signal, start_idx, signal_range)

            # 3. Baseline Interpolation
            signal_removed = np.copy(signal)
            signal_removed[start_idx:end_idx] = np.nan
            
            # Fill NaNs for interpolation
            valid_mask = ~np.isnan(signal_removed)
            temp_baseline = np.interp(time, time[valid_mask], signal_removed[valid_mask])
            smoothed_baseline = gaussian_filter(temp_baseline, sigma=self.sigma)
            
            # Final area calculation
            corrected_signal = signal - smoothed_baseline
            pulse_area = abs(simpson(y=corrected_signal[start_idx:end_idx], x=time[start_idx:end_idx]))
        
        except Exception as e:
            print(f"Math error: {e}")
            return None

        return SignalResult(
            time=time,
            raw_signal=signal,
            corrected_signal=corrected_signal,
            interpolated_baseline=smoothed_baseline,
            area=pulse_area,
            indices={'start': start_idx, 'end': end_idx},
            metadata=params
        )

class SignalVisualizer:
    @staticmethod
    def plot_result(res: SignalResult, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(res.time, res.raw_signal, label="Original", color="blue", alpha=0.5)
        plt.plot(res.time, res.corrected_signal, label="Corrected", color="green")
        plt.plot(res.time, res.interpolated_baseline, label="Baseline", color="red", linestyle="--")
        
        plt.axvspan(res.time[res.indices['start']], res.time[res.indices['end']], 
                    color="yellow", alpha=0.3, label="Pulse Region")
        
        info_text = (f"Area: {res.area:.3e}\nPulse: {res.metadata['pulse']}\n"
                     f"HV: {res.metadata['HV']}\nChannel: {res.metadata['channel']}")
        
        plt.text(0.05, 0.05, info_text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        if config.showPlot:
            plt.show()
        plt.close()

# --- Example Usage ---
if __name__ == "__main__":
    detector = AreaDetector(verbose=config.verbose)
    
    date = "2025-10-15"
    Z = 20
    HV = 7
    pulse = 2.0
    fileno = "0775"
    filename = f"/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/data/{date}/scope-results-{date}-{fileno}.csv"

    file_params = {
        'pulse': pulse, 'Z': Z, 'HV': HV, 'date': date,
        'beam': "Electrons 85V", 'channel': "CH1", 'file_path': filename
    }

    result = detector.process_signal(filename, file_params)
    
    if result:
        print(f"Calculated Area: {result.area}")
        SignalVisualizer.plot_result(result,f"isolated_150_or_less/{fileno}.png")