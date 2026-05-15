import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import re

# ===============================================================
# 1. REFERENCE DOSE MODEL (MATHEMATICAL INPUT)
# ===============================================================
# Reference dose as a function of distance:
#
#   D_ref(z)
#
# measured at:
#   - Voltage V = 85 V
#   - Pulse width PW = 1.0 µs
#
# Known only at discrete distances z_j.

distance = np.array([0.11, 0.36, 0.61, 1.51, 2.51])  # meters
base_dose = np.array([9.5e-03, 2.9e-03, 1.2e-03, 1.4e-04, 3.8e-05])

# Supported pulse widths (exact values used in the experiment)
pw_values = np.array([0.5, 1.0, 2.0, 3.0])

# Scaling factors S(V, PW)
# Mathematical model:
#
#   D(z, V, PW) = S(V, PW) · D_ref(z)
#
scale_factors = {
    85:  np.array([1.0, 1.3, 2.2, 3.1]),
    110: np.array([15.0, 19.5, 33.0, 46.5]),
    191: np.array([38.5, 52.3, 95.3, 143.8]),
}

# ===============================================================
# 2. BUILD CONTINUOUS DOSE FUNCTIONS (INTERPOLATION)
# ===============================================================
# For each voltage V and pulse width PW we build:
#
#   D̂_{V,PW}(z) = Interp[ S(V,PW) · D_ref(z) ]
#
# This yields a continuous function of z.

interpolators = {}

for V, factors in scale_factors.items():
    interpolators[V] = {}
    for pw, sf in zip(pw_values, factors):
        interpolators[V][pw] = interp1d(
            distance,
            base_dose * sf,
            kind="linear",
            fill_value="extrapolate"
        )

# ===============================================================
# 3. ROBUST VOLTAGE EXTRACTION FROM BEAM STRING
# ===============================================================
# Voltage V is encoded inside the Beam string.
# We extract V ∈ {85, 110, 191}.

def extract_voltage(beam_value):
    if pd.isna(beam_value):
        return np.nan

    s = str(beam_value).replace(" ", "").lower()

    # Prefer explicit pattern: number followed by 'v'
    m = re.search(r"(\d+)\s*v", s)
    if m:
        val = int(m.group(1))
        if val in [85, 110, 191]:
            return val

    # Fallback: manual search
    for v in [85, 110, 191]:
        if str(v) in s:
            return v

    return np.nan

# ===============================================================
# 4. DOSE EVALUATION FUNCTION (NO PULSE DISCRETIZATION)
# ===============================================================
# For a single measurement (row):
#
# Inputs:
#   z_i  = distance
#   PW_i = pulse width  (assumed exact: 0.5, 1.0, 2.0, 3.0)
#   V_i  = voltage
#
# Dose is computed as:
#
#   D_i = D̂_{V_i, PW_i}(z_i)

def compute_dose(z, pw, V):
    V = "".join(char for char in V if char.isdigit())
    V = int(V)
    # Dose undefined if any input is missing
    if pd.isna(z) or pd.isna(pw) or pd.isna(V):
        return np.nan

    # Direct lookup: pulse width is already exact
    f = interpolators[V][pw]

    return float(f(z))

def file_dose_converter(input_file, output_file):
    df = pd.read_csv(input_file)
    doses = []
    for _, row in df.iterrows():
        beam = row['Beam']
        dist_m = float(row['Z'])/100 # convert from cm to m
        pulse_width = float(row['pulsewidth'])
        converted_dose = compute_dose(dist_m, pulse_width, beam)
        doses.append(converted_dose)
    df['Dose'] = doses
    df.to_csv(output_file)
    print("doses converted successfully")

if __name__ == "__main__":
    file_dose_converter("total_10_15_CH1_plottable.csv","total_10_15_CH1_alternate_dose.csv")
    file_dose_converter("total_10_15_CH2_plottable.csv","total_10_15_CH2_alternate_dose.csv")