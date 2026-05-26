import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load CSV file
file_path = "lgad-2025-05-06-analysis.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# User-defined filters
shield = "No"
X = 0
Detector = "BNL"
Channel = 16
Dose="Abs_Dose"

# Apply filtering
filtered_df = df[
    (df["Detector"] == Detector) &
    (df["Channel"] == Channel) &
    (df["X"] == 0) &
    (df["area"].notna()) & (df["area"] != "") &
    ~((df["Pulse"] == 3)) &
    ~((df["HV"] == 0)) &
    ~((df["Z"] == 60)) &
    ~((df["Beam"] == "Electrons 110V") & (df["Z"] < 10))
]

print("Filtered Data:\n", filtered_df[['Beam', 'Pulse', 'HV', 'Z', 'Dose', 'Abs_Dose', 'area']])

# Group data by Z
grouped_df = filtered_df.groupby("Z")

# Plot setup
plt.figure(figsize=(10, 8))
variable = "area"

for z_value, group in grouped_df:
    # Sort the group by "Dose"
    group = group.sort_values(by=Dose)
# Scale Dose by 1.e6
    group[Dose] = group[Dose] * 1.e6
    beam_label = group["Beam"].iloc[0] 
    hv_label = group["HV"].iloc[0] 
    

    # Fit a linear regression **forcing the line through (0,0)**
    slope, intercept, _, _, _ = linregress(group[Dose], group[variable])

    # Adjust the y-values by subtracting the intercept
    adjusted_y = group[variable] - intercept

    # Calculate 10% errors
    x_err = group[Dose] * 0.1
    y_err = abs(adjusted_y * 0.1)

    # Plot with error bars
    plt.errorbar(group[Dose], adjusted_y,
                 xerr=x_err, yerr=y_err,
                 label=f"{beam_label},Z={z_value} cm ,HV={hv_label} V", linestyle='-',
                 marker='o', alpha=0.6)

    # Plot the fitted line through (0,0)
    x_fit = np.linspace(0, max(group[Dose]), 100)
    y_fit = slope * x_fit  # Since we force the line through (0,0), intercept is ignored
    #plt.plot(x_fit, y_fit, '--', label=f"Fit Z={z_value}")
    plt.plot(x_fit, y_fit, '--')

# Add labels, title, legend, and grid
plt.xlabel(f"{Dose} Gy/pulse x 1.e6")
plt.ylabel(f"{variable}")
plt.title(f"{Dose} vs. {variable}")
plt.legend()
plt.grid()
plt.show()
