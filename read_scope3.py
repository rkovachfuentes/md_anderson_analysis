import tkinter as tk
from tkinter import filedialog
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

def read_csv(file_path):
    """Reads the oscilloscope CSV file and extracts time and channel data."""
    time = []
    ch1 = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data_started = False
        for row in reader:
            if row and row[0] == "TIME":  # Start extracting data after header
                data_started = True
                continue
            if data_started and row:
                time.append(float(row[0]))
                ch1.append(float(row[1]))
    return np.array(time), np.array(ch1)

def update_plot(start_time=None, end_time=None):
    """Updates the plot based on input box or slider values."""
    global time, ch1, canvas, ax, span
    if start_time is None:
        start_time = float(entry_start_time.get())
    if end_time is None:
        end_time = float(entry_end_time.get())

    # Apply time mask
    mask = (time >= start_time) & (time <= end_time)
    only_positive_times = ((time >= 0) & (time <= end_time))

    # Clear the previous plot and re-plot the data
    ax.clear()
    ax.plot(time[mask], -1*ch1[mask], label="CH1")
    ax.axhline(-1*np.mean(ch1[mask]), color='g', linestyle='--', label="Baseline: mean of all data")
    ax.axhline(-1*np.mean(ch1[only_positive_times]), color='y', linestyle='--', label="Baseline: mean, excluding negative times")
    ax.plot(time[only_positive_times],np.abs(ch1[only_positive_times]-np.mean(ch1[only_positive_times])),color='b',linestyle='--',label="Subtracting background")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Oscilloscope Data ({start_time} s to {end_time} s)")
    ax.legend()
    ax.grid()

    # Re-enable SpanSelector for updated plot
    span = SpanSelector(ax, on_select, 'horizontal', useblit=True, interactive=True)

    # Redraw the canvas
    canvas.draw()

def on_select(xmin, xmax):
    """Handles time range selection via cursor drag."""
    entry_start_time.delete(0, tk.END)
    entry_end_time.delete(0, tk.END)
    entry_start_time.insert(0, str(xmin))
    entry_end_time.insert(0, str(xmax))
    update_plot(xmin, xmax)

def open_file():
    """Opens a file dialog to select a file and initializes the plot."""
    global loaded_file_path, time, ch1, slider_start, slider_end
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_path:
        return

    try:
        loaded_file_path = file_path
        time, ch1 = read_csv(file_path)

        # Update sliders and input boxes with file's time range
        slider_start.config(from_=time[0], to=time[-1])
        slider_end.config(from_=time[0], to=time[-1])
        slider_start.set(time[0])
        slider_end.set(time[-1])
        entry_start_time.delete(0, tk.END)
        entry_end_time.delete(0, tk.END)
        entry_start_time.insert(0, str(time[0]))
        entry_end_time.insert(0, str(time[-1]))
        label_status.config(text="File loaded successfully!")

        # Plot the full data initially
        update_plot(time[0], time[-1])
    except Exception as e:
        label_status.config(text=f"Error loading file: {e}")

def update_from_sliders():
    """Updates the plot based on slider values."""
    start_time = slider_start.get()
    end_time = slider_end.get()
    entry_start_time.delete(0, tk.END)
    entry_end_time.delete(0, tk.END)
    entry_start_time.insert(0, str(start_time))
    entry_end_time.insert(0, str(end_time))
    update_plot(start_time, end_time)

# Initialize global variables
loaded_file_path = "/Users/rkfuentes/Documents/md_anderson_analysis/tek0124CH1.csv"
time = np.array([])
ch1 = np.array([])

# Create GUI window
window = tk.Tk()
window.title("Oscilloscope Data Viewer")

frame_plot = tk.Frame(window)
frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_controls = tk.Frame(window)
frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

# Matplotlib plot embedding
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# File selection button
btn_open_file = tk.Button(frame_controls, text="Select CSV File", command=open_file)
btn_open_file.pack(pady=10)

# Input boxes for start and end time
label_start_time = tk.Label(frame_controls, text="Start Time (Input Box):")
label_start_time.pack()
entry_start_time = tk.Entry(frame_controls)
entry_start_time.pack()

label_end_time = tk.Label(frame_controls, text="End Time (Input Box):")
label_end_time.pack()
entry_end_time = tk.Entry(frame_controls)
entry_end_time.pack()

# Sliders for time selection
label_slider_start = tk.Label(frame_controls, text="Start Time (Slider):")
label_slider_start.pack()
slider_start = tk.Scale(frame_controls, orient=tk.HORIZONTAL, resolution=1e-9, command=lambda _: update_from_sliders())
slider_start.pack()

label_slider_end = tk.Label(frame_controls, text="End Time (Slider):")
label_slider_end.pack()
slider_end = tk.Scale(frame_controls, orient=tk.HORIZONTAL, resolution=1e-9, command=lambda _: update_from_sliders())
slider_end.pack()

btn_plot_cursor = tk.Button(frame_controls, text="Select Time Range on Plot", command=lambda: SpanSelector(ax, on_select, 'horizontal', useblit=True, interactive=True))
btn_plot_cursor.pack(pady=10)

label_status = tk.Label(frame_controls, text="")
label_status.pack()

open_file()

# Run GUI loop
window.mainloop()
