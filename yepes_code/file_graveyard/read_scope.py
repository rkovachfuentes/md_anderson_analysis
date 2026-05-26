
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import Listbox, Scrollbar, MULTIPLE, EXTENDED
from tkinter import font
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numbers
import numpy as np
import pandas as pd
import os
import re
from scipy.signal import find_peaks  # Import the function for peak detection
from matplotlib.widgets import SpanSelector


class ScopeViewApp(ttk.Frame):  # Changed base class to ttk.Frame
    def __init__(self, parent, master=None, **kwargs):
        super().__init__(master, **kwargs)  # Corrected superclass initialization
        self.parent = parent
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)  # Add this line

        self.loaded_file_path = None
        self.time = np.array([])
        self.ch1 = np.array([])
        self.plots = {}
        self.ax = None
        self.span_selector = None
        self.bins_around = 25
        self.create_widgets()

    def create_widgets(self):
        # Matplotlib plot embedding
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.plot_titles=[]


     
        canvas_frame = ttk.Frame(self)
        #canvas_frame.pack(side=tk.TOP, fill=tk.X)
        canvas_frame.pack(side=tk.TOP)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)  # Use 'self' as master
        self.canvas_widget = self.canvas.get_tk_widget()
        #self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_widget.pack(side=tk.LEFT)

        #self.file_info_text = tk.Text(canvas_frame, wrap=tk.WORD)
  
        # Control Frame
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_info_text = tk.Text(control_frame, wrap=tk.WORD,width=25)
        #self.file_info_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_info_text.pack(side=tk.LEFT, fill=tk.Y)
        self.file_info_text.insert(tk.END, f"Info\n")


        self.frame_files = ttk.Frame(control_frame)
        self.frame_files.pack(pady=5)
        # File selection button
        btn_open_file = ttk.Button(self.frame_files, text="Select CSV Files", command=self.open_file)
        btn_open_file.pack(side=tk.LEFT, pady=10)

        # Dropdown for plot selection
        self.plot_selector = ttk.Combobox(self.frame_files, values=[], state="readonly")
        self.plot_selector.pack(side=tk.LEFT, padx=10, pady=10)
        self.plot_selector.bind("<<ComboboxSelected>>", lambda event: self.update_plot(self.plot_selector.get()))

        # Start time controls
        default_value = 0
        self.frame_time = ttk.Frame(control_frame)
        self.frame_time.pack(pady=5)

        start_time_label = ttk.Label(self.frame_time, text="Start Time:")
        start_time_label.pack(side=tk.LEFT)
        self.slider_start = tk.Scale(self.frame_time, from_=0, to=100, orient=tk.HORIZONTAL, resolution=1.e-9, showvalue=0,
                                        command=self.update_from_sliders)
        self.slider_start.set(default_value)
        self.slider_start.pack(side=tk.LEFT)

        self.entry_start_time = ttk.Entry(self.frame_time, width=10)
        self.entry_start_time.insert(0, str(default_value))
        self.entry_start_time.pack(side=tk.LEFT, padx=(0, 20))
        self.entry_start_time.bind("<Return>", self.entry_start_time_changed)

        default_value = 1
        end_time_label = ttk.Label(self.frame_time, text="End Time:")
        end_time_label.pack(side=tk.LEFT)
        self.slider_end = tk.Scale(self.frame_time, from_=0, to=100, orient=tk.HORIZONTAL, resolution=1.e-9, showvalue=0,
                                    command=self.update_from_sliders)
        self.slider_end.set(default_value)
        self.slider_end.pack(side=tk.LEFT)
        self.entry_end_time = ttk.Entry(self.frame_time, width=10)
        self.entry_end_time.insert(0, str(default_value))
        self.entry_end_time.pack(side=tk.LEFT)
        self.entry_end_time.bind("<Return>", self.entry_end_time_changed)

        self.reset_plot_button = ttk.Button(self.frame_time, text="Reset", command=self.reset_plot)
        self.reset_plot_button.pack(side=tk.LEFT, padx=10)

        self.frame_base = ttk.Frame(control_frame)
        self.frame_base.pack(pady=5)
        label_baseline_offset = ttk.Label(self.frame_base, text="Baseline - Offset:")
        label_baseline_offset.pack(side=tk.LEFT)
        self.entry_baseline_offset = ttk.Entry(self.frame_base, width=5)
        self.entry_baseline_offset.insert(0, "0.1")
        self.entry_baseline_offset.pack(side=tk.LEFT, padx=(0, 20))
        self.entry_baseline_offset.bind("<Return>", self.entry_baseline_changed)

        label_baseline_bins_around = ttk.Label(self.frame_base, text="bins around:")
        label_baseline_bins_around.pack(side=tk.LEFT)
        self.entry_baseline_bins_around = ttk.Entry(self.frame_base, width=5)
        self.entry_baseline_bins_around.insert(0, str(self.bins_around))
        self.entry_baseline_bins_around.pack(side=tk.LEFT)
        self.entry_baseline_bins_around.bind("<Return>", self.entry_baseline_changed)

        self.btn_plot_cursor = ttk.Button(control_frame, text="Select Time Range on Plot",
                                            command=lambda: self.on_select(self.ax, self.on_select))
        self.btn_plot_cursor.pack(pady=10)

        self.label_status = ttk.Label(control_frame, text="")
        self.label_status.pack()


    def create_widgets1(self):
           # Create a frame for the canvas and text widget
           canvas_frame = ttk.Frame(self)
           canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

           # Configure grid layout for canvas_frame
           canvas_frame.columnconfigure(0, weight=9)  # Canvas column gets 3x the horizontal space
           canvas_frame.columnconfigure(1, weight=1)  # Text widget column gets less space
           canvas_frame.rowconfigure(0, weight=1)     # Allow vertical expansion

           # Canvas for Matplotlib (left side)
           self.fig = plt.figure(figsize=(10, 6))
           self.ax = self.fig.add_subplot(1, 1, 1)
           self.plot_titles = []

           self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
           self.canvas_widget = self.canvas.get_tk_widget()
           self.canvas_widget.grid(row=0, column=0, sticky="nsew")  # Expand in all directions

           # Text widget (right side)
           self.file_info_text = tk.Text(canvas_frame, wrap=tk.WORD)
           self.file_info_text.grid(row=0, column=1, sticky="nsew")  # Expand vertically and horizontally as needed
           # Example text content
           self.file_info_text.insert(tk.END, "This is a test\n")

           # Control frame below the canvas_frame
           control_frame = ttk.Frame(self)
           control_frame.pack(side=tk.BOTTOM, fill=tk.X)

           # Control frame below the canvas_frame
           control_frame = ttk.Frame(self)
           control_frame.pack(side=tk.BOTTOM, fill=tk.X)

           # Frame for file selection and plot options
           self.frame_files = ttk.Frame(control_frame)
           self.frame_files.pack(pady=5)

           # File selection button
           btn_open_file = ttk.Button(self.frame_files, text="Select CSV Files", command=self.open_file)
           btn_open_file.pack(side=tk.LEFT, pady=10)

           # Dropdown for plot selection
           self.plot_selector = ttk.Combobox(self.frame_files, values=[], state="readonly")
           self.plot_selector.pack(side=tk.LEFT, padx=10, pady=10)
           self.plot_selector.bind("<<ComboboxSelected>>", lambda event: self.update_plot(self.plot_selector.get()))

           # Frame for time controls
           self.frame_time = ttk.Frame(control_frame)
           self.frame_time.pack(pady=5)

           # Start time label and slider
           start_time_label = ttk.Label(self.frame_time, text="Start Time:")
           start_time_label.pack(side=tk.LEFT)
           self.slider_start = tk.Scale(self.frame_time, from_=0, to=100, orient=tk.HORIZONTAL, resolution=1.e-9, showvalue=0, command=self.update_from_sliders)
           self.slider_start.set(0)
           self.slider_start.pack(side=tk.LEFT)

           self.entry_start_time = ttk.Entry(self.frame_time, width=10)
           self.entry_start_time.insert(0, str(0))
           self.entry_start_time.pack(side=tk.LEFT, padx=(0, 20))
           self.entry_start_time.bind("<Return>", self.entry_start_time_changed)

           # End time label and slider
           end_time_label = ttk.Label(self.frame_time, text="End Time:")
           end_time_label.pack(side=tk.LEFT)
           self.slider_end = tk.Scale(self.frame_time, from_=0, to=100, orient=tk.HORIZONTAL, resolution=1.e-9, showvalue=0, command=self.update_from_sliders)
           self.slider_end.set(1)
           self.slider_end.pack(side=tk.LEFT)

           self.entry_end_time = ttk.Entry(self.frame_time, width=10)
           self.entry_end_time.insert(0, str(1))
           self.entry_end_time.pack(side=tk.LEFT)
           self.entry_end_time.bind("<Return>", self.entry_end_time_changed)

           # Reset plot button
           self.reset_plot_button = ttk.Button(self.frame_time, text="Reset", command=self.reset_plot)
           self.reset_plot_button.pack(side=tk.LEFT, padx=10)

           # Additional controls (baseline settings)
           self.frame_base = ttk.Frame(control_frame)
           self.frame_base.pack(pady=5)

           label_baseline_offset = ttk.Label(self.frame_base, text="Baseline - Offset:")
           label_baseline_offset.pack(side=tk.LEFT)
           self.entry_baseline_offset = ttk.Entry(self.frame_base, width=5)
           self.entry_baseline_offset.insert(0, "0.1")
           self.entry_baseline_offset.pack(side=tk.LEFT, padx=(0, 20))
           self.entry_baseline_offset.bind("<Return>", self.entry_baseline_changed)

           label_baseline_bins_around = ttk.Label(self.frame_base, text="bins around:")
           label_baseline_bins_around.pack(side=tk.LEFT)
           self.entry_baseline_bins_around = ttk.Entry(self.frame_base, width=5)
           self.entry_baseline_bins_around.insert(0, str(self.bins_around))
           self.entry_baseline_bins_around.pack(side=tk.LEFT)
           self.entry_baseline_bins_around.bind("<Return>", self.entry_baseline_changed)

           self.btn_plot_cursor = ttk.Button(control_frame, text="Select Time Range on Plot", command=lambda: self.on_select(self.ax, self.on_select))
           self.btn_plot_cursor.pack(pady=10)

           # Status label
           self.label_status = ttk.Label(control_frame, text="")
           self.label_status.pack()

    def read_csv(self, file_path):
        """Reads the oscilloscope CSV file and extracts time and channel data."""
        time = []
        ch1 = []
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
                            time.append(float(row[0]))
                            ch1.append(float(row[1]))
                        except ValueError:
                            print(f"Skipping invalid data row: {row}")
        except Exception:
            print(f"Error: File not found at {file_path}")
            return np.array([]), np.array([])
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return np.array([]), np.array([])

        return np.array(time), np.array(ch1)


    def update_plot(self, plot_key, start_time=None, end_time=None, baseline_offset=0.0, bins_around=25):
        """
        Updates the plot and detects peaks below a dynamic baseline.

        Args:
            plot_key (str): The key of the plot to update (filename).
            start_time (float, optional): Start time for the selected range. Defaults to None.
            end_time (float, optional): End time for the selected range. Defaults to None.
            baseline_offset (float, optional): Offset for baseline calculation. Defaults to 0.0.
            bins_around (int, optional): Number of bins around for baseline. Defaults to 25.
        """
        if plot_key not in self.plots:
            return

        if hasattr(self.parent, "db"):
           if "File" in self.parent.db.columns:
              result = self.parent.db.loc[self.parent.db["File"] == plot_key] 
              #print("result ", result );
              self.file_info_text.delete("1.0", tk.END)
              #bold_font = font.Font(weight="bold")
              #self.file_info_text.insert(tk.END, f"Info\n", ("bold"))
              # Loop over the rows and columns in the DataFrame
              for index, row in result.iterrows():
                 #self.file_info_text.insert(tk.END, f"Row {index}:\n")  # Optional: Identify the row
                 for column_name, value in row.items():
                    if ( column_name == "File" ): continue
                    if ( value       == "None" ): continue
                    if ( value       ==  None  ): continue
                    self.file_info_text.insert(tk.END, f"{column_name:<10}: {value}\n")
                    #self.file_info_text.insert(tk.END, "\n")  # Add a blank line between rows

        time, ch1 = self.plots[plot_key]['time'], self.plots[plot_key]['ch1']

        if start_time is None:
            start_time = float(self.entry_start_time.get())
        if end_time is None:
            end_time = float(self.entry_end_time.get())
        baseline_offset = float(self.entry_baseline_offset.get())

        mask = (time >= start_time) & (time <= end_time)
        selected_time = time[mask]
        selected_ch1 = ch1[mask]

        dynamic_baseline = np.array([
            np.mean(ch1[max(0, i - bins_around):min(len(ch1), i + bins_around)]) - baseline_offset
            for i in np.where(mask)[0]
        ])

        # Find peaks below the dynamic baseline
        inverted_ch1 = -(selected_ch1 - dynamic_baseline)
        peaks_below, _ = find_peaks(inverted_ch1, height=0)

        self.ax.clear()
        self.ax.plot(selected_time, selected_ch1, label="CH1")
        self.ax.plot(selected_time, dynamic_baseline, color='g', linestyle='--', label="Dynamic Baseline")
        self.ax.plot(selected_time[peaks_below], selected_ch1[peaks_below], 'x', label="Peaks Below")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title(plot_key)
        self.ax.legend()
        self.ax.grid()

        self.canvas.draw()
        self.label_status.config(text=f"Number of Peaks Below Baseline: {len(peaks_below)}")
        self.file_info_text.insert(tk.END, f"Peaks     : {len(peaks_below)}\n")

        return 



    def on_select(self, xmin, xmax):
        """Handles time range selection via cursor drag."""
        self.entry_start_time.delete(0, tk.END)
        self.entry_end_time.delete(0, tk.END)
        if (not isinstance(xmin, numbers.Number)): return
        self.entry_start_time.insert(0, str(f"{xmin:.2e}"))
        if (not isinstance(xmax, numbers.Number)): return
        self.entry_end_time.insert(0, str(f"{xmax:.2e}"))

        self.slider_start.set(xmin)
        self.slider_end.set(xmax)
        baseline_offset = self.entry_baseline_offset.get()
        bins_around = int(self.entry_baseline_bins_around.get())
        self.update_plot(self.plot_selector.get(), xmin, xmax, baseline_offset, bins_around)
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
        self.span_selector = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True,
                                    interactive=True)



    def open_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        self.open_file_from_paths(file_paths)
    def open_file_from_paths(self,file_paths):
        """Opens a file dialog to select files and initializes the plots."""
        #file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_paths:
            self.label_status.config(text="No files selected.")
            return

        try:
            self.plots = {}
            self.plot_titles = []
            timeMin = 1.e10
            timeMax = -1e10
            for i, file_path in enumerate(file_paths):
                #file_name = os.path.splitext(os.path.basename(file_path))[0]
                file_name = file_path
                plot_title = file_name
                time, ch1 = self.read_csv(file_path)

                if ( len(time) < 1 ): continue

                self.plots[plot_title] = {'time': time, 'ch1': ch1}
                if (time[0] < timeMin): timeMin = time[0]
                if (time[-1] > timeMax): timeMax = time[-1]
                self.plot_titles.append(plot_title)

            self.canvas.draw()
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
            self.span_selector = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True,
                                        interactive=True)
            self.plot_selector['values'] = self.plot_titles

    
            if self.plot_titles:
                self.plot_selector.current(0)
                self.slider_start.config(from_=timeMin, to=timeMax)
                self.slider_end.config(from_=timeMin, to=timeMax)
                self.slider_start.set(timeMin)
                self.slider_end.set(timeMax)
                self.entry_start_time.delete(0, tk.END)
                self.entry_end_time.delete(0, tk.END)
                self.entry_start_time.insert(0, str(f"{timeMin:.2e}"))
                self.entry_end_time.insert(0, str(f"{timeMax:.2e}"))
                self.update_plot(self.plot_titles[0], timeMin, timeMax)
            self.label_status.config(text="Files loaded successfully!")

        except Exception as e:
            self.label_status.config(text=f"Error loading files: {e}")
            print(f"Error loading file: {e}")



    def update_from_sliders(self, value):
        """Updates the plot based on slider values."""
        start_time = self.slider_start.get()
        end_time = self.slider_end.get()
        self.entry_start_time.delete(0, tk.END)
        self.entry_end_time.delete(0, tk.END)
        self.entry_start_time.insert(0, str(f"{start_time:.2e}"))
        self.entry_end_time.insert(0, str(f"{end_time:.2e}"))
        self.update_plot(self.plot_selector.get(), start_time, end_time)



    def entry_start_time_changed(self, event):
        try:
            slider_value = float(self.entry_start_time.get())
            self.slider_start.set(slider_value)
            self.update_plot(self.plot_selector.get(), start_time=slider_value)
        except ValueError:
            print("Invalid input.  Please enter a number.")



    def entry_end_time_changed(self, event):
        try:
            slider_value = float(self.entry_end_time.get())
            self.slider_end.set(slider_value)
            self.update_plot(self.plot_selector.get(), end_time=slider_value)
        except ValueError:
            print("Invalid input.  Please enter a number.")


    def entry_baseline_changed(self, event):
        start_time = self.slider_start.get()
        end_time = self.slider_end.get()
        baseline_offset = self.entry_baseline_offset.get()
        bins_around = int(self.entry_baseline_bins_around.get())
        self.update_plot(self.plot_selector.get(), start_time, end_time, baseline_offset, bins_around)

    def reset_plot(self, event=None):
        self.slider_start.set(self.slider_start['from'])
        self.slider_end.set(self.slider_end['to'])

        tMin = self.slider_start.get()
        tMax = self.slider_end.get()
        self.entry_start_time.delete(0, tk.END)
        self.entry_end_time.delete(0, tk.END)
        self.entry_start_time.insert(0, str(f"{tMin:.2e}"))
        self.entry_end_time.insert(0, str(f"{tMax:.2e}"))
        self.update_plot(self.plot_selector.get(), tMin, tMax)

class CSVDataFilterApp(ttk.Frame):
    def __init__(self, parent, master):
        self.parent = parent
        self.master = master
        super().__init__(master)

        self.df = None
        self.unique_values = {}
        self.selected_values = {}
        self.current_column = None
        self.filter_listbox = None
        self.column_units = {}  # Store column units

        self.file_path_label = ttk.Label(self.master, text="No file selected")
        self.file_path_label.pack(pady=1)

        self.frame_files = ttk.Frame(self.master)
        self.frame_files.pack(pady=1)
        self.load_button = ttk.Button(self.frame_files, text="Load CSV File", command=self.load_csv_file)
        self.load_button.pack(side=tk.LEFT, pady=1)

        self.col_dropdown = ttk.Combobox(self.frame_files, values=[], state="readonly")
        self.col_dropdown.pack(side=tk.LEFT, padx=10, pady=10)
        self.col_dropdown.bind("<<ComboboxSelected>>", self.show_filter_options)

        self.reset_button = ttk.Button(self.frame_files, text="Reset Filters", command=self.reset_filters)
        self.reset_button.pack(side=tk.LEFT)

        self.get_files_button = ttk.Button(self.frame_files, text="Load Files", command=self.get_files)
        self.get_files_button.pack(side=tk.LEFT)

        self.frame_filters = ttk.Frame(self.master)
        self.frame_filters.pack(pady=1)
        self.active_filters_label = ttk.Label(self.frame_filters, text="Active Filters:")
        self.active_filters_label.pack(side=tk.LEFT, pady=10)

        self.filter_frame = ttk.Frame(self.master)  # Frame for filter options
        self.filter_frame.pack(pady=20)  # Use grid for filter frame
        
        #self.result_text = tk.Text(self.master, wrap=tk.WORD, height=25)
        self.result_text = tk.Text(self.master, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH,expand=True)

    def load_csv_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.file_path_label.config(text=f"Selected file: {file_path}")
            try:
                # Read the CSV file, handling potential errors
                df = pd.read_csv(file_path, header=None)  # Read without a header initially
                if len(df) < 2:
                    raise ValueError("CSV file must have at least two rows (header and data).")

                new_header = df.iloc[0].astype(str).tolist()  # Get the first row as the new header
                self.column_units = {}
                if len(df) > 1:
                    units_row = df.iloc[1].astype(str).tolist()  # Get the second row as units
                    for i, col in enumerate(new_header):
                        if i < len(units_row):
                            if ( units_row[i] != 'nan' ) : self.column_units[col] = f"({units_row[i]})"
                     #       else:                          self.column_units[col] = ""
                        else:
                            self.column_units[col] = ""
                else:
                    for col in new_header:
                        self.column_units[col] = ""

                df = df[2:]  # Keep rows starting from the third row (index 2)
                df.columns = new_header  # Set the new header
                df = df.replace({np.nan: None}) # Replace NaN with None

                self.df = df
                self.populate_column_dropdown()
            except (ValueError, pd.errors.EmptyDataError, IndexError) as e:
                self.result_text.insert(tk.END, f"Error loading CSV file: {e}\n")
                self.df = None
        else:
            self.file_path_label.config(text="No file selected")

    def populate_column_dropdown(self):
        """Populates the column selection dropdown."""
        if self.df is not None:
            columns = list(self.df.columns)
            self.col_dropdown['values'] = columns
            if columns:
                self.col_dropdown.current(0)
                self.current_column = columns[0]  # Set initial column
                self.show_filter_options(columns[0])  # Show options for the first column
        else:
            self.col_dropdown['values'] = []
            self.filter_frame.winfo_children().clear()
            self.result_text.delete("1.0", tk.END)

    def show_filter_options(self, column=None):
        """Displays filter options (Listbox) for a given column within a tab."""
        for widget in self.filter_frame.winfo_children():
            widget.destroy()

        if self.df is None:
            return

        if isinstance(column, tk.Event):
            column_name = self.col_dropdown.get()
        else:
            column_name = column

        if not column_name:  # prevent error
            return

        tab = ttk.Frame(self.filter_frame)
        tab.pack(fill=tk.BOTH, expand=True)

        col_frame = ttk.Frame(tab)  # Use the passed-in tab
        col_frame.pack(fill=tk.BOTH, expand=True)

        col_label_text = f"Select Filter for: {column_name} {self.column_units.get(column_name, '')}"
        col_label = ttk.Label(col_frame, text=col_label_text, font=('Arial', 12, 'bold'))
        col_label.pack(anchor=tk.W)

        values = [v for v in self.df[column_name].unique()]
        self.unique_values[column_name] = values
        if column_name not in self.selected_values:
            self.selected_values[column_name] = []

        self.filter_listbox = Listbox(col_frame, selectmode=MULTIPLE, height=min(10, len(values)))
        for value in values:
            if ( value == None ): valueI = "None"
            else: valueI = value
            self.filter_listbox.insert(tk.END, valueI)


        # Pre-select previously selected values
        for i, value in enumerate(values):
            if value in self.selected_values[column_name]:
                self.filter_listbox.selection_set(i)

        self.filter_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) #changed this to LEFT

        # Add a scrollbar to the Listbox
        scrollbar = Scrollbar(col_frame, command=self.filter_listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.filter_listbox.config(yscrollcommand=scrollbar.set)

        self.filter_listbox.bind('<<ListboxSelect>>', self.update_selection)

        button = ttk.Button(col_frame, text=f"Filter", command=self.filter_data)
        button.pack(pady=5, anchor=tk.W)

        #col_frame.grid_columnconfigure(0, weight=1)

    def on_tab_changed(self, event):
        """Handles tab selection changes."""
        selected_tab = self.notebook.select()  # Get the selected tab's widget
        if selected_tab:
            tab_text = self.notebook.tab(selected_tab, "text")  # Get the tab's text label
            self.current_column = tab_text
            self.show_filter_options(tab_text)

    def update_selection(self, event):
        """Called when items are selected or deselected in the Listbox."""
        self.current_column = self.col_dropdown.get()
        if not self.current_column:
            return
        listbox = event.widget
        selected_indices = listbox.curselection()
        self.selected_values[self.current_column] = [
            None if listbox.get(index) == "None" else listbox.get(index)
            for index in selected_indices
        ]
        self.active_filters_label.config(text=f"Active Filters: {self.selected_values}")

    def filter_data(self):
        """Filters the DataFrame based on the selected values and displays the result."""
        if self.df is None:
            self.result_text.insert("1.0", "Please load a CSV file first.\n")
            return

        filters = {}
        for col in self.df.columns:
            if col in self.selected_values and self.selected_values[col]:
                filters[col] = self.selected_values[col]
            else:
                filters[col] = list(self.df[col].unique())

        # Create a boolean mask by combining all column filters
        mask = np.ones(len(self.df), dtype=bool)
        for col, selected_values in filters.items():
            mask &= self.df[col].isin(selected_values)

        filtered_df = self.df[mask]

        self.result_text.delete("1.0", tk.END)
        if len(filtered_df) > 100:
            self.result_text.insert("1.0", "Displaying the first 100 rows of the filtered data:\n")
            self.result_text.insert(tk.END, filtered_df.head(100).to_string() + "\n")
        else:
            self.result_text.insert("1.0", "Filtered data:\n")
            self.result_text.insert(tk.END, filtered_df.to_string() + "\n")

    def get_files(self):
        """Filters the DataFrame based on the selected values and displays the result."""
        if self.df is None:
            self.result_text.insert("1.0", "Please load a CSV file first.\n")
            return

        filters = {}
        for col in self.df.columns:
            if col in self.selected_values and self.selected_values[col]:
                filters[col] = self.selected_values[col]
            else:
                filters[col] = list(self.df[col].unique())

        # Create a boolean mask by combining all column filters
        mask = np.ones(len(self.df), dtype=bool)
        for col, selected_values in filters.items():
            mask &= self.df[col].isin(selected_values)

        filtered_df = self.df[mask]
        specific_columns=["FileMin","FileMax"]
        #print("filtered_df ",filtered_df)
        file_info=filtered_df[specific_columns]
        #print("file_info ",file_info)

        all_files = []
        file_df_rows=[]
        #for index, row in file_info.iterrows():
        for index, row in filtered_df.iterrows():
           start_file = row['FileMin']
           end_file   = row['FileMax']

           if ( len(start_file) < 2 ): continue;
           if ( len(end_file)   < 2 ): continue;
           #self.get_files_between_pairs([(fileMin,fileMax)])
           #print(f"Index: {index}, start_file: {start_file}, end_file: {end_file}")

           try:
               # Update regex to match 3-digit or 4-digit numbers
               prefix_start, num_start, suffix_start = re.match(r"(.*?)(\d{3,4})(.*)", start_file).groups()
               prefix_end, num_end, suffix_end = re.match(r"(.*?)(\d{3,4})(.*)", end_file).groups()

                  # Debug output for extracted components
               #print(f"Start file: {start_file} -> Prefix: {prefix_start}, Number: {num_start}, Suffix: {suffix_start}")
               #print(f"End file: {end_file} -> Prefix: {prefix_end}, Number: {num_end}, Suffix: {suffix_end}")

           except AttributeError:
               raise ValueError(f"Invalid file name format in pair: {start_file} {end_file}")

           # Ensure prefixes and suffixes match
           if prefix_start != prefix_end or suffix_start != suffix_end:
               raise ValueError(f"File prefixes or suffixes do not match in pair: {pair}")

           # Generate all files between the start and end numbers
           start_num = int(num_start)
           end_num = int(num_end)
           if start_num > end_num:
               raise ValueError(f"Start number {start_num} is greater than end number {end_num} in pair: {pair}")

           # Dynamically pad numbers based on their original length (3 or 4 digits)
           num_length = len(num_start)
           files = [f"{prefix_start}{str(i).zfill(num_length)}{suffix_start}" for i in range(start_num, end_num + 1)]
           all_files.extend(files)

           for file in files:
               new_row = row.copy()
               new_row["File"]=file
               file_df_rows.append(new_row)
           #print("files ", files)
        #print("all_files ", all_files)
        self.parent.db = pd.DataFrame(file_df_rows).drop(columns=["FileMin","FileMax"])

        self.parent.scope_view.open_file_from_paths(all_files)

        #print(self.parent.db)
    def reset_filters(self):
        """Resets all selected filters and clears the result text."""
        if self.df is not None:
            #self.selected_values = {col: [] for col in self.df.columns}  # Clear all selections
            self.selected_values = {}  # Clear all selections
            self.active_filters_label.config(text=f"Active Filters: None")
            self.result_text.delete("1.0", tk.END)  # Clear the result text area
            self.populate_column_dropdown()
            self.result_text.insert("1.0", "All filters have been reset.\n")
        else:
            self.result_text.insert("1.0", "Load a CSV file to reset.\n")



class DetectorAnalysis(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detector Analysis Application")
        self.geometry("1000x900")  # Increased window size for better usability

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Oscilloscope Data Viewer")
        self.notebook.add(self.tab2, text="CSV Data Filter")

        db = pd.DataFrame()

        self.scope_view = ScopeViewApp(self, self.tab1) # Use the class
        self.csv_data   = CSVDataFilterApp(self,self.tab2) # Use the class


def main():
    app = DetectorAnalysis()
    app.mainloop()

if __name__ == "__main__":
    main()
