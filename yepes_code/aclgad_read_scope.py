#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import Listbox, Scrollbar, MULTIPLE, EXTENDED
import tkinter.messagebox as messagebox
from tkinter import font
from tkinter import Toplevel, Checkbutton, Button
from tkinter.simpledialog import askstring
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

        self.baseline_bin_around = 25
        self.baseline_offset  = .15
        self.selected_ch1 = None
        self.loaded_file_path = None
        self.time = np.array([])
        self.ch1 = np.array([])
        self.plots = {}
        self.ax = None
        self.span_selector = None
        self.bins_around = 10
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

        # Previous Button
        self.prev_button = tk.Button(self.frame_files, text="Previous", command=self.select_previous)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        # Next Button
        self.next_button = tk.Button(self.frame_files, text="Next", command=self.select_next)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Delete Button
        self.delete_button = tk.Button(self.frame_files, text="Delete File", command=self.delete_file)
        self.delete_button.pack(side=tk.LEFT, padx=5)
 

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
        self.entry_baseline_offset.insert(0, "0.15")
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



    def delete_file(self):
       """
       Asks for user confirmation via a popup window before deleting a file.

       Args:
           filepath (str): The path to the file to be deleted.
       """
       filepath=self.plot_selector.get()
       print("filepath ",filepath)
       if not os.path.exists(filepath):
           messagebox.showerror("Error", f"File not found: {filepath}")
           return

       confirmed = messagebox.askyesno(
           "Confirmation",
           f"Are you sure you want to delete the file:\n'{os.path.basename(filepath)}'?",
           icon='warning'
       )

       if confirmed:
           try:
               os.remove(filepath)
               # Remove the filename from the combobox and go to the next
               values_list = list(self.plot_selector['values']) #get the values.  It was missing.
               if filepath in values_list:
                  index_to_remove = values_list.index(filepath)
                  values_list.remove(filepath)  # Remove from the source list
                  self.plot_selector['values'] = values_list  # Update combobox values

                  if values_list:  # Check if the list is not empty
                     if index_to_remove >= len(values_list):
                        index_to_select = len(values_list) - 1
                     else:
                        index_to_select = index_to_remove
                     self.plot_selector.current(index_to_select)  # Set combobox to the next value
                  else:
                     self.plot_selector.set('')  # Clear the combobox

               self.update_plot(self.plot_selector.get())
               messagebox.showinfo("Success", f"File '{os.path.basename(filepath)}' deleted successfully.")
           except Exception as e:
               messagebox.showerror("Error", f"Error deleting file '{os.path.basename(filepath)}':\n{e}")
       else:
           messagebox.showinfo("Info", f"Deletion of '{os.path.basename(filepath)}' cancelled.")

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

    def select_previous(self):
        # Get the current selection
        values = self.plot_selector['values']
        current_index = values.index(self.plot_selector.get())
        # Select the previous option if available
        if current_index > 0:
            self.plot_selector.set(values[current_index - 1])
            self.update_plot(self.plot_selector.get())

    def select_next(self):
        # Get the current selection
        values = self.plot_selector['values']
        current_index = values.index(self.plot_selector.get())
        # Select the next option if available
        if current_index < len(values) - 1:
            self.plot_selector.set(values[current_index + 1])
            self.update_plot(self.plot_selector.get())

    def get_peaks(self, plot_key, start_time=None, end_time=None):

        time, ch1 = self.plots[plot_key]['time'], self.plots[plot_key]['ch1']

        if start_time is None:
            start_time = float(self.entry_start_time.get())
        if end_time is None:
            end_time = float(self.entry_end_time.get())
        #print("start_time end_time ", start_time, end_time)

        self.baseline_offset = float(self.entry_baseline_offset.get())
        self.bins_around = int(self.entry_baseline_bins_around.get())
        #print("self.baseline_offset ", self.baseline_offset)
        #print("self.bins_around    ", self.bins_around)

        mask = (time >= start_time) & (time <= end_time)
        self.selected_time = time[mask]
        self.selected_ch1 = ch1[mask]

        self.dynamic_baseline = np.array([
            np.mean(ch1[max(0, i - self.bins_around):min(len(ch1), i + self.bins_around)]) - self.baseline_offset
            for i in np.where(mask)[0]
        ])

        self.selected_ch1 = np.nan_to_num(self.selected_ch1, nan=0.0, posinf=0.0, neginf=0.0)
        self.dynamic_baseline = np.nan_to_num(self.dynamic_baseline, nan=0.0, posinf=0.0, neginf=0.0)
        # Find peaks below the dynamic baseline
        self.inverted_ch1 = -(self.selected_ch1 - self.dynamic_baseline)
        self.peaks_below, _ = find_peaks(self.inverted_ch1, height=0)

        return 

    def update_plot(self, plot_key, start_time=None, end_time=None):
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

        self.file_info_text.delete("1.0", tk.END)
        if hasattr(self.parent, "df"):
           if "File" in self.parent.df.columns:
              result = self.parent.df.loc[self.parent.df["File"] == plot_key] 
              #print("result ", result );
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


        self.get_peaks(plot_key, start_time, end_time)

        self.ax.clear()
        self.ax.plot(self.selected_time, self.selected_ch1, label="CH1")
        self.ax.plot(self.selected_time, self.dynamic_baseline, color='g', linestyle='--', label="Dynamic Baseline")
        self.ax.plot(self.selected_time[self.peaks_below], self.selected_ch1[self.peaks_below], 'x', label="Peaks Below")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title(plot_key)
        self.ax.legend()
        self.ax.grid()

        self.canvas.draw()
        self.label_status.config(text=f"Number of Peaks Below Baseline: {len(self.peaks_below)}")
        self.file_info_text.insert(tk.END, f"Peaks     : {len(self.peaks_below)}\n")

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
        self.baseline_offset = self.entry_baseline_offset.get()
        self.bins_around = int(self.entry_baseline_bins_around.get())
        self.update_plot(self.plot_selector.get(), xmin, xmax, self.baseline_offset, self.bins_around)
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
        self.baseline_offset = self.entry_baseline_offset.get()
        self.bins_around = int(self.entry_baseline_bins_around.get())
        self.update_plot(self.plot_selector.get(), start_time, end_time)

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
        self.hidden_columns = []  # List to store names of hidden columns

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

        self.load_files_button = ttk.Button(self.frame_files, text="Load Files", command=self.load_files)
        self.load_files_button.pack(side=tk.LEFT, padx=1)

        self.hide_columns_button = ttk.Button(self.frame_files, text="Hide Columns", command=self.show_hide_columns_dialog)
        self.hide_columns_button.pack(side=tk.LEFT, padx=1)


        self.frame_filters = ttk.Frame(self.master)
        self.frame_filters.pack(pady=1)
        self.active_filters_label = ttk.Label(self.frame_filters, text="Active Filters:")
        self.active_filters_label.pack(side=tk.LEFT, pady=10)

        self.filter_frame = ttk.Frame(self.master)  # Frame for filter options
        self.filter_frame.pack(pady=20)  # Use grid for filter frame
        
        self.result_text = tk.Text(self.master, wrap=tk.NONE)

        # Vertical scrollbar
        self.v_scrollbar = tk.Scrollbar(self.master, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=self.v_scrollbar.set)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        self.h_scrollbar = tk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.result_text.xview)
        self.result_text.config(xscrollcommand=self.h_scrollbar.set)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Pack the text widget
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        

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

        #filtered_df = self.df[mask]
        filtered_df0 = self.df[mask]
        filtered_df  = filtered_df0.drop(columns=self.hidden_columns, errors='ignore')

        self.result_text.delete("1.0", tk.END)
        self.show_files(filtered_df)
        #if len(filtered_df) > 100:
        #    self.result_text.insert("1.0", "Displaying the first 100 rows of the filtered data:\n")
        #    self.result_text.insert(tk.END, filtered_df.head(100).to_string() + "\n")
        #else:
        #    self.result_text.insert("1.0", "Filtered data:\n")
        #    self.result_text.insert(tk.END, filtered_df.to_string() + "\n")

    def extract_file_parts(self, filename):
       basename = filename.split("/")[-1]  # Extract just the filename
       path = "/".join(filename.split("/")[:-1]) + "/"  # Preserve full directory path

       if "tek" in basename:
           match = re.match(r"(.*?)(\d{3,4})([^/]*\.csv)$", basename)  # Handles 'tek' files
       else:
           match = re.match(r"(.*?-)(\d{4})(\.csv)$", basename)  # Handles 'scope-results' format

       if match:
           prefix, num, suffix = match.groups()
           return path + prefix, num, suffix  # Combine prefix with full path

       return filename, None, None


    def load_files(self):
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
        specific_columns=["Filename min","Filename max"]
        #print("filtered_df ",filtered_df)
        file_info=filtered_df[specific_columns]
        #print("file_info ",file_info)

        all_files = []
        file_df_rows=[]
        #for index, row in file_info.iterrows():
        for index, row in filtered_df.iterrows():
           start_file = row['Filename min']
           end_file   = row['Filename max']

           if ( len(start_file) < 2 ): continue;
           if ( len(end_file)   < 2 ): continue;
           #self.load_files_between_pairs([(fileMin,fileMax)])
           #print(f"Index: {index}, start_file: {start_file}, end_file: {end_file}")


           try:
# Apply extraction
               prefix_start, num_start, suffix_start = self.extract_file_parts(start_file)
               prefix_end, num_end, suffix_end       = self.extract_file_parts(end_file)
                  # Debug output for extracted components
               #print(f"Start file: {start_file} -> Prefix: {prefix_start}, Number: {num_start}, Suffix: {suffix_start}")
               #print(f"End file: {end_file} -> Prefix: {prefix_end}, Number: {num_end}, Suffix: {suffix_end}")

           except AttributeError:
               raise ValueError(f"Invalid file name format in pair: {start_file} {end_file}")

           # Ensure prefixes and suffixes match
           if prefix_start != prefix_end or suffix_start != suffix_end:
               raise ValueError(f"File prefixes do not match in pair: {prefix_start} {prefix_end} {suffix_start} {suffix_end}")

           # Generate all files between the start and end numbers
           start_num = int(num_start)
           end_num = int(num_end)
           if start_num > end_num:
               raise ValueError(f"Start number {start_num} is greater than end number {end_num} in pair: {pair}")

           # Dynamically pad numbers based on their original length (3 or 4 digits)
           num_length = len(num_start)
           prefix_start = prefix_start.replace("/home/pyepes/data/", "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/")
           prefix_start = prefix_start.replace("2025-05-06","2025-05-05")
           files = [f"{prefix_start}{str(i).zfill(num_length)}{suffix_start}" for i in range(start_num, end_num + 1)]

           for file in files:
               if os.path.exists(file):  # Check if the file exists
                  new_row = row.copy()
                  new_row["File"]=file
                  file_df_rows.append(new_row)
                  all_files.append(file)
               else:
                  print(f"Warning: File '{file}' does not exist. Skipping...")
               #print("file ",file, "nPeaks ", nPeaks)

           #print("files ", files)
        #print("all_files ", all_files)
        print("file_df_rows ", file_df_rows)

        if not file_df_rows:
            print("No file found.")
        else:
           self.parent.df = pd.DataFrame(file_df_rows).drop(columns=["Filename min","Filename max"])
           print("CCCC upload_files self.parent.df ", self.parent.df)

           self.parent.scope_view.open_file_from_paths(all_files)

           nPeaks_list = []  # List to store nPeaks for each file
           pDose_list  = []  # List to store nPeaks for each file
           doseScale = 0.0054
           for file in all_files:
              self.parent.scope_view.get_peaks(file)
              nPeaks = len(self.parent.scope_view.peaks_below)
              pDose  = doseScale*nPeaks
              #print("file ",file, "nPeaks ", nPeaks)
              nPeaks_list.append(nPeaks)  # Store nPeaks in the list
              pDose_list.append(pDose)  # Store nPeaks in the list

           #print("Before  self.parent.df with nPeaks column:\n", self.parent.df)
           #print("Before  nPeaks_list:\n", len(nPeaks_list))
           self.parent.df['nPeaks'] = nPeaks_list
           self.parent.df['pDose']  = pDose_list
           #print("Updated self.parent.df with nPeaks column:\n", self.parent.df)

           self.parent.analysis.update_plot_tab_columns()
           #print(self.parent.df)
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

    def show_hide_columns_dialog(self):
        """
        Displays a dialog window to allow the user to select columns to hide.
        """
        if self.df is None:
            messagebox.showwarning("Warning", "No DataFrame loaded.")
            return

        # Create a new top-level window for the dialog
        dialog = Toplevel(self.master)
        dialog.title("Hide Columns")

        # Get all column names
        columns = list(self.df.columns)

        # Create a dictionary to store the Checkbutton variables
        self.column_vars = {}
        for col in columns:
            self.column_vars[col] = tk.BooleanVar()
            # Set the initial state of the Checkbutton based on whether the column is currently hidden
            self.column_vars[col].set(col in self.hidden_columns)

            # Create a Checkbutton for each column
            check_button = Checkbutton(dialog, text=col, variable=self.column_vars[col])
            check_button.pack(anchor=tk.W)

        # Create a frame for the buttons at the bottom of the dialog
        button_frame = ttk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Add a "OK" button to apply the changes
        ok_button = ttk.Button(button_frame, text="OK", command=lambda: self.on_hide_columns_ok(dialog))
        ok_button.pack(side=tk.RIGHT, padx=5)

        # Add a "Cancel" button to close the dialog without applying changes
        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT)

        # Make the dialog modal (optional)
        dialog.grab_set()
        dialog.focus_set()
        dialog.wait_window()  # Wait until the dialog is closed

    def on_hide_columns_ok(self, dialog):
        """
        Handles the "OK" button click in the hide columns dialog.
        Updates the list of hidden columns and refreshes the view.

        Args:
            dialog (tk.Toplevel): The dialog window.
        """
        self.hidden_columns = [col for col, var in self.column_vars.items() if var.get()]
        print("hidden_columns ", self.hidden_columns)
        #self.update_view_tab()  # Refresh the view to reflect the hidden columns
        self.filter_data()  # Refresh the view to reflect the hidden columns
        self.parent.analysis.update_plot_tab_columns()
        dialog.destroy()  # Close the dialog window

    def update_view_tab0(self):
        """
        Updates the "View Data" tab with the current DataFrame, excluding hidden columns.
        """
        if self.df is not None:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            # Display the DataFrame, excluding hidden columns
            df_to_display = self.df.drop(columns=self.hidden_columns, errors='ignore')
            self.result_text.insert(tk.END, df_to_display.to_string())
            self.result_text.config(state=tk.DISABLED)
        else:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "No DataFrame to display. Open a CSV file.")
            self.result_text.config(state=tk.DISABLED)


    def show_files(self, filtered_df):
        if self.df is not None:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)

            # Remove directory path from FileMin & FileMax columns
            df_to_display = filtered_df.drop(columns=self.hidden_columns, errors='ignore').copy()
            if 'Filename min' in df_to_display.columns:
                df_to_display['Filename min'] = df_to_display['Filename min'].apply(os.path.basename)
            if 'Filename max' in df_to_display.columns:
                df_to_display['Filename max'] = df_to_display['Filename max'].apply(os.path.basename)

            # Configure alternating row colors
            self.result_text.tag_configure("even", background="light blue")
            self.result_text.tag_configure("odd", background="white")

            for i, line in enumerate(df_to_display.to_string().split("\n")):
                tag = "even" if i % 2 == 0 else "odd"
                self.result_text.insert(tk.END, line + "\n", tag)

            self.result_text.config(state=tk.DISABLED)
        else:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "No DataFrame to display. Open a CSV file.")
            self.result_text.config(state=tk.DISABLED)


class DataAnalysisApp(ttk.Frame):
    """
    A class to display a Pandas DataFrame in a Tkinter GUI with enhanced features:
    - Two tabs: one for viewing data, one for plotting.
    - Option to hide columns from view and plotting.
    - Option to select rows for plotting using a query.
    - Averages y values for duplicate x values before plotting.
    """

    def __init__(self, parent, master=None, **kwargs):
        self.parent = parent
        self.master = master
        super().__init__(master, **kwargs)  # Corrected superclass initialization
        """
        Initializes the DataAnalysisApp.

        Args:
            parent (tk.Tk): The root Tkinter window.
            df (pd.DataFrame, optional): Initial DataFrame to display. Defaults to None.
        """
        self.df     = None
        self.parent = parent
        self.plot_titles = []
        self.row_query = ""  # String to store the row selection query

        #self.file_path_label = ttk.Label(self.master, text="No file selected")
        #self.file_path_label.pack(pady=1)
        self.create_widgets()

    def create_widgets(self):
        # Create frames for the tabs
        #self.view_frame = ttk.Frame(self.notebook)

        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        plot_input_frame = ttk.Frame(self.plot_frame)
        plot_input_frame.pack(pady=10)

        # Labels and dropdowns for selecting columns
        ttk.Label(plot_input_frame, text="X Column:").grid(row=0, column=0, padx=5, pady=5)
        self.x_column_var = tk.StringVar()
        self.x_column_dropdown = ttk.Combobox(plot_input_frame, textvariable=self.x_column_var, state="readonly")
        self.x_column_dropdown.grid(row=0, column=1, padx=5, pady=5)

        #ttk.Label(plot_input_frame, text="Y Column:").grid(row=1, column=0, padx=5, pady=5)
        #self.y_column_var = tk.StringVar()
        #self.y_column_dropdown = ttk.Combobox(plot_input_frame, textvariable=self.y_column_var, state="readonly")
        #self.y_column_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(plot_input_frame, text="Y Column:").grid(row=1, column=0, padx=5, pady=5)
        self.y_column_var = tk.StringVar()  # Variable to track the selected value
        self.y_column_listbox = tk.Listbox(plot_input_frame, selectmode=MULTIPLE, height=5, exportselection=False)  # `height` sets visible rows
        self.y_column_listbox.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(plot_input_frame, text="Group By Column:").grid(row=2, column=0, padx=5, pady=5)
        self.group_column_var = tk.StringVar()
        self.group_column_dropdown = ttk.Combobox(plot_input_frame, textvariable=self.group_column_var, state="readonly")
        self.group_column_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Entry for row selection
        ttk.Label(plot_input_frame, text="Select Rows (e.g., col1 > 5 and col2 == 'abc'):").grid(row=3, column=0, padx=5, pady=5)
        self.row_query_entry = ttk.Entry(plot_input_frame, width=40)
        self.row_query_entry.grid(row=3, column=1, padx=5, pady=5)

        # Create a button to generate the plot
        self.plot_button = ttk.Button(plot_input_frame, text="Generate Plot", command=self.generate_plot)
        self.plot_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Create a button to generate the plot
        self.save_plot_button = ttk.Button(plot_input_frame, text="Save Plot", command=self.save_plot)
        self.save_plot_button.grid(row=4, column=2, columnspan=2, pady=10)


        # Canvas for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))  # Store fig and ax
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add a label for status messages
        self.status_label = ttk.Label(self.plot_frame, text="")
        self.status_label.pack(pady=5)


        # Add the frames to the notebook
        #self.notebook.add(self.view_frame, text="View Data")
        #self.notebook.add(self.plot_frame, text="Plot Data")

    def update_plot_tab_columns(self):
        """
        Updates the dropdown menus in the "Plot Data" tab with the columns
        from the current DataFrame, excluding hidden columns.
        """
        self.df = self.parent.df
        hidden_columns = self.parent.csv_data.hidden_columns
        print("update_plot_tab_columns hidden_columns ", hidden_columns)
        if self.df is not None:
            print("not empty self.df ", self.df)
            available_columns = [col for col in self.df.columns if col not in self.parent.csv_data.hidden_columns]
            self.x_column_dropdown['values'] = available_columns
            #self.y_column_dropdown['values'] = available_columns
            self.y_column_listbox.delete(0, tk.END)  # Deletes all items from the Listbox
            for col in available_columns:
               self.y_column_listbox.insert(tk.END, col)
            
            self.group_column_dropdown['values'] = available_columns
            if available_columns:
                self.x_column_var.set(available_columns[0])
                self.y_column_var.set(available_columns[0])
                self.group_column_var.set(available_columns[0])
        else:
            print("empty self.df ", self.df)
            self.x_column_dropdown['values'] = []
            #self.y_column_dropdown['values'] = []
            self.y_column_listbox.delete(0, tk.END)
            self.group_column_dropdown['values'] = []
            self.x_column_var.set('')
            self.y_column_var.set('')
            self.group_column_var.set('')

    def generate_plot(self):
        """
        Generates a plot based on selected columns and row query.  Averages
        y values for duplicate x values within each group.
        """
        x_col = self.x_column_var.get()
        selected_indices = self.y_column_listbox.curselection() 
        y_cols = [self.y_column_listbox.get(i) for i in selected_indices]
        print("Selected values:", y_cols)
        group_col = self.group_column_var.get()
        self.row_query = self.row_query_entry.get()  # Get row selection query

        self.df = self.df.fillna(value=np.nan)

        if not (self.df.empty):
            if x_col and y_cols and group_col:
                try:
                    # Apply row selection if a query is provided
                    if self.row_query:
                        try:
                            filtered_df = self.parent.df.query(self.row_query)
                        except Exception as e:
                            messagebox.showerror("Error", f"Invalid row selection query:\n{e}")
                            self.status_label.config(text=f"Error: Invalid row query: {e}")
                            return
                    else:
                        filtered_df = self.parent.df

                    print("plot filtered_df ", filtered_df)

                    # Clear previous plot
                    self.ax.clear()
                    self.plot_titles = []

                    grouped_data = filtered_df.groupby(group_col)

                    global_y_min = float('inf')
                    global_y_max = float('-inf')

                    print("grouped_dat ", grouped_data)
                    for y_col in y_cols:
         # Check if the selected columns are in the DataFrame after filtering
                       if x_col not in filtered_df.columns or y_col not in filtered_df.columns or group_col not in filtered_df.columns:
                          messagebox.showerror("Error", "Selected columns not found in filtered data.  Make sure the columns are not hidden and exist after row selection.")
                          self.status_label.config(text="Error: Selected columns not found in filtered data.")
                          return

                       for group_name, group_df in grouped_data:
                           #print("group_name ", group_name)
                           #print("group_df ", group_df)
                           #print("x_col y_col ", x_col, y_col)
                           group_df[y_col] = pd.to_numeric(group_df[y_col], errors='coerce')
                           group_df[x_col] = pd.to_numeric(group_df[x_col], errors='coerce')
                        # Average y values for duplicate x values within each group
                           if len(group_df[x_col].unique()) > 0:
                               # Group by x_col and calculate the mean of y_col
                               average_df = group_df.groupby(x_col)[y_col].mean().reset_index()
                               #print("average_df ", average_df)

                               global_y_min = min(global_y_min, average_df[y_col].min())
                               global_y_max = max(global_y_max, average_df[y_col].max())
                               
                            #self.ax.plot(average_df[x_col], average_df[y_col], label=group_name)
                               line, = self.ax.plot(average_df[x_col], average_df[y_col], label=f"{y_col}-{group_name}", linestyle='-')
                               self.ax.plot(average_df[x_col], average_df[y_col], 'o', color=line.get_color())
                               self.plot_titles.append(f"{y_col} vs {x_col} for {group_col} = {group_name}")
                           else:
                               messagebox.showerror("Error", f"Within group '{group_name}', '{x_col}' has the same value for all rows. Cannot plot.")
                               self.status_label.config(text=f"Error: '{x_col}' has same value for all rows in group '{group_name}'")
                               return

                    if global_y_min < float('inf') and global_y_max > float('-inf'):
                       self.ax.set_ylim(global_y_min, global_y_max)
                    
                    graphics_file_name=""
                    if hasattr(self, 'parent') and hasattr(self.parent, 'csv_data') and hasattr(self.parent.csv_data, 'selected_values'):
                        #comment_text = "\n".join(f"{key}: {value}" for key, value in self.parent.csv_data.selected_values.items())
                        #self.fig.text(0.02, 0.02, comment_text, fontsize=10, verticalalignment='bottom', ha='left')
                        #  selected_values_text = ""
                        not_first=0
                        for key, value in self.parent.csv_data.selected_values.items():
                           cleaned_key = key.strip()
                           cleaned_value = str(value).strip()  # Ensure value is converted to string before stripping
                           if ( not_first == 1 ) : graphics_file_name  +="-"
                           graphics_file_name  +="cleaned_key"
                           graphics_file_name  +="-"
                           graphics_file_name  +="cleaned_value"
                           #selected_values_text += f"{cleaned_key}: {cleaned_value}, "
                           # Remove the trailing comma and space
                           #selected_values_text = selected_values_text.rstrip(', ')
                           #comment_text = selected_values_text

                    self.ax.set_xlabel(x_col)
                    self.ax.set_ylabel(y_col)
                    self.ax.set_title(f"{y_col} vs {x_col} Grouped by {group_col}")
                    self.ax.legend()
                    self.canvas.draw()
                    self.status_label.config(text="Plot generated successfully.")

                except KeyError as e:
                    messagebox.showerror("Error", f"Column not found: {e}")
                    self.status_label.config(text=f"Error: Column not found: {e}")
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred while plotting: {e}")
                    self.status_label.config(text=f"Error: {e}")
            else:
                messagebox.showwarning("Warning", "Please select X, Y, and Group By columns.")
                self.status_label.config(text="Warning: Please select columns.")
        else:
            messagebox.showwarning("Warning", "Please open a CSV file first.")
            self.status_label.config(text="Warning: No data to plot. Open a CSV file.")

    def save_plot(self):
      filename="try.jpeg"
      filename=""
      if hasattr(self, 'parent') and hasattr(self.parent, 'csv_data') and hasattr(self.parent.csv_data, 'selected_values'):
         not_first=0
         for key, value in self.parent.csv_data.selected_values.items():
            cleaned_key = key.strip()
            cleaned_value = str(value).strip()  # Ensure value is converted to string before stripping
            if ( not_first == 1 ) : graphics_file_name  +="-"
            filename  +="cleaned_key"
            filename  +="-"
            filename  +="cleaned_value"
            # Remove the trailing comma and space

      if filename:
        try:
            self.fig.savefig(filename)
            messagebox.showinfo("Success", f"Plot saved as '{filename}'")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving plot: {e}")
        

class DetectorAnalysis(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detector Analysis Application")
        self.geometry("1000x900")  # Increased window size for better usability

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Oscilloscope Data Viewer")
        self.notebook.add(self.tab2, text="CSV Data Filter")
        self.notebook.add(self.tab3, text="Plot Viewer")

        self.df = pd.DataFrame()

        self.scope_view  = ScopeViewApp(self, self.tab1) # Use the class
        self.csv_data    = CSVDataFilterApp(self,self.tab2) # Use the class
        self.analysis    = DataAnalysisApp(self,self.tab3)


def main():
    app = DetectorAnalysis()
    app.mainloop()

if __name__ == "__main__":
    main()
