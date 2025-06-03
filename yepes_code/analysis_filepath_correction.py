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

df = pd.read_csv('/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/ACLGAD_only.csv')
df['Filename min'] = df['Filename min'].str.replace("mar24", "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/2025-03-24/tek")
df['Filename max'] = df['Filename min'].str.replace("mar24", "/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/2025-03-24/tek")
suffix = "CH1.csv"
df['Filename min'] = df['Filename min'].apply(lambda x: f"{x}{suffix}")
df['Filename max'] = df['Filename max'].apply(lambda x: f"{x}{suffix}")
df.to_csv('/Users/rkfuentes/Documents/md_anderson_analysis/yepes_code/ACLGAD_only.csv')