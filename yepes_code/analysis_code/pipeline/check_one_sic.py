import pandas as pd
import numpy as np

df = pd.read_csv("swapped_dose_hv.csv")
print("HV 40.0 | Beam 191V ")
df = df[df['HV'] == 40.0]
# print(df)
df = df[df['Beam'] == 'Electron 191V']
df_1 = df[df['Pulse'] == 1.0]['Area'].mean()
df_2 = df[df['Pulse'] == 2.0]['Area'].mean()
df_3 = df[df['Pulse'] == 3.0]['Area'].mean()
df_2_scale = df_2/df_1
df_3_scale = df_3/df_1
print("Pulse | Mean Area | Scale factor relative to 1.0us pulse")
print(f"1.0 | {df_1} | 1.0")
print(f"2.0 | {df_2} | {df_2_scale}")
print(f"3.0 | {df_3} | {df_3_scale}")