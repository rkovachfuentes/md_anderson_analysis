
import pandas as pd

# Load the CSV file
input_file  = "lgad-2025-04-15-log.csv"  # Your actual input file name
output_file = "lgad-2025-04-15-log-compaq.csv"  # Updated output file name

# Read the CSV file
df = pd.read_csv(input_file)
df = df.fillna(0)

# Identify columns for grouping
group_columns = [col for col in df.columns if col != "File" and col != "FileNumber"]

# Create a unique group identifier for contiguous rows
df["GroupID"] = (df[group_columns] != df[group_columns].shift()).any(axis=1).cumsum()

# Get the first and last file for each contiguous group
grouped = df.groupby(group_columns + ["GroupID"]).agg(FileMin=("File", "first"), FileMax=("File", "last")).reset_index()

# Sort the result by FileMin
grouped_sorted = grouped.sort_values(["FileMin"])

# Save the processed file
grouped_sorted.to_csv(output_file, index=False)

print(f"Processed file saved as {output_file}")
