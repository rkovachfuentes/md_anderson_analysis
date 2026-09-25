import pandas as pd

# Load the source and target datasets
doses_df = pd.read_csv("/Users/rkfuentes/Documents/phd/research/md_anderson_analysis/yepes_code/dose_csvs/selected_doses_from_mda.csv")
generator_df = pd.read_csv("0922_generator_out.csv")

# Ensure merge key column types and string formats match across both DataFrames
merge_keys = ["beam", "collimator", "pulse"]

for key in merge_keys:
    doses_df[key] = doses_df[key].astype(str).str.strip()
    generator_df[key] = generator_df[key].astype(str).str.strip()

# Select only the key columns and the dose values to substitute
lookup_df = doses_df[merge_keys + ["avg. (nC)"]].drop_duplicates(
    subset=merge_keys
)

# Merge to update the dose column
merged_df = generator_df.merge(lookup_df, on=merge_keys, how="left")

# Replace the existing 'dose' column with values from 'avg (nC)' where a match was found
merged_df["dose"] = merged_df["avg. (nC)"].fillna(merged_df["dose"])

# Drop the temporary lookup column
merged_df = merged_df.drop(columns=["avg. (nC)"])

# Save the updated DataFrame to a new CSV file
merged_df.to_csv("0922_generator_out_updated.csv", index=False)