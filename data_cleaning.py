
# ============================================
# PROJECT 01: CUSTOMER SUBSCRIPTION PREDICTION
# FILE: 02_data_cleaning.py
# ============================================

import pandas as pd                         # Loads and analyzes our dataset
import matplotlib.pyplot as plt             # Creates data quality graphs
from pathlib import Path                    # Handles project file paths

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the data and output folders
DATA_FOLDER = PROJECT_ROOT / "data"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

# Make sure the output folder exists
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Find the original dataset, including possible subfolders
csv_files = list(DATA_FOLDER.rglob("bank-full.csv"))

# Stop if the original dataset cannot be found
if not csv_files:
    raise FileNotFoundError("Please put bank-full.csv inside the data folder.")

# Select the original dataset
DATA_PATH = csv_files[0]

# Load the original dataset using its semicolon separator
df = pd.read_csv(DATA_PATH, sep=";")

# --------------------------------------------
# 1. CHECK THE ORIGINAL DATASET
# --------------------------------------------

print("\nORIGINAL DATASET SHAPE")

# Show the original number of rows and columns
print(df.shape)

# --------------------------------------------
# 2. CHECK MISSING VALUES
# --------------------------------------------

print("\nMISSING VALUES PER COLUMN")

# Count missing values in every column
missing_values = df.isnull().sum()

# Display the missing value counts
print(missing_values)

# Calculate the total number of missing values
print("\nTOTAL MISSING VALUES:", missing_values.sum())

# --------------------------------------------
# 3. CHECK DUPLICATE RECORDS
# --------------------------------------------

print("\nDUPLICATE RECORDS")

# Count rows that are exact duplicates of earlier rows
duplicate_count = df.duplicated().sum()

# Display the number of duplicate records
print("Number of duplicates:", duplicate_count)

# Remove duplicate rows and create a separate cleaned DataFrame
df_cleaned = df.drop_duplicates().copy()

# --------------------------------------------
# 4. CHECK UNKNOWN CATEGORICAL VALUES
# --------------------------------------------

print("\nUNKNOWN VALUES PER COLUMN")

# Count how many times the word 'unknown' appears in each column
unknown_counts = (df_cleaned == "unknown").sum()

# Display the counts
print(unknown_counts[unknown_counts > 0])

# We keep 'unknown' because it is meaningful information
# It tells us that a particular customer detail was not recorded

# --------------------------------------------
# 5. REMOVE DATA LEAKAGE
# --------------------------------------------

print("\nREMOVING DATA LEAKAGE")

# Check whether the duration column exists before removing it
if "duration" in df_cleaned.columns:

    # Remove call duration because it is unavailable before the call
    df_cleaned = df_cleaned.drop(columns=["duration"])

    # Confirm that the column has been removed
    print("Removed duration column.")

else:
    print("Duration column was already removed.")

# --------------------------------------------
# 6. SAVE THE CLEANED DATASET
# --------------------------------------------

# Define the destination for our cleaned dataset
CLEANED_DATA_PATH = DATA_FOLDER / "bank-cleaned.csv"

# Save the cleaned data as a CSV file
df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

# --------------------------------------------
# 7. CHECK THE CLEANED DATASET
# --------------------------------------------

print("\nCLEANED DATASET SHAPE")

# Display the new number of rows and columns
print(df_cleaned.shape)

print("\nCLEANED DATASET COLUMNS")

# Display the remaining column names
print(df_cleaned.columns.tolist())

print("\nMISSING VALUES AFTER CLEANING")

# Confirm the number of missing values after cleaning
print(df_cleaned.isnull().sum().sum())

print("\nDUPLICATES AFTER CLEANING")

# Confirm that duplicate records have been removed
print(df_cleaned.duplicated().sum())

# --------------------------------------------
# 8. CREATE A MISSING VALUES GRAPH
# --------------------------------------------

# Create a figure for missing-value counts
plt.figure(figsize=(12, 5))

# Draw one bar for each column's missing-value count
plt.bar(missing_values.index, missing_values.values)

# Give the graph a title
plt.title("Missing Values Before Cleaning")

# Label the horizontal axis
plt.xlabel("Dataset Columns")

# Label the vertical axis
plt.ylabel("Number of Missing Values")

# Rotate column names so they are easier to read
plt.xticks(rotation=45, ha="right")

# Adjust spacing around the graph
plt.tight_layout()

# Save the graph in our outputs folder
plt.savefig(OUTPUT_FOLDER / "missing_values.png", dpi=300)

# Display the graph
plt.show()

# --------------------------------------------
# 9. FINISH
# --------------------------------------------

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")

# Display where the cleaned dataset was saved
print("Cleaned dataset:", CLEANED_DATA_PATH)

# Display the final number of records
print("Final records:", len(df_cleaned))
