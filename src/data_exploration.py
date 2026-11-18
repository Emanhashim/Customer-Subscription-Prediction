
# ============================================
# PROJECT 01: CUSTOMER SUBSCRIPTION PREDICTION
# FILE: 01_data_exploration.py
# ============================================

import pandas as pd                         # Used to load and analyze tabular data
import numpy as np                          # Used for numerical operations
import matplotlib.pyplot as plt             # Used to create graphs
from pathlib import Path                    # Used to manage file and folder paths

# Get the main project folder (one level above src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the folder where our dataset should be located
DATA_FOLDER = PROJECT_ROOT / "data"

# Define the folder where we will save graphs
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

# Create the outputs folder if it does not already exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Search inside data and its subfolders for bank-full.csv
csv_files = list(DATA_FOLDER.rglob("bank-full.csv"))

# Stop with a helpful message if the dataset cannot be found
if not csv_files:
    raise FileNotFoundError(
        "bank-full.csv was not found. Put it inside the project's data folder."
    )

# Select the first matching CSV file
DATA_PATH = csv_files[0]

# Display the exact location of the dataset we found
print("Dataset location:", DATA_PATH)

# Load the CSV file; sep=';' is important because this dataset uses semicolons
df = pd.read_csv(DATA_PATH, sep=";")

# --------------------------------------------
# 1. DISPLAY THE FIRST FIVE RECORDS
# --------------------------------------------

print("\nFIRST FIVE ROWS")

# Show the first five customer records
print(df.head())

# --------------------------------------------
# 2. UNDERSTAND THE DATASET SIZE
# --------------------------------------------

print("\nDATASET SHAPE")

# Display the number of rows and columns
print(df.shape)

# --------------------------------------------
# 3. DISPLAY COLUMN NAMES
# --------------------------------------------

print("\nCOLUMN NAMES")

# Display all column names as a Python list
print(df.columns.tolist())

# --------------------------------------------
# 4. UNDERSTAND COLUMN DATA TYPES
# --------------------------------------------

print("\nDATASET INFORMATION")

# Show each column's data type and non-empty record count
df.info()

# --------------------------------------------
# 5. UNDERSTAND NUMERICAL DATA
# --------------------------------------------

print("\nNUMERICAL SUMMARY")

# Calculate count, mean, standard deviation, minimum and maximum
print(df.describe())

# --------------------------------------------
# 6. CHECK OUR TARGET COLUMN
# --------------------------------------------

print("\nTARGET DISTRIBUTION")

# Count how many customers said yes and how many said no
print(df["y"].value_counts())

print("\nTARGET PERCENTAGES")

# Calculate the percentage of yes and no customers
target_percentages = df["y"].value_counts(normalize=True) * 100

# Display the percentages rounded to two decimal places
print(target_percentages.round(2))

# --------------------------------------------
# 7. CREATE OUR FIRST GRAPH
# --------------------------------------------

# Count the number of customers in each subscription category
subscription_counts = df["y"].value_counts()

# Create a graph area with a width of 8 inches and height of 5 inches
plt.figure(figsize=(8, 5))

# Draw a bar for each subscription category
plt.bar(
    subscription_counts.index,             # Category names: no and yes
    subscription_counts.values              # Number of customers in each category
)

# Give the graph a meaningful title
plt.title("Customer Subscription Distribution")

# Label the horizontal axis
plt.xlabel("Subscription Status")

# Label the vertical axis
plt.ylabel("Number of Customers")

# Add the exact customer count above each bar
for index, value in enumerate(subscription_counts.values):
    plt.text(index, value, str(value), ha="center")

# Adjust spacing so labels and titles fit properly
plt.tight_layout()

# Save the graph as an image inside the outputs folder
plt.savefig(OUTPUT_FOLDER / "target_distribution.png", dpi=300)

# Display the graph on your screen
plt.show()

# --------------------------------------------
# 8. CONFIRM THAT EXPLORATION IS COMPLETE
# --------------------------------------------

print("\nDATA EXPLORATION COMPLETED SUCCESSFULLY!")

# Tell us where the graph was saved
print("Graph saved at:", OUTPUT_FOLDER / "target_distribution.png")
