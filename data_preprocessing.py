
# =====================================================
# PROJECT 01: CUSTOMER SUBSCRIPTION PREDICTION
# FILE: 03_data_preprocessing.py
# PURPOSE: Prepare customer data for neural network training
# =====================================================

import pandas as pd                              # Loads and handles tabular data
import numpy as np                               # Handles numerical arrays
import matplotlib.pyplot as plt                  # Creates graphs
import joblib                                     # Saves our fitted preprocessing pipeline

from pathlib import Path                         # Handles file and folder paths

from sklearn.model_selection import train_test_split  # Splits data into subsets
from sklearn.preprocessing import StandardScaler     # Scales numerical features
from sklearn.preprocessing import OneHotEncoder       # Converts categories into numbers
from sklearn.compose import ColumnTransformer         # Applies different transformations to different columns

# -----------------------------------------------------
# 1. DEFINE PROJECT PATHS
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Find the main project folder

DATA_FOLDER = PROJECT_ROOT / "data"                    # Locate the data folder
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"               # Locate the graph/output folder
MODEL_FOLDER = PROJECT_ROOT / "models"                 # Folder for saved model-related files

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)       # Create outputs if it does not exist
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)        # Create models if it does not exist

# -----------------------------------------------------
# 2. LOAD THE CLEANED DATASET
# -----------------------------------------------------

DATA_PATH = DATA_FOLDER / "bank-cleaned.csv"           # Path to our cleaned CSV file

df = pd.read_csv(DATA_PATH)                            # Load the cleaned customer dataset

print("\nCLEANED DATASET SHAPE:")
print(df.shape)                                        # Show the number of rows and columns

# -----------------------------------------------------
# 3. SEPARATE FEATURES (X) AND TARGET (y)
# -----------------------------------------------------

X = df.drop(columns=["y"])                             # X contains customer information
y = df["y"].map({"no": 0, "yes": 1})                   # Convert no to 0 and yes to 1

# Check that every target value was converted successfully
if y.isnull().any():
    raise ValueError("Unexpected target values found in column y.")

y = y.astype("int64")                                  # Store target labels as integers

print("\nFEATURE DATA SHAPE:")
print(X.shape)                                        # Show how many input records and columns we have

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())                               # Count non-subscribers (0) and subscribers (1)

# -----------------------------------------------------
# 4. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# -----------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()                                     # Find columns containing numbers

categorical_columns = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()                                     # Find columns containing categories/text

print("\nNUMERICAL COLUMNS:")
print(numerical_columns)                              # Display numerical input features

print("\nCATEGORICAL COLUMNS:")
print(categorical_columns)                            # Display categorical input features

# -----------------------------------------------------
# 5. SPLIT DATA INTO TRAINING, VALIDATION AND TEST SETS
# -----------------------------------------------------

# First, reserve 30% of the records for validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X,                                                   # Original customer features
    y,                                                   # Original customer labels
    test_size=0.30,                                      # Reserve 30% for temporary data
    random_state=42,                                     # Make the split reproducible
    stratify=y                                           # Preserve the class proportions
)

# Split the reserved 30% equally into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,                                              # Reserved customer features
    y_temp,                                              # Reserved customer labels
    test_size=0.50,                                      # Half for testing, half for validation
    random_state=42,                                     # Keep the split reproducible
    stratify=y_temp                                      # Preserve the class proportions
)

print("\nDATASET SPLIT SIZES:")
print("Training:", len(X_train))                        # Number of training records
print("Validation:", len(X_val))                        # Number of validation records
print("Testing:", len(X_test))                          # Number of test records

# -----------------------------------------------------
# 6. CREATE THE PREPROCESSING PIPELINE
# -----------------------------------------------------

# Create an encoder that converts categories into separate binary columns
# sparse_output=False produces a normal dense array for PyTorch
try:
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",                         # Safely handle unseen categories
        sparse_output=False                              # Return a dense numerical array
    )
except TypeError:
    # Compatibility with older scikit-learn versions
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",                         # Safely handle unseen categories
        sparse=False                                     # Older parameter name for dense output
    )

# Create a transformer that handles numerical and categorical features differently
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",                                 # Name of the numerical transformation
            StandardScaler(),                            # Standardize numerical features
            numerical_columns                            # Apply it to numerical columns
        ),
      
        (
            "categorical",                               # Name of the categorical transformation
            categorical_encoder,                         # Apply One-Hot Encoding
            categorical_columns                           # Apply it to categorical columns
        )
    ]
)

# -----------------------------------------------------
# 7. FIT ONLY ON TRAINING DATA
# -----------------------------------------------------

# Learn scaling values and category mappings from training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the already-learned transformations to validation data
X_val_processed = preprocessor.transform(X_val)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

# -----------------------------------------------------
# 8. CONVERT FEATURES AND LABELS TO NUMPY FLOAT ARRAYS
# -----------------------------------------------------

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)                                                     # Convert training features to float32

X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)                                                     # Convert validation features to float32

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)                                                     # Convert test features to float32

y_train_processed = y_train.to_numpy(
    dtype=np.float32
).reshape(-1, 1)                                       # Convert training labels to column shape

y_val_processed = y_val.to_numpy(
    dtype=np.float32
).reshape(-1, 1)                                       # Convert validation labels to column shape

y_test_processed = y_test.to_numpy(
    dtype=np.float32
).reshape(-1, 1)                                       # Convert test labels to column shape

# -----------------------------------------------------
# 9. DISPLAY THE PROCESSED DATA SHAPES
# -----------------------------------------------------

print("\nPROCESSED DATA SHAPES:")
print("Training features:", X_train_processed.shape)
print("Training labels:", y_train_processed.shape)
print("Validation features:", X_val_processed.shape)
print("Validation labels:", y_val_processed.shape)
print("Test features:", X_test_processed.shape)
print("Test labels:", y_test_processed.shape)

# -----------------------------------------------------
# 10. CHECK SUBSCRIBER PERCENTAGES IN EACH SPLIT
# -----------------------------------------------------

train_yes_percentage = y_train_processed.mean() * 100  # Percentage of subscribers in training
val_yes_percentage = y_val_processed.mean() * 100      # Percentage of subscribers in validation
test_yes_percentage = y_test_processed.mean() * 100    # Percentage of subscribers in testing

print("\nSUBSCRIBER PERCENTAGES:")
print(f"Training: {train_yes_percentage:.2f}%")
print(f"Validation: {val_yes_percentage:.2f}%")
print(f"Testing: {test_yes_percentage:.2f}%")

# -----------------------------------------------------
# 11. CREATE A GRAPH TO COMPARE THE DATA SPLITS
# -----------------------------------------------------

split_names = ["Training", "Validation", "Testing"]    # Names displayed on the graph

subscriber_percentages = [
    train_yes_percentage,                               # Training subscriber percentage
    val_yes_percentage,                                 # Validation subscriber percentage
    test_yes_percentage                                 # Test subscriber percentage
]

plt.figure(figsize=(8, 5))                              # Create a graph area

plt.bar(
    split_names,                                        # X-axis: dataset split names
    subscriber_percentages                              # Y-axis: subscriber percentages
)

plt.title("Subscriber Percentage Across Data Splits")  # Graph title
plt.xlabel("Dataset Split")                             # X-axis label
plt.ylabel("Subscribers (%)")                           # Y-axis label

plt.ylim(0, 15)                                         # Set a readable vertical scale

for index, value in enumerate(subscriber_percentages):
    plt.text(
        index,                                          # Position above each bar
        value + 0.3,                                    # Slightly above the bar
        f"{value:.2f}%",                                # Display percentage
        ha="center"                                     # Center the text
    )

plt.tight_layout()                                      # Adjust graph spacing

plt.savefig(
    OUTPUT_FOLDER / "subscriber_split_distribution.png",
    dpi=300                                             # Save a high-resolution graph
)

plt.show()                                              # Display graph in a separate window

# -----------------------------------------------------
# 12. SAVE THE PREPROCESSING PIPELINE
# -----------------------------------------------------

joblib.dump(
    preprocessor,                                       # The fitted encoder and scaler
    MODEL_FOLDER / "preprocessor.joblib"                # Destination file
)

# -----------------------------------------------------
# 13. SAVE ALL PROCESSED DATA FOR PYTORCH
# -----------------------------------------------------

np.savez_compressed(
    DATA_FOLDER / "processed_data.npz",                 # File containing all processed arrays
    X_train=X_train_processed,                          # Training customer features
    y_train=y_train_processed,                          # Training customer labels
    X_val=X_val_processed,                              # Validation customer features
    y_val=y_val_processed,                              # Validation customer labels
    X_test=X_test_processed,                            # Test customer features
    y_test=y_test_processed                             # Test customer labels
)

# Save the names of all features created by preprocessing
feature_names = preprocessor.get_feature_names_out()

with open(OUTPUT_FOLDER / "feature_names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(feature_names.tolist()))

# -----------------------------------------------------
# 14. FINISH
# -----------------------------------------------------

print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY!")

print("Saved processed data:", DATA_FOLDER / "processed_data.npz")
print("Saved preprocessing pipeline:", MODEL_FOLDER / "preprocessor.joblib")
print("Saved feature names:", OUTPUT_FOLDER / "feature_names.txt")
print("Number of final input features:", X_train_processed.shape[1])
