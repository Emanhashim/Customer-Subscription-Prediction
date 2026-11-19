
# Import the libraries needed for prediction.
import torch
import torch.nn as nn
import pandas as pd
import joblib

# Import pathlib for reliable project file paths.
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT FILE LOCATIONS
# --------------------------------------------------

# Locate the main project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the saved preprocessing pipeline.
PREPROCESSOR_PATH = PROJECT_DIR / "models" / "preprocessor.joblib"

# Locate the trained neural network weights.
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"

# Locate the selected decision threshold.
THRESHOLD_PATH = PROJECT_DIR / "models" / "decision_threshold.txt"


# --------------------------------------------------
# 2. SELECT THE COMPUTING DEVICE
# --------------------------------------------------

# Use GPU if available; otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 3. LOAD THE PREPROCESSING PIPELINE
# --------------------------------------------------

# Load the same scaler and encoder fitted during training.
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# 4. LOAD THE DECISION THRESHOLD
# --------------------------------------------------

# Read the threshold selected using validation data.
with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    decision_threshold = float(file.read().strip())


# --------------------------------------------------
# 5. DEFINE THE SAME NEURAL NETWORK ARCHITECTURE
# --------------------------------------------------

class CustomerSubscriptionModel(nn.Module):

    def __init__(self, input_features):
        # Initialize the parent PyTorch class.
        super().__init__()

        # Recreate the architecture used during training.
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Pass input through the neural network.
        return self.network(x)


# --------------------------------------------------
# 6. CREATE AND LOAD THE TRAINED MODEL
# --------------------------------------------------

# Our preprocessing pipeline produces 50 features.
model = CustomerSubscriptionModel(input_features=50).to(device)

# Load the best weights saved during training.
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

# Switch off dropout for prediction.
model.eval()


# --------------------------------------------------
# 7. RECEIVE NEW CUSTOMER INFORMATION
# --------------------------------------------------

# Example customer information.
# These are original raw features, before encoding and scaling.
customer = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

# Convert the dictionary into a pandas DataFrame.
# The preprocessing pipeline expects a DataFrame with these column names.
customer_df = pd.DataFrame([customer])


# --------------------------------------------------
# 8. PREPROCESS THE NEW CUSTOMER
# --------------------------------------------------

# Apply the already-fitted preprocessing pipeline.
# Do not call fit() or fit_transform() here.
processed_customer = preprocessor.transform(customer_df)

# Convert the processed features into a PyTorch tensor.
customer_tensor = torch.tensor(
    processed_customer,
    dtype=torch.float32
).to(device)


# --------------------------------------------------
# 9. GENERATE THE SUBSCRIPTION PROBABILITY
# --------------------------------------------------

# No gradients are required for inference.
with torch.no_grad():

    # Get the raw output logit from the neural network.
    logit = model(customer_tensor)

    # Convert the logit into a probability between 0 and 1.
    probability = torch.sigmoid(logit).item()


# --------------------------------------------------
# 10. MAKE THE FINAL CLASSIFICATION
# --------------------------------------------------

# Compare the probability against our selected threshold.
prediction = (
    "Likely to subscribe"
    if probability >= decision_threshold
    else "Unlikely to subscribe"
)


# --------------------------------------------------
# 11. DISPLAY THE RESULT
# --------------------------------------------------

print("\n" + "=" * 45)
print("CUSTOMER SUBSCRIPTION PREDICTION")
print("=" * 45)

print("Customer age:", customer["age"])
print("Customer job:", customer["job"])

print(f"\nSubscription probability: {probability:.2%}")
print(f"Decision threshold: {decision_threshold:.2f}")
print("Prediction:", prediction)

print("\nPREDICTION COMPLETED!")
