
# Import the libraries we need.
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Import metrics for evaluating different thresholds.
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

# Locate the project directories.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed_data.npz"
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"

# Choose the available computing device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load our previously processed datasets.
data = np.load(DATA_PATH)

# We use validation data to select the threshold.
X_val = torch.tensor(data["X_val"], dtype=torch.float32).to(device)
y_val = data["y_val"].ravel().astype(int)

# Keep the test data separate until threshold selection is complete.
X_test = torch.tensor(data["X_test"], dtype=torch.float32).to(device)
y_test = data["y_test"].ravel().astype(int)


# Recreate exactly the same architecture used during training.
class CustomerSubscriptionModel(nn.Module):

    def __init__(self, input_features):
        super().__init__()

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
        return self.network(x)


# Initialize the model with the number of input features.
model = CustomerSubscriptionModel(X_val.shape[1]).to(device)

# Load the best weights saved during training.
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

# Disable dropout and use evaluation behavior.
model.eval()


# Get probabilities for validation and test data.
# No gradients are needed because we are only predicting.
with torch.no_grad():

    # Predict validation probabilities.
    val_logits = model(X_val)
    val_probabilities = torch.sigmoid(val_logits).cpu().numpy().ravel()

    # Predict test probabilities.
    # We will not use these to choose the threshold.
    test_logits = model(X_test)
    test_probabilities = torch.sigmoid(test_logits).cpu().numpy().ravel()


# --------------------------------------------------
# 1. FIND THE BEST THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

# Start with a default best F1 score.
best_f1 = -1

# Store the threshold that produces the best validation F1.
best_threshold = 0.5

# Try thresholds from 0.10 to 0.90 in increments of 0.01.
for threshold in np.arange(0.10, 0.91, 0.01):

    # Convert probabilities into predicted classes.
    val_predictions = (val_probabilities >= threshold).astype(int)

    # Calculate precision and recall.
    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )

    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    # Calculate the F1 score for this threshold.
    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    # Keep the threshold if its F1 score is better.
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# Print the selected validation threshold.
print("\nVALIDATION THRESHOLD SELECTION")
print("--------------------------------")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Best validation F1: {best_f1:.4f}")


# --------------------------------------------------
# 2. EVALUATE ON THE TEST DATA
# --------------------------------------------------

# Apply the threshold selected using validation data.
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

# Calculate final test metrics.
test_accuracy = accuracy_score(y_test, test_predictions)

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

# ROC-AUC uses probabilities, not thresholded predictions.
test_roc_auc = roc_auc_score(y_test, test_probabilities)

# Calculate the confusion matrix.
cm = confusion_matrix(y_test, test_predictions)

# Extract true negatives, false positives, false negatives, true positives.
tn, fp, fn, tp = cm.ravel()


# --------------------------------------------------
# 3. DISPLAY THE FINAL RESULTS
# --------------------------------------------------

print("\nFINAL TEST RESULTS WITH SELECTED THRESHOLD")
print("-------------------------------------------")

print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_roc_auc:.4f}")

print("\nCONFUSION MATRIX:")
print(cm)

print("\nTRUE NEGATIVES:", tn)
print("FALSE POSITIVES:", fp)
print("FALSE NEGATIVES:", fn)
print("TRUE POSITIVES:", tp)


# Save the selected threshold for future predictions.
threshold_path = PROJECT_DIR / "models" / "decision_threshold.txt"

with open(threshold_path, "w", encoding="utf-8") as file:
    file.write(str(best_threshold))

print("\nSelected threshold saved to:", threshold_path)
print("THRESHOLD TUNING COMPLETED!")
