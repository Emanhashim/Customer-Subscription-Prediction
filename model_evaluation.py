
# Import NumPy to load our saved processed datasets.
import numpy as np

# Import PyTorch to load and evaluate our trained neural network.
import torch
import torch.nn as nn

# Import evaluation metrics from scikit-learn.
from sklearn.metrics import (
    accuracy_score,          # Overall percentage of correct predictions.
    precision_score,         # How many predicted subscribers were truly subscribers.
    recall_score,            # How many actual subscribers we successfully identified.
    f1_score,                # Harmonic balance between precision and recall.
    roc_auc_score,           # Measures separation across classification thresholds.
    average_precision_score, # Summarizes precision-recall performance.
    confusion_matrix,        # Counts correct and incorrect predictions.
    classification_report   # Prints precision, recall, and F1 for each class.
)

# Import Matplotlib to visualize the confusion matrix.
import matplotlib.pyplot as plt

# Import pathlib to build file paths reliably.
from pathlib import Path


# --------------------------------------------------
# 1. SET UP PROJECT PATHS
# --------------------------------------------------

# Find the main project folder (one level above src).
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Point to the saved processed dataset.
DATA_PATH = PROJECT_DIR / "data" / "processed_data.npz"

# Point to the best model saved during training.
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"

# Point to the folder where evaluation results will be saved.
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create the outputs folder if it does not already exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SELECT THE COMPUTING DEVICE
# --------------------------------------------------

# Use a GPU if available; otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("COMPUTING DEVICE:", device)


# --------------------------------------------------
# 3. LOAD THE TEST DATA
# --------------------------------------------------

# Open the NumPy archive containing our training, validation, and test arrays.
data = np.load(DATA_PATH)

# Retrieve only the test features and labels.
# We do not use training or validation data to calculate final test metrics.
X_test = data["X_test"]
y_test = data["y_test"]

# Convert the NumPy arrays into PyTorch tensors.
# float32 is the appropriate numerical type for this neural network.
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print("\nTEST DATA SHAPES:")
print("Features:", X_test_tensor.shape)
print("Labels:", y_test_tensor.shape)


# --------------------------------------------------
# 4. RECREATE THE SAME MODEL ARCHITECTURE
# --------------------------------------------------

# This class must match the architecture used during training.
class CustomerSubscriptionModel(nn.Module):

    def __init__(self, input_features):
        # Initialize the parent PyTorch Module.
        super().__init__()

        # Define the neural network layers in sequence.
        self.network = nn.Sequential(

            # First layer: 50 input features become 32 neurons.
            nn.Linear(input_features, 32),

            # ReLU introduces non-linearity.
            nn.ReLU(),

            # Dropout randomly disables 20% of neurons during training.
            nn.Dropout(0.2),

            # Second layer: 32 neurons become 16 neurons.
            nn.Linear(32, 16),

            # Apply another ReLU activation.
            nn.ReLU(),

            # Apply another dropout layer.
            nn.Dropout(0.2),

            # Final layer: produce one output logit.
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Pass the input through all the defined layers.
        return self.network(x)


# Create the model with 50 input features.
model = CustomerSubscriptionModel(input_features=X_test_tensor.shape[1])

# Move the model to the selected computing device.
model = model.to(device)


# --------------------------------------------------
# 5. LOAD THE BEST TRAINED WEIGHTS
# --------------------------------------------------

# Load the weights saved by our training script.
# weights_only=True loads the model's parameter dictionary.
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Apply the saved weights to our recreated model.
model.load_state_dict(state_dict)

# Switch the model into evaluation mode.
# This disables dropout while making predictions.
model.eval()

print("\nBEST TRAINED MODEL LOADED SUCCESSFULLY!")


# --------------------------------------------------
# 6. MAKE PREDICTIONS
# --------------------------------------------------

# Disable gradient calculations because we are not training.
with torch.no_grad():

    # Get the model's raw output logits for each test customer.
    logits = model(X_test_tensor)

    # Convert logits into probabilities between 0 and 1.
    probabilities = torch.sigmoid(logits)

# Move predictions from the device to ordinary CPU NumPy arrays.
y_prob = probabilities.cpu().numpy().ravel()

# Convert actual labels into a one-dimensional NumPy array.
y_true = y_test_tensor.cpu().numpy().ravel().astype(int)

# Use 0.5 as our initial decision threshold.
# Probability >= 0.5 means predict subscriber (1).
# Probability < 0.5 means predict non-subscriber (0).
y_pred = (y_prob >= 0.5).astype(int)


# --------------------------------------------------
# 7. CALCULATE EVALUATION METRICS
# --------------------------------------------------

# Calculate overall accuracy.
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision for the subscriber class (class 1).
precision = precision_score(y_true, y_pred, zero_division=0)

# Calculate recall for the subscriber class (class 1).
recall = recall_score(y_true, y_pred, zero_division=0)

# Calculate the F1-score for the subscriber class.
f1 = f1_score(y_true, y_pred, zero_division=0)

# Calculate ROC-AUC using probabilities, not class predictions.
roc_auc = roc_auc_score(y_true, y_prob)

# Calculate average precision, useful for imbalanced datasets.
average_precision = average_precision_score(y_true, y_prob)


# --------------------------------------------------
# 8. PRINT THE RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("CUSTOMER SUBSCRIPTION MODEL - TEST RESULTS")
print("=" * 50)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Show a more detailed report for both classes.
print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Not subscribed", "Subscribed"],
        zero_division=0
    )
)


# --------------------------------------------------
# 9. CREATE THE CONFUSION MATRIX
# --------------------------------------------------

# Rows represent actual classes; columns represent predicted classes.
cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX:")
print(cm)

# Extract the four confusion matrix values.
tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX EXPLANATION:")
print("True negatives:", tn)  # Correctly predicted non-subscribers.
print("False positives:", fp) # Predicted subscriber, but actually not.
print("False negatives:", fn) # Missed actual subscribers.
print("True positives:", tp)  # Correctly identified subscribers.


# --------------------------------------------------
# 10. VISUALIZE THE CONFUSION MATRIX
# --------------------------------------------------

# Create a figure for the confusion matrix.
plt.figure(figsize=(7, 5))

# Display the matrix as a heatmap.
plt.imshow(cm, interpolation="nearest")

# Add a title and axis labels.
plt.title("Customer Subscription - Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

# Name the two classes.
plt.xticks([0, 1], ["Not subscribed", "Subscribed"])
plt.yticks([0, 1], ["Not subscribed", "Subscribed"])

# Write the actual count inside each cell.
for row in range(2):
    for column in range(2):
        plt.text(
            column,
            row,
            str(cm[row, column]),
            ha="center",
            va="center"
        )

# Adjust spacing so labels fit.
plt.tight_layout()

# Save the chart in the outputs folder.
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

# Display the chart.
plt.show()


# --------------------------------------------------
# 11. SAVE THE METRICS
# --------------------------------------------------

# Store the main metrics in a text file for our project documentation.
metrics_path = OUTPUT_DIR / "test_metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as file:
    file.write("CUSTOMER SUBSCRIPTION MODEL - TEST RESULTS\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    file.write(f"Average Precision: {average_precision:.4f}\n")
    file.write(f"Confusion Matrix:\n{cm}\n")

print("\nEvaluation metrics saved to:", metrics_path)
print("Confusion matrix saved to:", OUTPUT_DIR / "confusion_matrix.png")
print("\nMODEL EVALUATION COMPLETED!")
