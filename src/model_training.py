
# =====================================================
# PROJECT 01: CUSTOMER SUBSCRIPTION PREDICTION
# FILE: 04_model_training.py
# PURPOSE: Build and train our PyTorch neural network
# =====================================================

import numpy as np                              # Loads our saved numerical arrays
import torch                                    # Deep learning framework
import torch.nn as nn                           # Neural network layers and loss functions
import torch.optim as optim                     # Optimization algorithms

import matplotlib.pyplot as plt                # Creates training graphs

from torch.utils.data import TensorDataset      # Combines features and labels into a dataset
from torch.utils.data import DataLoader         # Loads data in batches

from pathlib import Path                        # Handles file and folder paths

# -----------------------------------------------------
# 1. SET RANDOM SEEDS
# -----------------------------------------------------

torch.manual_seed(42)                           # Makes PyTorch initialization reproducible
np.random.seed(42)                              # Makes NumPy operations reproducible

# -----------------------------------------------------
# 2. DEFINE PROJECT PATHS
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Find the main project folder

DATA_FOLDER = PROJECT_ROOT / "data"                    # Locate processed data
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"               # Locate graph output folder
MODEL_FOLDER = PROJECT_ROOT / "models"                 # Locate model storage folder

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)       # Create outputs folder if needed
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)        # Create models folder if needed

# -----------------------------------------------------
# 3. SELECT COMPUTING DEVICE
# -----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a GPU if available; otherwise use the CPU
print("\nCOMPUTING DEVICE:", device)

# -----------------------------------------------------
# 4. LOAD THE PROCESSED DATA
# -----------------------------------------------------

data = np.load(DATA_FOLDER / "processed_data.npz")     # Load all saved processed arrays

X_train = data["X_train"]                              # Training customer features
y_train = data["y_train"]                              # Training customer labels

X_val = data["X_val"]                                  # Validation customer features
y_val = data["y_val"]                                  # Validation customer labels

X_test = data["X_test"]                                # Test customer features
y_test = data["y_test"]                                # Test customer labels

# Display the shapes to verify the loaded arrays
print("\nLOADED DATA SHAPES:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

# -----------------------------------------------------
# 5. CONVERT NUMPY ARRAYS INTO PYTORCH TENSORS
# -----------------------------------------------------

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Convert training features into float32 PyTorch tensors

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# Convert training labels into float32 PyTorch tensors

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# Convert validation features into PyTorch tensors

y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
# Convert validation labels into PyTorch tensors

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Convert test features into PyTorch tensors

y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# Convert test labels into PyTorch tensors

# -----------------------------------------------------
# 6. CREATE PYTORCH DATASETS
# -----------------------------------------------------

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# Combine each training customer's features with its correct label

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# Combine validation features with validation labels

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# Combine test features with test labels

# -----------------------------------------------------
# 7. CREATE DATALOADERS
# -----------------------------------------------------

BATCH_SIZE = 256                                # Number of customers processed per batch

train_loader = DataLoader(
    train_dataset,                              # Training dataset
    batch_size=BATCH_SIZE,                      # Process 256 customers at a time
    shuffle=True                                # Randomize training record order
)

val_loader = DataLoader(
    val_dataset,                                # Validation dataset
    batch_size=BATCH_SIZE,                      # Process validation in batches
    shuffle=False                               # Keep validation order unchanged
)

test_loader = DataLoader(
    test_dataset,                               # Test dataset
    batch_size=BATCH_SIZE,                      # Process test records in batches
    shuffle=False                               # Keep test order unchanged
)

# -----------------------------------------------------
# 8. CALCULATE CLASS WEIGHT
# -----------------------------------------------------

number_of_negative = (y_train_tensor == 0).sum()
# Count customers who did not subscribe

number_of_positive = (y_train_tensor == 1).sum()
# Count customers who subscribed

pos_weight_value = number_of_negative / number_of_positive
# Give positive examples more weight because they are less common

pos_weight = torch.tensor(
    [pos_weight_value.item()],
    dtype=torch.float32,
    device=device
)
# Create a PyTorch tensor containing the positive class weight

print("\nCLASS WEIGHT:")
print("Positive class weight:", pos_weight.item())

# -----------------------------------------------------
# 9. BUILD THE NEURAL NETWORK
# -----------------------------------------------------

class CustomerSubscriptionModel(nn.Module):
    # Define our neural network as a PyTorch model

    def __init__(self, input_features):
        # Initialize the layers and model settings

        super().__init__()
        # Initialize the parent nn.Module class

        self.network = nn.Sequential(
            nn.Linear(input_features, 32),       # Transform 50 input features into 32 neurons
            nn.ReLU(),                           # Apply non-linearity
            nn.Dropout(0.2),                     # Randomly drop 20% of activations during training

            nn.Linear(32, 16),                   # Transform 32 neurons into 16 neurons
            nn.ReLU(),                           # Apply another non-linear activation
            nn.Dropout(0.2),                     # Apply dropout again

            nn.Linear(16, 1)                     # Produce one output logit
        )

    def forward(self, x):
        # Define how customer data moves through the network

        return self.network(x)
        # Pass the input through all layers and return the output logit

# -----------------------------------------------------
# 10. CREATE THE MODEL
# -----------------------------------------------------

input_features = X_train.shape[1]
# Get the number of input features automatically (expected: 50)

model = CustomerSubscriptionModel(input_features)
# Create the neural network using the 50 input features

model = model.to(device)
# Move the model to the selected computing device

print("\nMODEL ARCHITECTURE:")
print(model)

# -----------------------------------------------------
# 11. DEFINE LOSS FUNCTION AND OPTIMIZER
# -----------------------------------------------------

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Measure binary classification error while weighting positive examples

optimizer = optim.Adam(
    model.parameters(),                         # Optimize all trainable model parameters
    lr=0.001                                    # Set the learning rate
)

# -----------------------------------------------------
# 12. PREPARE TRAINING SETTINGS
# -----------------------------------------------------

EPOCHS = 50                                     # Maximum number of complete training passes
PATIENCE = 8                                    # Stop if validation loss does not improve for 8 epochs

train_losses = []                               # Store training loss after every epoch
val_losses = []                                 # Store validation loss after every epoch

best_val_loss = float("inf")                    # Start with an infinitely high validation loss
epochs_without_improvement = 0                  # Count epochs without improvement

best_model_path = MODEL_FOLDER / "best_model.pth"
# Define where the best model will be saved

# -----------------------------------------------------
# 13. START THE TRAINING LOOP
# -----------------------------------------------------

for epoch in range(EPOCHS):

    model.train()
    # Activate training behavior, including dropout

    running_train_loss = 0.0
    # Store the total training loss for this epoch

    for batch_features, batch_labels in train_loader:
        # Load one batch of customer features and labels

        batch_features = batch_features.to(device)
        # Move the customer features to the selected device

        batch_labels = batch_labels.to(device)
        # Move the correct labels to the selected device

        optimizer.zero_grad()
        # Clear gradients from the previous training batch

        predictions = model(batch_features)
        # Forward propagation: calculate predicted logits

        loss = loss_function(predictions, batch_labels)
        # Compare predictions with the correct customer labels

        loss.backward()
        # Backpropagation: calculate gradients for model parameters

        optimizer.step()
        # Update weights and biases using Adam

        running_train_loss += loss.item() * len(batch_features)
        # Add this batch's total loss contribution

    epoch_train_loss = running_train_loss / len(train_dataset)
    # Calculate average training loss across all training records

    train_losses.append(epoch_train_loss)
    # Save this epoch's training loss for our graph

    # -------------------------------------------------
    # 14. VALIDATE THE MODEL
    # -------------------------------------------------

    model.eval()
    # Activate evaluation behavior and disable training-time dropout

    running_val_loss = 0.0
    # Store the total validation loss

    with torch.no_grad():
        # Disable gradient calculation because we are not updating weights

        for batch_features, batch_labels in val_loader:
            # Load a validation batch

            batch_features = batch_features.to(device)
            # Move validation features to the selected device

            batch_labels = batch_labels.to(device)
            # Move validation labels to the selected device

            predictions = model(batch_features)
            # Predict using the current trained model

            val_loss = loss_function(predictions, batch_labels)
            # Calculate validation error

            running_val_loss += val_loss.item() * len(batch_features)
            # Add this batch's validation loss contribution

    epoch_val_loss = running_val_loss / len(val_dataset)
    # Calculate average validation loss for this epoch

    val_losses.append(epoch_val_loss)
    # Save validation loss for our graph

    # -------------------------------------------------
    # 15. SAVE THE BEST MODEL
    # -------------------------------------------------

    if epoch_val_loss < best_val_loss:
        # Check whether validation loss has improved

        best_val_loss = epoch_val_loss
        # Update the best validation loss

        torch.save(model.state_dict(), best_model_path)
        # Save the model's learned weights and biases

        epochs_without_improvement = 0
        # Reset the early-stopping counter

    else:
        # Run this when validation loss has not improved

        epochs_without_improvement += 1
        # Increase the number of epochs without improvement

    # -------------------------------------------------
    # 16. DISPLAY TRAINING PROGRESS
    # -------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Training Loss: {epoch_train_loss:.4f} "
        f"Validation Loss: {epoch_val_loss:.4f}"
    )
    # Display the current training and validation losses

    if epochs_without_improvement >= PATIENCE:
        # Check whether validation loss has stopped improving

        print("\nEarly stopping activated.")
        # Tell us why training is ending

        break
        # Stop training to reduce unnecessary epochs

# -----------------------------------------------------
# 17. LOAD THE BEST SAVED MODEL
# -----------------------------------------------------

model.load_state_dict(
    torch.load(best_model_path, map_location=device, weights_only=True)
)
# Restore the model weights from the best validation-loss epoch

model.eval()
# Set the restored model to evaluation mode

# -----------------------------------------------------
# 18. CREATE TRAINING AND VALIDATION LOSS GRAPH
# -----------------------------------------------------

plt.figure(figsize=(10, 6))
# Create a graph area

plt.plot(train_losses, label="Training Loss")
# Draw the training loss curve

plt.plot(val_losses, label="Validation Loss")
# Draw the validation loss curve

plt.title("Training and Validation Loss")
# Give the graph a title

plt.xlabel("Epoch")
# Label the horizontal axis

plt.ylabel("Loss")
# Label the vertical axis

plt.legend()
# Display the names of the two curves

plt.grid(True)
# Add grid lines to make interpretation easier

plt.tight_layout()
# Adjust spacing

plt.savefig(
    OUTPUT_FOLDER / "training_validation_loss.png",
    dpi=300
)
# Save the graph as a high-resolution image

plt.show()
# Display the graph

# -----------------------------------------------------
# 19. FINISH
# -----------------------------------------------------

print("\nMODEL TRAINING COMPLETED!")
print("Best validation loss:", round(best_val_loss, 4))
print("Best model saved at:", best_model_path)
print("Loss graph saved at:", OUTPUT_FOLDER / "training_validation_loss.png")
