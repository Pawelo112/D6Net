# main.py
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# Import custom modules
from data import DiceDataset, DiceBinaryDataset, DiceDotsDataset, get_transforms
from models import CNN, BinaryCNN
from train_eval import train_model, evaluate_model
from inference import  plot_multiple_confusion_matrices
from visualization import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Data Preparation
transform = get_transforms()
dataset = DiceDataset('dice_ext.csv', transform=transform)

# Split data into training and testing using stratified split based on column 0
labels = dataset.data.iloc[:, 0]
train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=labels)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
# Creating train and test data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. Approach 1: Single Model (7 classes)
num_classes = 7
model1 = CNN(num_classes=num_classes).to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn1 = torch.nn.CrossEntropyLoss()

epochs = 15
patience = 3
best_val_loss = float('inf')
patience_counter = 0

# Lists for visualization (model 1)
train_loss_list_1 = []
train_acc_list_1 = []
val_loss_list_1 = []
val_acc_list_1 = []

print("Training Approach 1 (Single Model, 7 classes):")
for epoch in range(epochs):
    train_loss, train_acc = train_model(model1, optimizer1, loss_fn1, train_loader, device)
    val_loss, val_acc = evaluate_model(model1, loss_fn1, test_loader, device)

    # Early stopping, if there is zero improvement on validation data
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Stopping training at epoch {epoch + 1} due to no improvement on validation.")
        break

    # Update lists
    train_loss_list_1.append(train_loss)
    train_acc_list_1.append(train_acc)
    val_loss_list_1.append(val_loss)
    val_acc_list_1.append(val_acc)

    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " +
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot training and validation curves for model 1
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_list_1) + 1), train_loss_list_1, label="Train Loss", marker='o')
plt.plot(range(1, len(val_loss_list_1) + 1), val_loss_list_1, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Approach 1)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_acc_list_1) + 1), train_acc_list_1, label="Train Accuracy", marker='o', color='blue')
plt.plot(range(1, len(val_acc_list_1) + 1), val_acc_list_1, label="Val Accuracy", marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy (Approach 1)")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize sample predictions for Approach 1 on 10 random test images
visualize_predictions_model1(test_loader, model1, device, num_samples=10)


# 3. Approach 2: Ensemble of Two Models

# Data preparation for binary model
binary_dataset = DiceBinaryDataset('dice_ext.csv', transform=transform)
labels_bin = binary_dataset.data.iloc[:, 0]
train_indices_bin, test_indices_bin = train_test_split(np.arange(len(binary_dataset)), test_size=0.2,
                                                       stratify=labels_bin)
train_bin_dataset = Subset(binary_dataset, train_indices_bin)
test_bin_dataset = Subset(binary_dataset, test_indices_bin)

train_bin_loader = DataLoader(train_bin_dataset, batch_size=batch_size, shuffle=True)
test_bin_loader = DataLoader(test_bin_dataset, batch_size=batch_size, shuffle=False)

binary_model = BinaryCNN().to(device)
optimizer_bin = optim.Adam(binary_model.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn_bin = torch.nn.CrossEntropyLoss()

# Statistics lists for binary model charts
train_loss_list_bin = []
train_acc_list_bin = []
val_loss_list_bin = []
val_acc_list_bin = []

# Training the model
epochs = 15
final_epochs = epochs

# Early stopping mechanism
patience = 3  # Number of epochs with no improvement in validation loss
best_val_loss = float('inf')
patience_counter = 0

# Binary model training
print("\nTraining the binary model (approach 2):\n")
for epoch in range(epochs):
    # Compute loss and accuracy
    train_loss, train_acc = train_model(binary_model, optimizer_bin, loss_fn_bin, train_bin_loader, device)
    val_loss, val_acc = evaluate_model(binary_model, loss_fn_bin, test_bin_loader, device)

    # Early stopping if no improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Training stopped at epoch {epoch+1} due to no improvement in validation loss.")
        final_epochs = epoch
        break

    # Update lists
    train_loss_list_bin.append(train_loss)
    train_acc_list_bin.append(train_acc)
    val_loss_list_bin.append(val_loss)
    val_acc_list_bin.append(val_acc)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# Visualization of results for the binary model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, final_epochs+1), train_loss_list_bin, label="Train Loss", marker='o')
plt.plot(range(1, final_epochs+1), val_loss_list_bin, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (approach 2)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, final_epochs+1), train_acc_list_bin, label="Train Accuracy", marker='o', color='blue')
plt.plot(range(1, final_epochs+1), val_acc_list_bin, label="Val Accuracy", marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy (approach 2)")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize sample predictions for Binary model on 10 random test images
visualize_predictions_binarymodel(test_bin_loader, binary_model, device, num_samples=10)

# Preparing the DiceDotsDataset.
dots_dataset = DiceDotsDataset('dice_ext.csv', transform=transform)
labels_dots = dots_dataset.data.iloc[:, 0]
train_indices_dots, test_indices_dots = train_test_split(np.arange(len(dots_dataset)), test_size=0.2,
                                                         stratify=labels_dots)
train_dots_dataset = Subset(dots_dataset, train_indices_dots)
test_dots_dataset = Subset(dots_dataset, test_indices_dots)

train_dots_loader = DataLoader(train_dots_dataset, batch_size=batch_size, shuffle=True)
test_dots_loader = DataLoader(test_dots_dataset, batch_size=batch_size, shuffle=False)

dots_model = CNN(num_classes=6).to(device)
optimizer_dots = optim.Adam(dots_model.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn_dots = torch.nn.CrossEntropyLoss()

# Statistics lists for dots model charts
train_loss_list_dots = []
train_acc_list_dots = []
val_loss_list_dots = []
val_acc_list_dots = []

epochs = 15
final_epochs = epochs

# Early stopping mechanism
patience = 3  # Number of epochs with no improvement in validation loss
best_val_loss = float('inf')
patience_counter = 0

print("\nTraining the dot-counting model (approach 2):\n")
for epoch in range(epochs):
    # Compute loss and accuracy
    train_loss, train_acc = train_model(dots_model, optimizer_dots, loss_fn_dots, train_dots_loader, device)
    val_loss, val_acc = evaluate_model(dots_model, loss_fn_dots, test_dots_loader, device)

    # Early stopping if validation loss does not improve
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Training stopped at epoch {epoch+1} due to no improvement in validation loss.")
        final_epochs = epoch
        break

    # Update lists
    train_loss_list_dots.append(train_loss)
    train_acc_list_dots.append(train_acc)
    val_loss_list_dots.append(val_loss)
    val_acc_list_dots.append(val_acc)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Visualization of training results for the dot-counting model (approach 2)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, final_epochs+1), train_loss_list_dots, label="Train Loss", marker='o')
plt.plot(range(1, final_epochs+1), val_loss_list_dots, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve - Dot-Counting Model (Approach 2)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, final_epochs+1), train_acc_list_dots, label="Train Accuracy", marker='o', color='blue')
plt.plot(range(1, final_epochs+1), val_acc_list_dots, label="Validation Accuracy", marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve - Dot-Counting Model (Approach 2)")
plt.legend()

plt.tight_layout()
plt.show()

# Visualize sample predictions for Dots model on 10 random test images
visualize_predictions_dotsmodel(test_dots_loader, dots_model, device, num_samples=10)

# Visualize sample predictions for ensemble approach
visualize_predictions_ensemble(test_loader, binary_model, dots_model, device, num_samples=10)


# 4. Approach 3: Uncertainty Model (using model1)
# For uncertainty-based predictions, we use the model1 from Approach 1.

# Visualize sample predictions for model with uncertainty
visualize_predictions_uncertain(test_loader, model1, device, num_samples=10)


# 5. Final visualization and confusion matrices for all approaches
visualize_predictions(test_loader, model1, binary_model, dots_model, device, num_samples=10)

plot_multiple_confusion_matrices(model1=model1, binary_model=binary_model, dots_model=dots_model,
                                 data_loader=test_loader, test_loader=test_loader, device=device, num_classes=7,
                                 threshold=0.9)
