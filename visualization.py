# visualization.py
import random
import torch
import matplotlib.pyplot as plt
from inference import ensemble_inference, uncertain_inference


def visualize_predictions_model1(test_loader, model1, device, num_samples=5):
    """
    Display predictions from model1 on a few random samples from the test set.

    Shows:
    - Ground truth label
    - Prediction from model1
    """

    model1.eval()

    # Collect all images and labels
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    cols = num_samples // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = all_images[indices[i]]
        label = all_labels[indices[i]].item()

        out = model1(img.unsqueeze(0).to(device))
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        result = "Incorrect" if pred == 0 else f"{pred} dots"

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {result}", fontsize=20)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions_binarymodel(test_loader, binary_model, device, num_samples=5):
    """
    Display predictions from the binary model on a few test samples.

    Shows:
    - Ground truth label
    - Prediction: valid (1) or invalid (0)
    """

    binary_model.eval()

    # Collecting all data from test_loader
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    cols = num_samples // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = all_images[indices[i]]
        label = all_labels[indices[i]].item()

        out = binary_model(img.unsqueeze(0).to(device))
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        result = "Incorrect (0)" if pred == 0 else "Correct (1)"

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {result}", fontsize=20)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions_dotsmodel(test_loader, dots_model, device, num_samples=5):
    """
    Display predictions from the dots model (predicts number of dots on die).

    Shows:
    - Ground truth number of dots
    - Model prediction
    """

    dots_model.eval()

    # Collecting all data from test_loader
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    cols = num_samples // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = all_images[indices[i]]
        label = all_labels[indices[i]].item()

        out = dots_model(img.unsqueeze(0).to(device))
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        result = f'{pred + 1} dots'

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label + 1} oczka\nPred: {result}", fontsize=20)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions_ensemble(test_loader, binary_model, dots_model, device, num_samples=5):
    """
    Display predictions using an ensemble of binary and dots model.

    Shows:
    - Ground truth class
    - Prediction using ensemble approach
    """

    # Collecting all data from test_loader
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    cols = num_samples // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = all_images[indices[i]]
        label = all_labels[indices[i]].item()

        result = ensemble_inference(img, binary_model, dots_model, device)

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {result}", fontsize=16)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions_uncertain(test_loader, model1, device, num_samples=5):
    """
    Display predictions with uncertainty estimation using model1.

    Shows:
    - Ground truth label
    - Prediction with uncertainty check
    """

    model1.eval()

    # Collecting all data from test_loader
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    cols = num_samples // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        img = all_images[indices[i]]
        label = all_labels[indices[i]].item()

        result = uncertain_inference(img, model1, device)

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {result}", fontsize=16)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions(test_loader, model1, binary_model, dots_model, device, num_samples=5):
    """
     Display predictions using all three approaches on the same samples:
    - Approach 1: Single model (model1)
    - Approach 2: Ensemble (binary + dots model)
    - Approach 3: Uncertainty-aware prediction

    Shows:
    - Ground truth
    - Predictions from all three approaches
    """

    # Collecting all data from test_loader
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.extend(images)
        all_labels.extend(labels)

    # Random images indices
    indices = random.sample(range(len(all_images)), num_samples)

    rows = 2
    cols = num_samples // 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    axes = axes.flatten()

    model1.eval()
    for i, idx in enumerate(indices):
        img = all_images[idx]
        label = all_labels[idx].item()

        # Approach 1
        out1 = model1(img.unsqueeze(0).to(device))
        prob1 = torch.softmax(out1, dim=1)
        pred1 = torch.argmax(prob1, dim=1).item()
        result1 = "Incorrect" if pred1 == 0 else f"{pred1} dots"

        # Approach 2
        result2 = ensemble_inference(img, binary_model, dots_model, device)

        # Approach 3
        result3 = uncertain_inference(img, model1, device)

        axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}\nP1: {result1}\nP2: {result2}\nP3: {result3}", fontsize=15)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
