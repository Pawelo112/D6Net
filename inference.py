# inference.py
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Inference function for approach 2 (ensemble):
def ensemble_inference(image, binary_model, dots_model, device):
    """
    Ensemble inference: first uses the binary model to decide if the die is correct.
    If predicted as correct, the dots model predicts the dots count (adding 1 to map from 0-5 to 1-6).
    If not, the image is classified as incorrect die.
    """

    # Adding the additional dimension needed for models
    image = image.to(device).unsqueeze(0)
    binary_model.eval()
    dots_model.eval()

    with torch.no_grad():
        out_bin = binary_model(image)
        prob_bin = torch.softmax(out_bin, dim=1)
        pred_bin = torch.argmax(prob_bin, dim=1).item()

        if pred_bin == 0:
            return "Incorrect die"
        else:
            out_dots = dots_model(image)
            prob_dots = torch.softmax(out_dots, dim=1)
            pred_dots = torch.argmax(prob_dots, dim=1).item() + 1  # Mapping to 1-6
            return f"Correct die, {pred_dots} dots"


def uncertain_inference(image, model, device, threshold=0.9):
    """
    Inference with uncertainty estimation:
    If the maximum probability is below the threshold or the predicted class is 0,
    the die is considered as incorrect. Otherwise, returns the predicted dot count.
    """
    model.eval()
    # Adding the additional dimension
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():

        out = model(image)
        probs = torch.softmax(out, dim=1).squeeze()

        max_prob, pred_class = torch.max(probs, 0)

        if max_prob.item() < threshold or pred_class.item() == 0:
            return "Incorrect die (uncertainty)"
        else:
            return f"Correct die, {pred_class.item()} dots"


def confusion_matrix_for_approach1(model, loader, device, num_classes):
    all_labels = []
    all_preds = []
    errors = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Prediction
            out = model(images)
            _, preds = torch.max(out, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Error counting
            errors += (preds != labels).sum().item()  # Adding up incorrect predictions

    # Calculating confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm, errors


def confusion_matrix_for_approach2(binary_model, dots_model, loader, device):
    all_labels = []
    all_preds = []
    errors = 0

    binary_model.eval()
    dots_model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Binary model prediction
            out_bin = binary_model(images)
            prob_bin = torch.softmax(out_bin, dim=1)
            pred_bin = torch.argmax(prob_bin, dim=1)

            # For indices where the binary model predicts 1 (correct), get dot prediction and add 1.
            pred_dots = pred_bin.clone()
            pred_dots[pred_bin == 1] = dots_model(images[pred_bin == 1]).argmax(
                dim=1) + 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_dots.cpu().numpy())

            # Error counting
            errors += (pred_dots != labels).sum().item()  # Adding up incorrect predictions

    # Calculating confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    return cm, errors


def calculate_confusion_matrix_with_errors(data_loader, model, device, threshold=0.9):
    """
    Calculates the confusion matrix using the provided data loader.
    Additionally, counts the total errors with the following logic:
    - If the model is uncertain (max probability < threshold or prediction equals 0)
      and the true label is not 0, count as an error.
    - Otherwise, any mismatch between prediction and true label is counted.
    """
    true_labels = []
    pred_labels = []
    errors = 0  # Variable for counting false predictions

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Prediction
            out = model(images)
            probs = torch.softmax(out, dim=1)

            max_prob, pred_class = torch.max(probs, 1)

            for i in range(len(pred_class)):
                # Checking if the model is uncertain (prediction 0 or probability below the threshold)
                if max_prob[i].item() < threshold or pred_class[i].item() == 0:
                    # If the true class is not 0, we treat it as an error
                    if labels[i].item() != 0:
                        true_labels.append(labels[i].item())
                        pred_labels.append(0)  # "Incorrect die"
                        errors += 1  # Error because the model thought it was an incorrect die, but it was a correct die
                    else:
                        true_labels.append(labels[i].item())
                        pred_labels.append(0)  # Model considered this as an "incorrect die" and predicted 0
                else:
                    # If the model predicts a class (certain prediction), we check if it is correct
                    true_labels.append(labels[i].item())
                    pred_labels.append(pred_class[i].item())

                    if pred_class[i].item() != labels[i].item():
                        errors += 1  # Error because the model predicted the wrong number of dots

    # Calculating confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    return cm, errors


def plot_multiple_confusion_matrices(model1, binary_model, dots_model, data_loader, test_loader, device, num_classes=7,
                                     threshold=0.9):
    # Calculation of the confusion matrix and error sum for 3 approaches
    cm_1, errors_1 = confusion_matrix_for_approach1(model1, test_loader, device, num_classes)
    cm_2, errors_2 = confusion_matrix_for_approach2(binary_model, dots_model, test_loader, device)
    cm_3, errors_3 = calculate_confusion_matrix_with_errors(data_loader, model1, device, threshold)

    # Chart preparations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Approach 1 - One model
    sns.heatmap(cm_1, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)), ax=axes[0])
    axes[0].set_title("Approach 1 (One Model)")
    axes[0].set_xlabel("Predicted Labels")
    axes[0].set_ylabel("True Labels")

    # Approach 2 - Ensemble of two models
    sns.heatmap(cm_2, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)), ax=axes[1])
    axes[1].set_title("Approach 2 (Ensemble)")
    axes[1].set_xlabel("Predicted Labels")
    axes[1].set_ylabel("True Labels")

    # Approach 3 - Uncertainty Checking Model
    sns.heatmap(cm_3, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(num_classes)),
                yticklabels=list(range(num_classes)), ax=axes[2])
    axes[2].set_title("Approach 3 (Uncertainty)")
    axes[2].set_xlabel("Predicted Labels")
    axes[2].set_ylabel("True Labels")

    # Chart visualization
    plt.tight_layout()
    plt.show()

    # Printing the sum of errors for each attempt
    print(f"Total errors for Approach 1 (One Model): {errors_1}/{len(test_loader.dataset)}")
    print(f"Total errors for Approach 2 (Ensemble): {errors_2}/{len(test_loader.dataset)}")
    print(f"Total errors for Approach 3 (Uncertainty): {errors_3}/{len(test_loader.dataset)}")
