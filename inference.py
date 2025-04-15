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
    # Dodajemy potrzebny wymiar
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():

        out = model(image)
        probs = torch.softmax(out, dim=1).squeeze()

        max_prob, pred_class = torch.max(probs, 0)

        if max_prob.item() < threshold or pred_class.item() == 0:
            return "Incorrect die (uncertainty)"
        else:
            return f"Correct die, {pred_class.item()} dots"


