# train_eval.py
import torch


def train_model(model, optimizer, loss_fn, loader, device):
    """
    Training loop: sets the model to train mode, computes loss,
    backpropagates and updates weights. Returns the average loss and accuracy.
    """

    # Setting model to train mode
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Prediction
        out = model(images)
        # Loss computing
        loss = loss_fn(out, labels)

        # Setting gradients to zero
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Updating weights
        optimizer.step()

        loss_sum += loss.item() * images.size(0)

        # Accuracy calculation
        _, preds = torch.max(out, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return loss_sum / len(loader.dataset), correct / total


def evaluate_model(model, loss_fn, loader, device):
    """
    Evaluation loop: sets the model to eval mode, computes loss and accuracy
    on given dataset.
    """

    # Setting model to eval mode (disabling dropout)
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    # Turning off gradients to speed up calculations
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Prediction
            out = model(images)
            # Loss computing
            loss = loss_fn(out, labels)
            loss_sum += loss.item() * images.size(0)

            # Accuracy calculation
            _, preds = torch.max(out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return loss_sum / total, correct / total
