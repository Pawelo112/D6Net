# models.py
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional network for classifying 28x28 dice images.

    Output classes:
        - Class 0: Incorrect die.
        - Classes 1-6: Correct die with the corresponding number of dots.
    """
    def __init__(self, num_classes):
        # We inherit everything from the base constructor of nn.Module
        super(CNN, self).__init__()

        # Features extraction section
        self.features = nn.Sequential(
            # Convolutional layer: 1 channel --> 32 filters
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Image size from 28x28 to 14x14
            nn.MaxPool2d(2),
            # Second convolutional layer from 32 to 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Image size from 14x14 to 7x7
            nn.MaxPool2d(2)
        )

        # Classification section
        # Flattening fo feed to FC layers, Dense layer, Hidden layer, Output layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Adding Dropout to minimize overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Features extraction
        x = self.features(x)
        # Classification
        x = self.classifier(x)
        return x


class BinaryCNN(nn.Module):
    """
    Convolutional network for binary classification

    Output classes:
        - 0: Incorrect die.
        - 1: Correct die.
    """
    def __init__(self):
        # We inherit everything from the base constructor of nn.Module
        super(BinaryCNN, self).__init__()

        # Features extraction section
        self.features = nn.Sequential(
            # Convolutional layer: 1 channel --> 32 filters
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Image size from 28x28 to 14x14
            nn.MaxPool2d(2),
            # Second convolutional layer from 32 to 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Image size from 14x14 to 7x7
            nn.MaxPool2d(2)
        )

        # Classification section
        # Flattening fo feed to FC layers, Dense layer, Hidden layer, Output layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Adding Dropout to minimize overfitting
            # 2 output classes
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Features extraction
        x = self.features(x)
        # Classification
        x = self.classifier(x)
        return x
