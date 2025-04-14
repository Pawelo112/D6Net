# Data.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class DiceDataset(Dataset):
    """
    Custom dataset for dice_ext.csv.

    For classification, the target label is defined as:
      - If the die is correct (column 0 == 1), label = number of dots (1-6).
      - Otherwise, label = 0.

    Additionally, the image pixels (28x28) are normalized (between 0 and 1) and the channel dimension is added.
    """
    def __init__(self, csv_file, transform=None):
        # Data file reading
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        # Dataset size
        return len(self.data)

    def __getitem__(self, index):
        # Single image with a label
        row = self.data.iloc[index]

        # Dice type (correct\incorrect)
        flag = row[0]
        # Number of dots/type of irregularity
        info = row[1]

        # Labeling
        # correct = dots number, incorrect - 0
        if flag:
            label = info
        else:
            label = 0

        # Pixel normalization
        image = row[2:].values.reshape(28, 28)
        image = image.astype(np.float32) / 255.0

        # Applying given transformation
        if self.transform:
            image = self.transform(image)

        # Returning image tensor and label
        return image, int(label)


# Creating different dataset; checking whether dice is correct or incorrect
class DiceBinaryDataset(Dataset):
    """
    Dataset for binary classification.

    The label is taken directly from column 0 (0 or 1).
    """
    def __init__(self, csv_file, transform=None):
        # Data file reading
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        # Dataset size
        return len(self.data)

    def __getitem__(self, index):
        # Single image with a label
        row = self.data.iloc[index]

        # Directly 0 or 1 value from column 0
        label = row[0]

        # Pixel normalization
        image = row[2:].values.reshape(28, 28)
        image = image.astype(np.float32) / 255.0

        # Applying given transformation
        if self.transform:
            image = self.transform(image)

        # Returning image tensor and label
        return image, int(label)


# Dataset for dots number
class DiceDotsDataset(Dataset):
    """
    Dataset for dots counting.

    Uses only correct dice (column 0 == 1) and converts dots counts from 1-6 to 0-5 (for cross entropy loss)
    """
    def __init__(self, csv_file, transform=None):
        # Data file reading
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        # Using only correct dice
        self.data = self.data[self.data[0] == 1].reset_index(drop=True)

    def __len__(self):
        # Dataset size
        return len(self.data)

    def __getitem__(self, index):
        # Single image with a label
        row = self.data.iloc[index]

        # Converting dots number from 1-6 to 0-5
        label = int(row[1]) - 1

        # Pixel normalization
        image = row[2:].values.reshape(28, 28)
        image = image.astype(np.float32) / 255.0

        # Applying given transformation
        if self.transform:
            image = self.transform(image)

        # Returning image tensor and label
        return image, int(label)


def get_transforms():
    """
    Returns a torchvision transformation for converting numpy arrays to tensors.
    """
    return transforms.Compose([
        transforms.ToTensor()
    ])
