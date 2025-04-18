# 🎲Dice Classification 
## 📚 Table of Contents
<details>
  <summary><strong>Click to expand</strong></summary>

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Design Decisions & Implementation Details](#design-decisions--implementation-details)
  - [Data Preparation](#data-preparation)
  - [Network Architectures](#network-architectures)
  - [Training and Evaluation](#training-and-evaluation)
  - [Inference and Error Analysis](#inference-and-error-analysis)
  - [Tools](#tools)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)
- [License](#license)

</details>


## 🔍Overview

This is my university project that implements a classic D6 cube dice classification system using deep neural networks.  
The goal is to classify 28×28 grayscale images of dice based on the number of dots visible on the upper face.  
The dataset (`dice_ext`) contains images of both correct dice (with dot counts from 1 to 6) and incorrect dice (labeled as 0).  

The model is expected to:
- **Identify incorrect dice:** If the die is incorrect, the system should report it as such.
- **Count dots on correct dice:** If the die is correct, the system should accurately predict the number of dots.

Performance is measured in terms of overall accuracy - which includes both detecting incorrect dice and accurately counting dots on correct dice.

This project implements three different approaches:
1. **Approach 1 (Single Model):** Uses one convolutional neural network to classify images into 7 classes (0 representing an incorrect die, and 1–6 representing the number of dots on a correct die).
2. **Approach 2 (Ensemble):** Combines two models:
   - A binary classifier (to determine if the die is correct or incorrect).
   - A dot counting network (used only on images classified as correct).  
   In the ensemble, if the binary classifier predicts a die is correct, the dot counting model’s output (adjusted from a 0–5 range to 1–6) is used.
3. **Approach 3 (Uncertainty):** Uses the single-model architecture from Approach 1 but adds a confidence threshold. If the network’s maximum softmax probability is below the threshold (or if the predicted class is 0), the die is considered incorrect (indicating uncertainty).

Confusion matrices and error counts are generated for each approach to help in evaluating and comparing their performance.

## 📋Project Structure

The repository is organized into several modules for modularity and maintainability:

- **`data.py`**  
  Contains custom dataset classes:  
  - `DiceDataset` for the full multi-class classification problem.  
  - `DiceBinaryDataset` for binary classification.  
  - `DiceDotsDataset` for dot counting (remaps dot count from 1–6 to 0–5).  
  Also includes a helper function, `get_transforms()`, that returns the necessary image transformations.

- **`models.py`**  
  Defines the network architectures:  
  - `CNN`: A convolutional neural network used for both Approach 1 (7-class classification) and the dot counting model in Approach 2.  
  - `BinaryCNN`: A network for binary classification (used in Approach 2).

- **`train_eval.py`**  
  Contains the training and evaluation loops:  
  - `train_model()`: Runs the forward pass, computes the loss, backpropagates, and updates network weights.  
  - `evaluate_model()`: Evaluates the model on a validation/test set, computing average loss and accuracy.

- **`inference.py`**  
  Provides functions for inference and error analysis:  
  - `ensemble_inference()`: Implements the ensemble prediction (first using the binary model, then using the dot counting model if the die is predicted correct).  
  - `uncertain_inference()`: Applies uncertainty estimation by checking the maximum softmax probability against a threshold.  
  - Functions that compute confusion matrices and count errors for each approach, and a function (`plot_multiple_confusion_matrices()`) to display the confusion matrices side by side.
  
- **`visualization.py`**  
  Includes visualization utilities for displaying predictions and training progress:
  - Functions to show sample predictions for each individual model.
  - A combined visualizer that compares predictions across all three approaches for the same test samples.
  - Useful for debugging, understanding model behavior, and creating illustrative visuals for reports or presentations.

- **`main.py`**  
  The main script that integrates data loading, training, evaluation, and visualization. It trains the models (or loads pre-trained ones), plots training curves, displays sample predictions from the test set, and shows the confusion matrices for all three approaches.

- **`requirements.txt`**  
  Lists the Python packages needed for the project.

  
## 📝Design Decisions & Implementation Details

### 📊Data Preparation

- **Dataset Classes:**  
  - `DiceDataset` loads all images from `dice_ext.csv` and assigns labels based on a two-column scheme: column 0 is a flag for correct/incorrect; column 1 is either the dot count (if correct) or an error type (if incorrect).  
  - `DiceBinaryDataset` is used for binary classification with labels 0 or 1.  
  - `DiceDotsDataset` is used exclusively for dot counting. Since the dot counting model is trained only on correct dice, its labels are remapped from a 1–6 range to 0–5 (which makes indexing easier for training), and then later converted back by adding 1 when reporting predictions.

- **Transformations:**  
  A single transformation using `ToTensor()` converts normalized NumPy arrays (values between 0 and 1) into PyTorch tensors.

### 🌐Network Architectures

- **CNN (Approach 1 & Dot Counting):**  
  - **Feature Extraction:**  
    Consists of two convolutional layers. The first layer extracts 32 features using a 3×3 kernel with padding 1 (ensuring the output size remains the same before pooling). This is followed by a ReLU activation and a MaxPooling layer that downsamples the image from 28×28 to 14×14. The second convolutional layer increases the depth to 64 features, followed by ReLU and a second MaxPooling layer, reducing the spatial size to 7×7.
  - **Classifier:**  
    A `Flatten()` layer converts the 64 × 7 × 7 feature map into a 1D vector of size 64×7×7 (3136 features). This is passed through a fully connected (dense) layer with 128 neurons and a ReLU activation. A dropout layer (p=0.2) is applied for regularization, followed by a final linear layer mapping the 128 features to the desired number of classes (7 for Approach 1, or 6 for dot counting after converting back).

- **BinaryCNN (Ensemble’s Binary Classifier):**  
  Follows a similar architecture to the CNN used in Approach 1, but its final fully connected layer produces only 2 outputs (representing incorrect (0) and correct (1) dice).

### 🏁Training and Evaluation

- **Training Loop:**  
  The `train_model()` function sets the model to training mode. For each batch, it performs the forward pass, computes the cross-entropy loss, zeroes gradients, backpropagates errors, and updates model parameters. The function also keeps a cumulative loss and computes accuracy over the training data.
  
- **Evaluation Loop:**  
  The `evaluate_model()` function turns off gradient computation (using `torch.no_grad()`) and evaluates the model on a validation or test set. It similarly computes total loss and accuracy, but without updating any weights.

- **Early Stopping:**  
  Training is stopped early if the validation loss does not improve for a configurable number of epochs (patience).

### 🚨Inference and Error Analysis

- **Ensemble Inference:**  
  In Approach 2, the binary model first determines if a die is correct. For dice predicted as correct, the dots model (dot counter) predicts the dot count. The output from the dots model is adjusted by adding 1 (since it was trained on labels 0–5) so that the final prediction is in the range 1–6.
  
- **Uncertainty Inference:**  
  In Approach 3, the model checks the maximum softmax probability. If it is below a specified threshold or if the predicted class is 0, the output is interpreted as “Incorrect die (uncertainty)”—otherwise, it returns the dot count.

- **Confusion Matrices:**  
  Confusion matrices are computed for each approach to compare true and predicted labels. Additionally, each confusion function counts the number of errors (mismatched predictions) so that the overall performance can be quantitatively compared.

### 🧰Tools

- **PyTorch / Torchvision:**  
  Used for building and training models.
- **Pandas & NumPy:**  
  For data reading and preprocessing.
- **scikit-learn:**  
  For data splitting and calculating confusion matrices.
- **Matplotlib & Seaborn:**  
  Used for plotting training curves, sample predictions, and confusion matrices.

## 👨‍🏫How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/dice-classification.git
   cd dice-classification
    ```
2. **Install Dependencies:**  
It is recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```  

3. **Prepare the Data:**  
Place dice_ext.csv in the repository folder   

4. **Run the Main Script:**  

  ```bash
  python main.py
  ```
This script will:
+ Load the data and split it into training and test sets.
+ Initialize, train, and evaluate the models for all three approaches.
+ Generate and display training curves.
+ Display sample predictions on 10 random test images.
+ Plot confusion matrices (with error counts) for all three approaches.

## 📌Conclusion
This project demonstrates multiple approaches for dice classification using convolutional neural networks, including a single model approach, an ensemble method, and an uncertainty-based approach.    
The project is organized into separate Python modules for better maintainability and clarity, and results are thoroughly visualized with training curves, sample predictions, and confusion matrices.  
Feel free to explore and extend the project further!

## 📖License
Copyright © 2025 [Paweł Marcinkowski](https://github.com/Pawelo112).  
This project is [MIT](https://github.com/Pawelo112/D6Net/blob/main/LICENSE) licensed.
