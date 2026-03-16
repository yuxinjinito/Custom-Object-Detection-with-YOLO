# Digit Recognition with PyTorch MLP

Task: Implement a basic neural network using PyTorch to train a digit recognition model on the MNIST dataset.

## Overview

- **Framework**: PyTorch  
- **Model**: Multilayer Perceptron (MLP)  
- **Dataset**: MNIST (CSV format: `mnist_train.csv`, `mnist_test.csv`)  
- **Goal**: Classify handwritten digits 0–9 from 28×28 grayscale images.

## Data

- Each CSV row has **785 columns**:  
  - Column 0: digit label (0–9)  
  - Columns 1–784: flattened pixel values (28×28)  
- Training set: **60,000** samples  
- Test set: **10,000** samples  
- Data loading and preprocessing:
  - Read CSV with `pandas.read_csv`
  - Split labels/features via `.iloc`
  - Convert to `torch.tensor` (`float32` for `x`, `long` for `y`)

## Model

PyTorch `nn.Module` MLP:

- `fc1`: 784 → 1000, ReLU  
- `fc2`: 1000 → 500, ReLU  
- `fc3`: 500 → 10, output logits (one per class)

## Training

- Loss: `torch.nn.functional.cross_entropy`  
- Optimization: manual SGD-style update (no built-in optimizer):
  - `loss.backward()`  
  - For each parameter `p`: `p -= lr * p.grad`  
  - `mnist.zero_grad()` to avoid gradient accumulation  
- Hyperparameters:
  - Learning rate `lr = 0.001`  
  - Epochs: `100`

## Evaluation

- Inference under `torch.no_grad()` on the test set  
- Predicted class via `torch.max(logits, dim=1)`  
- Reported **test accuracy ≈ 89.05%**  
- Prints several test samples’ input vectors and predicted digits.

## Files

- `README.md`: (this file)  
- `question1.pdf`: original notebook export with full code, questions, and test cases
