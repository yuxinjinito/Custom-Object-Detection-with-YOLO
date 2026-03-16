## Deep Learning Course Projects 

This repository collects all coursework for a deep learning class, including five foundational assignments and a final custom object detection project.

- **Lab 1 – Digit Recognition (PyTorch MLP)**  
  - Implemented a multilayer perceptron in PyTorch on MNIST (CSV format).  
  - Architecture: 784→1000→500→10 with ReLU; manual SGD‑style parameter updates.  
  - Achieved ~89% test accuracy.

- **Lab 2 – Flower Classification (Logistic Regression)**  
  - Used `sklearn` Logistic Regression on the Iris dataset (150 samples, 3 classes).  
  - Explored data with `pandas`, trained multinomial logistic regression, and reached ~0.973 accuracy.

- **Lab 3 – House Price Regression (ANN/MLP)**  
  - Built an `MLPRegressor` on the California Housing dataset (20,640 samples, 8 features).  
  - Two hidden layers (100, 50), ReLU, Adam; evaluated MAE/MSE/RMSE (target in $100k units).

- **Lab 4 – Stock Price Prediction (RNN & LSTM)**  
  - Worked with Nasdaq price series (`prices-split-adjusted.csv`, 501 stocks).  
  - Constructed sliding windows and trained both a SimpleRNN and a stacked LSTM in TensorFlow to forecast next‑day OHLC prices; compared train/val/test MSE and MAE.

- **Lab 5 – Image Classification (CNN)**  
  - Implemented a Keras CNN for Fashion‑MNIST (28×28 grayscale, 10 classes).  
  - Two Conv2D+MaxPool+Dropout blocks followed by Flatten+Dense; trained with categorical cross‑entropy and Adam, and visualized predictions.

- **Final Project – Custom Object Detection with YOLO**  
  - Collected and annotated a web‑crawled 10‑class image dataset; built train/val/test splits (≈1863/200/220 images).  
  - Trained a YOLOv11‑based detector (Ultralytics) under GPU memory constraints (batch size, learning rate, optimizer, early stopping).  
  - Evaluated on the validation set (200 images, 518 instances) with mAP@0.5 = 0.34 and mAP@[0.5:0.95] = 0.23, and analyzed errors using confusion matrices and prediction visualizations.

## Project Overview Mind Map

The overall course project structure is summarized in the following mind map:

- Editable source: `docs/overview.xmind`
- Exported pdf: `docs/overview.pdf`
- Exported image:
![Overview](docs/overview.png)