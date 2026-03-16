# Image Classification with CNN (Fashion-MNIST)

Task: Develop a CNN model to classify images into 10 classes using the Fashion-MNIST dataset.

## Overview

- **Framework**: TensorFlow / Keras  
- **Model type**: Convolutional Neural Network (CNN)  
- **Dataset**: Fashion-MNIST (10 clothing / accessory classes)

## Data

- Loaded via `tf.keras.datasets.fashion_mnist.load_data()` or local gzip files  
- Image shape: **28×28** grayscale  
- Dataset size:
  - 60,000 training images  
  - 10,000 test images  
- Preprocessing:
  - Normalize pixel values: `x = x / 255.0` → range [0, 1]  
  - Split training into **train / validation**: 55,000 / 5,000  
  - Reshape to `(N, 28, 28, 1)`  
  - One‑hot encode labels with `tf.keras.utils.to_categorical(y, 10)`

## Model Architecture

Sequential CNN:

1. `Conv2D(64, kernel_size=2, padding='same', activation='relu')`  
2. `MaxPooling2D(pool_size=2)`  
3. `Dropout(0.3)`  
4. `Conv2D(32, kernel_size=2, padding='same', activation='relu')`  
5. `MaxPooling2D(pool_size=2)`  
6. `Dropout(0.3)`  
7. `Flatten()`  
8. `Dense(256, activation='relu')`  
9. `Dropout(0.5)`  
10. `Dense(10, activation='softmax')`  (class probabilities)

Total parameters: **412,778** (as reported by `model.summary()`).

## Training & Evaluation

- Compile:
  - Loss: `categorical_crossentropy`  
  - Optimizer: `adam`  
  - Metrics: `accuracy`
- Train:
  - `model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_valid, y_valid), callbacks=[ModelCheckpoint(..., save_best_only=True)])`  
  - Checkpoints: best validation weights saved to `model.weights.best.h5`
- Evaluate:
  - Final accuracy on **validation** and **test** sets reported in the notebook  
  - Use `plt.imshow(x)` + predicted labels to visualize classification results on sample images.

## Files

- `README.md`: (this file)  
- `question5.pdf`: notebook export with questions, full code, model summary, training curves, and evaluation
