House Price Prediction with ANN/MLP

Task: Design and implement an ANN/MLP model to predict house prices in California using the `sklearn` California Housing dataset.

## Overview

- **Framework**: scikit-learn (`MLPRegressor`)  
- **Model**: Feedforward neural network (two hidden layers)  
- **Task**: Regression – predict median house value in tens of thousands of dollars

## Data

- Loaded via `sklearn.datasets.fetch_california_housing()`  
- **Samples**: 20,640  
- **Features (8)**:
  - `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`,  
  - `Population`, `AveOccup`, `Latitude`, `Longitude`
- Target: `housing.target` (median house value, unit: \$100,000)  
- Preprocessing:
  - Wrap features in `pandas.DataFrame`  
  - Train/test split via `train_test_split(..., test_size=0.2, random_state=...)`

## Model

- `MLPRegressor(`  
  `hidden_layer_sizes=(100, 50),`  
  `learning_rate_init=0.001,`  
  `activation='relu',`  
  `solver='adam',`  
  `max_iter=1000`  
  `)`

## Training & Evaluation

- Fit: `mlp.fit(x_train, y_train)`  
- Training diagnostics:
  - Loss curve from `mlp.loss_curve_` plotted over iterations  
- Predictions:
  - `y_pred = mlp.predict(x_test)`  
- Metrics (on the test set):
  - **MAE** (Mean Absolute Error)  
  - **MSE** (Mean Squared Error)  
  - **RMSE** (Root Mean Squared Error)  
  - Example results from the notebook: MAE ≈ 0.63, RMSE ≈ 0.86 (units: \$100,000)

## Files

- `README.md`: (this file)  
- `question3.pdf`: notebook export with questions, full code, loss curves, and evaluation metrics
