Logistic Regression for Flower Classification (Iris)

Task: Build and implement a Logistic Regression model to classify three types of iris flowers using the Iris dataset in `sklearn`.

## Overview

- **Framework**: scikit-learn (`sklearn`)  
- **Model**: Multinomial Logistic Regression  
- **Dataset**: Iris (150 samples, 4 numeric features, 3 classes)

## Data

- Loaded via `sklearn.datasets.load_iris()`  
- Features:
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`
- Labels:
  - `0`: setosa  
  - `1`: versicolor  
  - `2`: virginica  
- Exploratory steps:
  - Wrap features in a `pandas.DataFrame`  
  - Use `.describe()`, `.head()` to inspect statistics  
  - Optionally pick 2 random features and visualize class separation with scatter plots.

## Model

- `sklearn.linear_model.LogisticRegression(max_iter=1000)`  
- Trained on all four features (and optionally on 2-feature subsets for visualization).

## Training & Evaluation

- Fit: `model.fit(X, y)`  
- Inspect parameters:
  - `model.coef_` (shape: 3×4)  
  - `model.intercept_` (shape: 3,)  
- Predictions:
  - `y_pred = model.predict(X)`  
- Metric:
  - `accuracy_score(y, y_pred)`  
  - Reported **accuracy ≈ 0.973** on the full dataset  
- Additional visualization: 2D decision boundaries when training on two selected features.

## Files

- `README.md`: (this file)  
- `question2.pdf`: notebook export with questions, full code, visualizations, and test cases
