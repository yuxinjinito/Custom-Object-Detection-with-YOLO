# Stock Price Prediction with RNN / LSTM

Task: Create recurrent models (SimpleRNN and stacked LSTM) to predict future stock prices for Nasdaq-listed companies using a historical prices dataset.

## Overview

- **Framework**: TensorFlow / Keras (tf 2.19.0)  
- **Models**:
  - Basic RNN: single `SimpleRNN` layer  
  - RNN+LSTM: two stacked `LSTM` layers  
- **Task**: Multi-output regression – predict the next-day open/high/low/close for a chosen stock

## Data

- Source CSV: `prices-split-adjusted.csv`  
- Size: 851,264 rows, **501** tickers  
- Columns: `symbol`, `open`, `close`, `low`, `high`, `volume`  
- Preprocessing steps:
  - Select one stock (e.g. `EQIX`, `AAPL`) via `df[df.symbol == 'EQIX']`  
  - Drop `symbol` and `volume` → keep `open`, `close`, `low`, `high`  
  - Normalize each price column with `MinMaxScaler` to [0, 1]  
  - Build sliding windows of length `seq_len = 20` and split into:
    - Train / validation / test ≈ 80% / 10% / 10% via a custom `load_data` function  
    - Example shapes: `x_train (1394, 19, 4)`, `y_train (1394, 4)` etc.

## Models

### Basic RNN

- Hyperparameters:
  - `n_steps = seq_len - 1 = 19`  
  - `n_inputs = 4` (open, close, low, high)  
  - `n_neurons = 200`, `n_outputs = 4`  
- Architecture:
  - `Input(n_steps, n_inputs)` → `SimpleRNN(200, return_sequences=False)` → `Dense(4)`  
  - Output: next‑day 4‑D price vector `[open, close, low, high]`.
### RNN+LSTM (stacked LSTM)

- Same input/output configuration as Basic RNN  
- Architecture:
  - `Input(n_steps, n_inputs)` → `LSTM(200, return_sequences=True)`  
    → `LSTM(200, return_sequences=False)` → `Dense(4)`  
  - The first LSTM produces a 19×200 sequence; the second LSTM models that sequence and outputs a single 200‑D vector for next‑day prediction.

## Training & Evaluation

- Loss: Mean Squared Error (`'mse'`)  
- Metric: Mean Absolute Error (`'mae'`)  
- Optimizer: `Adam(learning_rate=0.001)`  
- Training:
  - `model.fit(x_train, y_train, batch_size=50, epochs=100, validation_data=(x_valid, y_valid))`  
  - Validation loss/MAE logged every epoch for both Basic RNN and LSTM models  
- Evaluation:
  - `model.evaluate(x_test, y_test)` and `model1.evaluate(x_test, y_test)`  
  - Example results:
    - Basic RNN: Test loss ≈ 2e‑4, Test MAE ≈ 0.0095  
    - LSTM: Test loss ≈ 2e‑4, Test MAE ≈ 0.0113  
  - Use Matplotlib to draw true vs. predicted price curves (train / valid / test).

## Files

- `README.md`:  (this file)  
- `question4.pdf`: notebook export with full code, plots, and detailed explanations
