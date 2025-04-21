# === File: lstm_model.py ===
# Contains functions for training, predicting, and evaluating the LSTM Model.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Ensure reproducible results (optional but good practice)
tf.random.set_seed(42)
np.random.seed(42)

def create_lstm_sequences(data, seq_length, num_features, target_feature_index):
    """
    Creates sequences of data for LSTM input and corresponding target values.

    Args:
        data (np.ndarray): Scaled data array [n_samples, num_features].
        seq_length (int): Number of past timesteps in each sequence.
        num_features (int): Number of features per timestep.
        target_feature_index (int): Index of the target feature in the data array (e.g., Close price index).

    Returns:
        tuple: (X, y) where X is the sequences array [n_sequences, seq_length, num_features]
               and y is the target array [n_sequences,].
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Get the sequence of features for 'seq_length' days
        seq_X = data[i:(i + seq_length), :]
        # Get the target value (scaled target feature) for the day after the sequence ends
        seq_y = data[i + seq_length, target_feature_index]

        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(train_df, test_df, sequence_length=60, batch_size=32, epochs=100, lstm_units=50, random_state=42):
    """
    Trains an LSTM model on training data using OHLCV features, makes price predictions
    on test data, and evaluates the performance.

    Args:
        train_df (pd.DataFrame): Training data DataFrame (2000-2023).
                                 Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        test_df (pd.DataFrame): Testing data DataFrame (2024).
                                Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        sequence_length (int): Number of past timesteps for LSTM sequences.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of training epochs.
        lstm_units (int): Number of units in LSTM layers.
        random_state (int): Seed for reproducibility (set globally).

    Returns:
        dict: A dictionary containing evaluation metrics (RMSE, MAE).
    """
    print("\n--- Running LSTM Model ---")

    # --- Data Preparation for LSTM ---
    # Select the features to use (Open, High, Low, Close, Volume)
    features_lstm = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_feature = 'Close'

    # Ensure all required features exist
    if not all(f in train_df.columns and f in test_df.columns for f in features_lstm):
        missing_train = [f for f in features_lstm if f not in train_df.columns]
        missing_test = [f for f in features_lstm if f not in test_df.columns]
        print(f"Error: Missing required columns for LSTM. Train missing: {missing_train}, Test missing: {missing_test}")
        return {"LSTM_RMSE": np.nan, "LSTM_MAE": np.nan}

    # Combine train and test data temporarily for consistent scaling across the full range
    # But fit the scaler ONLY on the training data
    full_data_lstm = pd.concat([train_df[features_lstm], test_df[features_lstm]])

    # Scale the data - fit ONLY on training portion
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    scaler_lstm.fit(train_df[features_lstm].values) # Fit only on training data values

    # Transform the full data using the fitted scaler
    full_data_scaled_lstm = scaler_lstm.transform(full_data_lstm.values)

    # Find the index where the test data starts in the full scaled data array
    test_start_iloc_scaled = len(train_df)

    # Split scaled data for sequence creation
    train_scaled_lstm = full_data_scaled_lstm[:test_start_iloc_scaled]
    # Test sequences need preceding history from the end of training
    test_scaled_lstm = full_data_scaled_lstm[test_start_iloc_scaled - sequence_length:]

    # Find the index of the target feature ('Close') in the features list
    target_feature_index_lstm = features_lstm.index(target_feature)
    num_features_lstm = len(features_lstm)

    # Create sequences for training and testing
    X_train_lstm, y_train_lstm = create_lstm_sequences(train_scaled_lstm, sequence_length, num_features_lstm, target_feature_index_lstm)
    X_test_lstm, y_test_lstm = create_lstm_sequences(test_scaled_lstm, sequence_length, num_features_lstm, target_feature_index_lstm)

    # Reshape X data for LSTM input [samples, timesteps, features] (confirming shape)
    X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], X_train_lstm.shape[2]))
    X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], X_test_lstm.shape[2]))

    print(f"LSTM Training sequences shape: {X_train_lstm.shape}")
    print(f"LSTM Training target shape: {y_train_lstm.shape}")
    print(f"LSTM Testing sequences shape: {X_test_lstm.shape}")
    print(f"LSTM Testing target shape: {y_test_lstm.shape}")

    # --- LSTM Model Definition and Training ---
    print("Training LSTM...")
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(sequence_length, num_features_lstm)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=lstm_units, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history_lstm = model_lstm.fit(X_train_lstm, y_train_lstm,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.1,
                                callbacks=[early_stopping_lstm],
                                verbose=0) # Set verbose to 0 to reduce training output in main script
    print("LSTM training complete.")

    # --- LSTM Prediction ---
    print("Making LSTM Predictions...")
    predictions_scaled_lstm = model_lstm.predict(X_test_lstm)

    # Inverse transform predictions
    # Create a dummy array with the same number of columns as original features
    dummy_array_lstm = np.zeros((predictions_scaled_lstm.shape[0], num_features_lstm))
    # Place the scaled predictions into the target feature column (e.g., Close)
    dummy_array_lstm[:, target_feature_index_lstm] = predictions_scaled_lstm[:, 0]
    # Inverse transform the dummy array and extract the target column
    predictions_lstm = scaler_lstm.inverse_transform(dummy_array_lstm)[:, target_feature_index_lstm]

    # Inverse transform actual test prices (y_test_lstm is scaled target)
    dummy_array_actual_lstm = np.zeros((y_test_lstm.shape[0], num_features_lstm))
    dummy_array_actual_lstm[:, target_feature_index_lstm] = y_test_lstm
    actual_prices_lstm = scaler_lstm.inverse_transform(dummy_array_actual_lstm)[:, target_feature_index_lstm]


    # --- LSTM Evaluation ---
    print("Evaluating LSTM Predictions...")
    # Ensure lengths match for evaluation (can be off by a few days due to sequence creation)
    min_len_lstm_eval = min(len(actual_prices_lstm), len(predictions_lstm))
    actual_prices_lstm_eval = actual_prices_lstm[:min_len_lstm_eval]
    predictions_lstm_eval = predictions_lstm[:min_len_lstm_eval]

    rmse_lstm = np.sqrt(mean_squared_error(actual_prices_lstm_eval, predictions_lstm_eval))
    mae_lstm = mean_absolute_error(actual_prices_lstm_eval, predictions_lstm_eval)

    print(f"LSTM RMSE for test period: {rmse_lstm:.4f}")
    print(f"LSTM MAE for test period: {mae_lstm:.4f}")

    # --- LSTM Visualization ---
    # The dates for the test predictions correspond to the dates *after* the sequence length
    # in the original test data portion of the DataFrame.
    # Need to get the original test data dates and align with prediction length
    test_dates_for_plot_lstm = test_df.index[sequence_length : sequence_length + min_len_lstm_eval]


    plt.figure(figsize=(14, 7))
    plt.plot(test_dates_for_plot_lstm, actual_prices_lstm_eval, label='Actual SPY Close Price', color='blue')
    plt.plot(test_dates_for_plot_lstm, predictions_lstm_eval, label='Predicted SPY Close Price (LSTM)', color='red', linestyle='--')
    plt.title('SPY Price Prediction using LSTM (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history_lstm.history['loss'], label='Train Loss')
    plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


    return {"LSTM_RMSE": rmse_lstm, "LSTM_MAE": mae_lstm}

# Example of how to run this function if needed standalone for testing:
# if __name__ == '__main__':
#     # Load your data here for standalone testing
#     df_full = pd.read_csv('spy_2000_2023.csv', index_col='Date', parse_dates=True)
#     df_full.sort_index(inplace=True)
#     train_df_test = df_full.loc[df_full.index < '2024-01-01'].copy()
#     test_df_test = df_full.loc[df_full.index >= '2024-01-01'].copy()
#     lstm_results = train_and_evaluate_lstm(train_df_test, test_df_test)
#     print("\nStandalone LSTM Results:", lstm_results)

