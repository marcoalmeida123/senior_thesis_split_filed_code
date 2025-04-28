# === File: hmm_model.py ===
# Contains functions for training, predicting, and evaluating the Hidden Markov Model.
# Updated to use multiple features (Daily Return, Price Differences, Volume).

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_and_evaluate_hmm(train_df, test_df, n_states=3, random_state=42):
    """
    Trains a Gaussian HMM on training data using multiple features,
    makes indirect price predictions on test data based on learned state returns,
    and evaluates the performance.

    Args:
        train_df (pd.DataFrame): Training data DataFrame (e.g., 2000-2023).
                                 Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        test_df (pd.DataFrame): Testing data DataFrame (e.g., 2024).
                                Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        n_states (int): Number of hidden states for the HMM.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing evaluation metrics (RMSE, MAE).
    """
    print("\n--- Running Hidden Markov Model (Multi-Feature) ---")

    # --- Data Preparation for HMM ---
    # Create features for HMM from OHLCV and Volume
    train_df_hmm = train_df.copy()
    test_df_hmm = test_df.copy()

    # Calculate features for both train and test sets
    for df_hmm in [train_df_hmm, test_df_hmm]:
        df_hmm['Daily_Return'] = df_hmm['Close'].pct_change()
        df_hmm['High_Low_Diff'] = df_hmm['High'] - df_hmm['Low']
        df_hmm['Open_Close_Diff'] = df_hmm['Open'] - df_hmm['Close']
        # Using raw Volume might have large scale differences, consider scaling or using log volume
        # For simplicity, let's use raw Volume for now, scaling will handle the scale difference.
        # df_hmm['Volume'] is already present

    # Define the features to be used by the HMM
    hmm_features = ['Daily_Return', 'High_Low_Diff', 'Open_Close_Diff', 'Volume']

    # Drop rows with NaN values created by feature engineering (e.g., first row for Daily_Return)
    train_df_hmm.dropna(subset=hmm_features, inplace=True)
    test_df_hmm.dropna(subset=hmm_features, inplace=True)

    if train_df_hmm.empty or test_df_hmm.empty:
        print("Not enough data after creating features and dropping NaNs for HMM.")
        return {"HMM_RMSE": np.nan, "HMM_MAE": np.nan}

    # Select the feature values as NumPy arrays
    X_train_hmm = train_df_hmm[hmm_features].values
    X_test_hmm = test_df_hmm[hmm_features].values

    # Scale HMM data - Fit scaler ONLY on training data
    scaler_hmm = StandardScaler()
    X_train_scaled_hmm = scaler_hmm.fit_transform(X_train_hmm)
    # Transform test data using the scaler fitted on training data
    X_test_scaled_hmm = scaler_hmm.transform(X_test_hmm)

    print(f"HMM features used: {hmm_features}")
    print(f"HMM Training data shape (scaled): {X_train_scaled_hmm.shape}")
    print(f"HMM Testing data shape (scaled): {X_test_scaled_hmm.shape}")


    # --- Hidden Markov Model (HMM) - Training ---
    print("Training HMM...")
    # Define the HMM model
    # GaussianHMM assumes the observations within each state follow a Gaussian distribution
    # Using covariance_type="diag" assumes features are independent within states.
    # Using covariance_type="full" allows for correlations between features within states,
    # which might be more appropriate for OHLCV/Volume relationships, but requires more data.
    # Let's start with "diag" for robustness, but "full" is an option to explore.
    model_hmm = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=1000, random_state=random_state, tol=0.01)

    # Train the model on the scaled multi-feature training data
    try:
        model_hmm.fit(X_train_scaled_hmm)
        print("HMM training complete.")
    except Exception as e:
        print(f"Error during HMM training: {e}")
        return {"HMM_RMSE": np.nan, "HMM_MAE": np.nan}


    # --- HMM Prediction (Indirect Price Prediction) ---
    # Predict the most likely sequence of hidden states for the test data
    print("Making HMM Predictions...")
    hidden_states_test_hmm = model_hmm.predict(X_test_scaled_hmm)

    # To make an indirect price prediction, we need the mean Daily Return for each state.
    # The model.means_ contains the mean of the *scaled* features for each state.
    # We need to inverse transform these means and extract the mean of the Daily_Return feature.

    # Get the index of 'Daily_Return' in the hmm_features list
    daily_return_feature_index = hmm_features.index('Daily_Return')

    # Inverse transform the learned means for all features
    mean_features_per_state = scaler_hmm.inverse_transform(model_hmm.means_)

    # Extract the mean Daily Return for each state
    mean_returns_per_state_hmm = mean_features_per_state[:, daily_return_feature_index]

    # Generate predicted returns based on the predicted states for the test period
    predicted_returns_hmm = [mean_returns_per_state_hmm[state] for state in hidden_states_test_hmm]

    # Convert predicted returns to a price series
    # Start with the last closing price from the training data *after* dropping NaNs for HMM features
    last_train_close_hmm = train_df_hmm['Close'].iloc[-1]

    predicted_prices_hmm = [last_train_close_hmm]
    for i in range(len(predicted_returns_hmm)):
        next_predicted_price = predicted_prices_hmm[-1] * (1 + predicted_returns_hmm[i])
        predicted_prices_hmm.append(next_predicted_price)
    predicted_prices_hmm = predicted_prices_hmm[1:] # Remove the initial last_train_close

    # --- HMM Evaluation ---
    print("Evaluating HMM Predictions...")
    # We need the actual closing prices for the test period (2024) *after* dropping NaNs for HMM features
    actual_prices_hmm = test_df_hmm['Close'].values

    # Ensure the lengths match for evaluation (can be off by a few days due to NaN drops)
    min_len_hmm = min(len(actual_prices_hmm), len(predicted_prices_hmm))
    actual_prices_hmm = actual_prices_hmm[:min_len_hmm]
    predicted_prices_hmm = predicted_prices_hmm[:min_len_hmm]

    rmse_hmm = np.sqrt(mean_squared_error(actual_prices_hmm, predicted_prices_hmm))
    mae_hmm = mean_absolute_error(actual_prices_hmm, predicted_prices_hmm)

    print(f"HMM RMSE for test period: {rmse_hmm:.4f}")
    print(f"HMM MAE for test period: {mae_hmm:.4f}")

    # --- HMM Visualization ---
    # Ensure the dates for plotting match the length of the evaluated data
    test_dates_for_plot_hmm = test_df_hmm.index[:min_len_hmm]
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates_for_plot_hmm, actual_prices_hmm, label='Actual SPY Close Price', color='blue')
    plt.plot(test_dates_for_plot_hmm, predicted_prices_hmm, label='Predicted SPY Close Price (HMM)', color='red', linestyle='--')
    plt.title('SPY Price Prediction using Hidden Markov Model (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Visualize the predicted hidden states
    plt.figure(figsize=(14, 4))
    plt.plot(test_dates_for_plot_hmm, hidden_states_test_hmm[:min_len_hmm], label='Predicted HMM State', color='purple', marker='.', linestyle='None')
    plt.title('Predicted Hidden States (HMM) for Test Period')
    plt.xlabel('Date')
    plt.ylabel('State')
    plt.yticks(range(n_states))
    plt.legend()
    plt.grid(True)
    plt.show()


    return {"HMM_RMSE": rmse_hmm, "HMM_MAE": mae_hmm}

# Example of how to run this function if needed standalone for testing:
# if __name__ == '__main__':
#     # Load your data here for standalone testing
#     try:
#         train_df_test = pd.read_csv('spy_2000_2023.csv', skiprows=3, header=None, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], parse_dates=['Date'], index_col='Date')
#         test_df_test = pd.read_csv('spy_2024.csv', skiprows=3, header=None, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], parse_dates=['Date'], index_col='Date')
#         train_df_test.sort_index(inplace=True)
#         test_df_test.sort_index(inplace=True)
#         print("Standalone test data loaded.")
#     except Exception as e:
#         print(f"Error loading standalone test data: {e}")
#         exit()
#
#     hmm_results = train_and_evaluate_hmm(train_df_test, test_df_test)
#     print("\nStandalone HMM Results:", hmm_results)
