# === File: hmm_model.py ===
# Contains functions for training, predicting, and evaluating the Hidden Markov Model.

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_and_evaluate_hmm(train_df, test_df, n_states=3, random_state=42):
    """
    Trains a Gaussian HMM on training data, makes indirect price predictions
    on test data, and evaluates the performance.

    Args:
        train_df (pd.DataFrame): Training data DataFrame (2000-2023).
                                 Must contain a 'Close' column.
        test_df (pd.DataFrame): Testing data DataFrame (2024).
                                Must contain a 'Close' column.
        n_states (int): Number of hidden states for the HMM.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing evaluation metrics (RMSE, MAE).
    """
    print("\n--- Running Hidden Markov Model ---")

    # --- Data Preparation for HMM ---
    # HMM requires Daily_Return feature
    train_df_hmm = train_df.copy()
    test_df_hmm = test_df.copy()

    train_df_hmm['Daily_Return'] = train_df_hmm['Close'].pct_change()
    test_df_hmm['Daily_Return'] = test_df_hmm['Close'].pct_change()

    # Drop the first row which will have NaN for Daily_Return
    train_df_hmm.dropna(inplace=True)
    test_df_hmm.dropna(inplace=True)

    if train_df_hmm.empty or test_df_hmm.empty:
        print("Not enough data after calculating returns and dropping NaNs for HMM.")
        return {"HMM_RMSE": np.nan, "HMM_MAE": np.nan}


    X_train_hmm = train_df_hmm[['Daily_Return']].values
    X_test_hmm = test_df_hmm[['Daily_Return']].values

    # Scale HMM data
    scaler_hmm = StandardScaler()
    X_train_scaled_hmm = scaler_hmm.fit_transform(X_train_hmm)
    X_test_scaled_hmm = scaler_hmm.transform(X_test_hmm)

    print("Training HMM...")
    model_hmm = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=1000, random_state=random_state, tol=0.01)
    try:
        model_hmm.fit(X_train_scaled_hmm)
        print("HMM training complete.")
    except Exception as e:
        print(f"Error during HMM training: {e}")
        return {"HMM_RMSE": np.nan, "HMM_MAE": np.nan}


    # --- HMM Prediction (Indirect Price Prediction) ---
    print("Making HMM Predictions...")
    hidden_states_test_hmm = model_hmm.predict(X_test_scaled_hmm)
    mean_returns_per_state_hmm = scaler_hmm.inverse_transform(model_hmm.means_)
    predicted_returns_hmm = [mean_returns_per_state_hmm[state][0] for state in hidden_states_test_hmm]

    # Convert predicted returns to a price series
    last_train_close_hmm = train_df_hmm['Close'].iloc[-1]
    predicted_prices_hmm = [last_train_close_hmm]
    for i in range(len(predicted_returns_hmm)):
        next_predicted_price = predicted_prices_hmm[-1] * (1 + predicted_returns_hmm[i])
        predicted_prices_hmm.append(next_predicted_price)
    predicted_prices_hmm = predicted_prices_hmm[1:] # Remove the initial last_train_close

    # --- HMM Evaluation ---
    print("Evaluating HMM Predictions...")
    actual_prices_hmm = test_df_hmm['Close'].values
    min_len_hmm = min(len(actual_prices_hmm), len(predicted_prices_hmm))
    actual_prices_hmm = actual_prices_hmm[:min_len_hmm]
    predicted_prices_hmm = predicted_prices_hmm[:min_len_hmm]

    rmse_hmm = np.sqrt(mean_squared_error(actual_prices_hmm, predicted_prices_hmm))
    mae_hmm = mean_absolute_error(actual_prices_hmm, predicted_prices_hmm)

    print(f"HMM RMSE for test period: {rmse_hmm:.4f}")
    print(f"HMM MAE for test period: {mae_hmm:.4f}")

    # --- HMM Visualization ---
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
#     df_full = pd.read_csv('spy_2000_2023.csv', index_col='Date', parse_dates=True)
#     df_full.sort_index(inplace=True)
#     train_df_test = df_full.loc[df_full.index < '2024-01-01'].copy()
#     test_df_test = df_full.loc[df_full.index >= '2024-01-01'].copy()
#     hmm_results = train_and_evaluate_hmm(train_df_test, test_df_test)
#     print("\nStandalone HMM Results:", hmm_results)

