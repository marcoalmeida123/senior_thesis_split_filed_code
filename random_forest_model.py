# === File: random_forest_model.py ===
# Contains functions for training, predicting, and evaluating the Random Forest Classifier.
# Updated to use multiple features including lagged OHLCV, Volume, and price differences.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_and_evaluate_random_forest(train_df, test_df, sma_period=50, n_estimators=200, random_state=42):
    """
    Trains a Random Forest Classifier on training data to predict price direction,
    predicts on test data, and evaluates the classification performance.

    Uses multiple features including lagged OHLCV, Volume, Daily Return, and SMA.

    Args:
        train_df (pd.DataFrame): Training data DataFrame (e.g., 2000-2023).
                                 Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        test_df (pd.DataFrame): Testing data DataFrame (e.g., 2024).
                                Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.
        sma_period (int): Period for Simple Moving Average feature.
        n_estimators (int): Number of trees in the Random Forest.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing classification evaluation metrics
              (Accuracy, Precision, Recall, F1-Score).
    """
    print("\n--- Running Random Forest Model (Multi-Feature) ---")

    # --- Data Preparation for Random Forest ---
    # Create features for RF from OHLCV and Volume
    train_df_rf = train_df.copy()
    test_df_rf = test_df.copy()

    # Combine train and test temporarily for consistent feature creation (especially lags)
    # This avoids issues at the train/test boundary for lagged features.
    full_df_rf = pd.concat([train_df_rf, test_df_rf])

    # Calculate features on the combined data
    full_df_rf['Daily_Return'] = full_df_rf['Close'].pct_change()
    full_df_rf['SMA_' + str(sma_period)] = full_df_rf['Close'].rolling(window=sma_period).mean()
    full_df_rf['High_Low_Diff'] = full_df_rf['High'] - full_df_rf['Low']
    full_df_rf['Open_Close_Diff'] = full_df_rf['Open'] - full_df_rf['Close']

    # Add lagged features (e.g., lag 1 day)
    full_df_rf['Close_Lag1'] = full_df_rf['Close'].shift(1)
    full_df_rf['Open_Lag1'] = full_df_rf['Open'].shift(1)
    full_df_rf['High_Lag1'] = full_df_rf['High'].shift(1)
    full_df_rf['Low_Lag1'] = full_df_rf['Low'].shift(1)
    full_df_rf['Volume_Lag1'] = full_df_rf['Volume'].shift(1)
    full_df_rf['Daily_Return_Lag1'] = full_df_rf['Daily_Return'].shift(1)


    # Define the target variable: 1 if the next day's close is higher than today's close, 0 otherwise.
    # This is calculated on the combined data before splitting back.
    full_df_rf['Target'] = (full_df_rf['Close'].shift(-1) > full_df_rf['Close']).astype(int)

    # Drop rows with NaN values created by feature engineering (SMAs, lags, first return, last target)
    full_df_rf.dropna(inplace=True)

    # Split the data back into training and testing sets based on the original date boundary
    # Need to find the index in the full_df_rf where the test data starts
    test_start_date = test_df.index[0] # Get the first date of the original test_df
    train_df_rf = full_df_rf.loc[full_df_rf.index < test_start_date].copy()
    test_df_rf = full_df_rf.loc[full_df_rf.index >= test_start_date].copy()


    if train_df_rf.empty or test_df_rf.empty:
        print("Not enough data after feature engineering and dropping NaNs for Random Forest.")
        return {"RF_Accuracy": np.nan, "RF_Precision": np.nan, "RF_Recall": np.nan, "RF_F1-Score": np.nan}

    # Define features (X) and target (y)
    # Include all the newly created features
    features_rf = [
        'Daily_Return', 'SMA_' + str(sma_period), 'High_Low_Diff', 'Open_Close_Diff',
        'Close_Lag1', 'Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Volume_Lag1', 'Daily_Return_Lag1'
        # Add more lags or other indicators here if desired
    ]
    target_rf = 'Target'

    # Ensure all defined features exist in the dataframes after dropping NaNs
    if not all(f in train_df_rf.columns and f in test_df_rf.columns for f in features_rf):
         missing_train = [f for f in features_rf if f not in train_df_rf.columns]
         missing_test = [f for f in features_rf if f not in test_df_rf.columns]
         print(f"Error: Missing required features for Random Forest. Train missing: {missing_train}, Test missing: {missing_test}")
         # Adjust features_rf to only include existing columns or exit
         features_rf = [f for f in features_rf if f in train_df_rf.columns and f in test_df_rf.columns]
         print(f"Using available features: {features_rf}")


    X_train_rf = train_df_rf[features_rf]
    y_train_rf = train_df_rf[target_rf]
    X_test_rf = test_df_rf[features_rf]
    y_test_rf = test_df_rf[target_rf]

    print(f"RF features used: {features_rf}")
    print(f"RF Training data shape: {X_train_rf.shape}")
    print(f"RF Testing data shape: {X_test_rf.shape}")


    # Optional: Scale RF features (less critical for tree models but can sometimes help)
    # Scaling is applied *after* splitting back into train/test
    scaler_rf = StandardScaler()
    cols_to_scale_rf = features_rf # Scale all features for simplicity here

    # Use .loc for explicit assignment to avoid SettingWithCopyWarning
    X_train_rf.loc[:, cols_to_scale_rf] = scaler_rf.fit_transform(X_train_rf[cols_to_scale_rf])
    X_test_rf.loc[:, cols_to_scale_rf] = scaler_rf.transform(X_test_rf[cols_to_scale_rf])


    print("Training Random Forest Classifier...")
    model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model_rf.fit(X_train_rf, y_train_rf)
    print("Random Forest training complete.")

    # --- Random Forest Prediction ---
    print("Making Random Forest Predictions...")
    predictions_rf = model_rf.predict(X_test_rf)

    # --- Random Forest Evaluation (Classification Metrics) ---
    print("Evaluating Random Forest Predictions...")
    # Ensure y_test_rf and predictions_rf have the same index for alignment
    # This should be the case if the splitting and dropping NaNs were handled correctly
    # However, let's confirm lengths match before evaluation
    min_len_rf_eval = min(len(y_test_rf), len(predictions_rf))
    y_test_rf_eval = y_test_rf.iloc[:min_len_rf_eval]
    predictions_rf_eval = predictions_rf[:min_len_rf_eval]


    accuracy_rf = accuracy_score(y_test_rf_eval, predictions_rf_eval)
    precision_rf = precision_score(y_test_rf_eval, predictions_rf_eval)
    recall_rf = recall_score(y_test_rf_eval, predictions_rf_eval)
    f1_rf = f1_score(y_test_rf_eval, predictions_rf_eval)
    conf_matrix_rf = confusion_matrix(y_test_rf_eval, predictions_rf_eval)
    class_report_rf = classification_report(y_test_rf_eval, predictions_rf_eval)


    print(f"Random Forest Accuracy for test period: {accuracy_rf:.4f}")
    print(f"Random Forest Precision for test period: {precision_rf:.4f}")
    print(f"Random Forest Recall for test period: {recall_rf:.4f}")
    print(f"Random Forest F1-Score for test period: {f1_rf:.4f}")
    print("\nConfusion Matrix (RF):")
    print(conf_matrix_rf)
    print("\nClassification Report (RF):")
    print(class_report_rf)

    # --- Random Forest Visualization (Direction) ---
    # Ensure dates for plotting match the evaluated data length
    test_dates_for_plot_rf = y_test_rf_eval.index

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates_for_plot_rf, y_test_rf_eval.values, label='Actual Direction (1=Up, 0=Down/Flat)', color='blue', marker='.', linestyle='None', alpha=0.6)
    plt.plot(test_dates_for_plot_rf, predictions_rf_eval, label='Predicted Direction (1=Up, 0=Down/Flat)', color='red', marker='x', linestyle='None', alpha=0.6)
    plt.title('SPY Direction Prediction using Random Forest (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Direction (1=Up, 0=Down/Flat)')
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Feature Importance
    print("\n--- Random Forest Feature Importance ---")
    if features_rf:
        # Ensure feature_importances are calculated based on the features actually used
        feature_importances_rf = pd.Series(model_rf.feature_importances_, index=features_rf).sort_values(ascending=False)
        print(feature_importances_rf)
        plt.figure(figsize=(10, 6))
        feature_importances_rf.plot(kind='bar')
        plt.title('Random Forest Feature Importance')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()


    return {"RF_Accuracy": accuracy_rf, "RF_Precision": precision_rf,
            "RF_Recall": recall_rf, "RF_F1-Score": f1_rf}

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
#     rf_results = train_and_evaluate_random_forest(train_df_test, test_df_test)
#     print("\nStandalone RF Results:", rf_results)
