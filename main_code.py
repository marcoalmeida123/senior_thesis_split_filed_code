# === File: main_code.py ===
# This script loads SPY training data (2000-2023) and testing data (2024)
# from separate files, calls functions from separate model files
# to train on the training data, predict on the testing data,
# and prints a summary of the results.

import pandas as pd
import os
import numpy as np # Needed for np.nan in results dictionary

# Import the training and evaluation functions from your model files
# Make sure these files (hmm_model.py, random_forest_model.py, lstm_model.py)
# are in the same directory as main_code.py
try:
    from hmm_model import train_and_evaluate_hmm
    from random_forest_model import train_and_evaluate_random_forest
    from lstm_model import train_and_evaluate_lstm
    print("Successfully imported model functions.")
except ImportError as e:
    print(f"Error importing model functions: {e}")
    print("Please ensure hmm_model.py, random_forest_model.py, and lstm_model.py")
    print("are in the same directory as main_code.py and contain the required functions.")
    exit()


# --- Configuration ---
# Define the file paths for your separate training and testing data
TRAIN_DATA_FILE = 'spy_2000_2023.csv' # Data for training (e.g., 2000-2023)
TEST_DATA_FILE = 'spy_2024.csv'       # Data for testing (e.g., 2024)

# Model Hyperparameters (can be adjusted)
HMM_N_STATES = 3
RF_SMA_PERIOD = 50
RF_N_ESTIMATORS = 200
LSTM_SEQUENCE_LENGTH = 60
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 100 # Max epochs for LSTM, Early Stopping is used
LSTM_UNITS = 50
RANDOM_STATE = 42 # For reproducibility


# --- Data Loading ---
print(f"--- Loading Training Data from {TRAIN_DATA_FILE} ---")
if not os.path.exists(TRAIN_DATA_FILE):
    print(f"Error: Training data file not found at {TRAIN_DATA_FILE}")
    print("Please ensure '{TRAIN_DATA_FILE}' is in the correct directory.")
    exit()

try:
    # Load Training CSV with specific parameters to handle extra headers
    # skip the first 3 rows which contain extra header information
    # header=None because the row we start reading from (row 4) does not have column names
    # names: manually provide the column names in the correct order
    # parse_dates: tell pandas to parse the 'Date' column as datetime objects
    # index_col: set the 'Date' column as the DataFrame index
    train_df_master = pd.read_csv(
        TRAIN_DATA_FILE,
        skiprows=3,       # Skip the first 3 lines based on previous CSV format
        header=None,      # The 4th line (after skipping) is data, not a header
        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], # Provide correct column names
        parse_dates=['Date'], # Tell pandas to parse the 'Date' column as dates
        index_col='Date'  # Set the 'Date' column as the index
    )

    print(f"Successfully loaded training data from {TRAIN_DATA_FILE}")
    print(f"Training data columns: {train_df_master.columns.tolist()}")
    print(f"Training data shape: {train_df_master.shape}")
    print("First 5 rows of training data:")
    print(train_df_master.head())

except Exception as e:
    print(f"An error occurred during training data loading or processing: {e}")
    print("Please check the format of your training CSV file, especially the number of header rows.")
    exit()

# Ensure training data is sorted by date
train_df_master.sort_index(inplace=True)


print(f"\n--- Loading Testing Data from {TEST_DATA_FILE} ---")
if not os.path.exists(TEST_DATA_FILE):
    print(f"Error: Testing data file not found at {TEST_DATA_FILE}")
    print("Please ensure '{TEST_DATA_FILE}' is in the correct directory.")
    exit()

try:
    # Load Testing CSV with the SAME specific parameters to handle extra headers
    test_df_master = pd.read_csv(
        TEST_DATA_FILE,
        skiprows=3,       # Skip the first 3 lines based on previous CSV format
        header=None,      # The 4th line (after skipping) is data, not a header
        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], # Provide correct column names
        parse_dates=['Date'], # Tell pandas to parse the 'Date' column as dates
        index_col='Date'  # Set the 'Date' column as the index
    )

    print(f"Successfully loaded testing data from {TEST_DATA_FILE}")
    print(f"Testing data columns: {test_df_master.columns.tolist()}")
    print(f"Testing data shape: {test_df_master.shape}")
    print("First 5 rows of testing data:")
    print(test_df_master.head())


except Exception as e:
    print(f"An error occurred during testing data loading or processing: {e}")
    print("Please check the format of your testing CSV file, especially the number of header rows.")
    exit()

# Ensure testing data is sorted by date
test_df_master.sort_index(inplace=True)


# Check if test data is empty after loading
if test_df_master.empty:
    print(f"Error: Testing data loaded from {TEST_DATA_FILE} is empty.")
    print("Please ensure the file contains data.")
    exit()

print(f"\nMaster Training data shape: {train_df_master.shape}")
print(f"Master Testing data shape: {test_df_master.shape}")


# --- Run Models and Collect Results ---
all_results = {}

# Run HMM Model
print("\n" + "="*60)
print("--- Running HMM Model ---")
print("="*60)
# Pass the separate training and testing dataframes to the HMM function
hmm_results = train_and_evaluate_hmm(
    train_df=train_df_master.copy(), # Pass a copy to prevent modifications
    test_df=test_df_master.copy(),   # Pass a copy
    n_states=HMM_N_STATES,
    random_state=RANDOM_STATE
)
all_results.update(hmm_results)


# Run Random Forest Model
print("\n" + "="*60)
print("--- Running Random Forest Model ---")
print("="*60)
# Pass the separate training and testing dataframes to the RF function
rf_results = train_and_evaluate_random_forest(
    train_df=train_df_master.copy(), # Pass a copy
    test_df=test_df_master.copy(),   # Pass a copy
    sma_period=RF_SMA_PERIOD,
    n_estimators=RF_N_ESTIMATORS,
    random_state=RANDOM_STATE
)
all_results.update(rf_results)


# Run LSTM Model
print("\n" + "="*60)
print("--- Running LSTM Model ---")
print("="*60)
# Pass the separate training and testing dataframes to the LSTM function
lstm_results = train_and_evaluate_lstm(
    train_df=train_df_master.copy(), # Pass a copy
    test_df=test_df_master.copy(),   # Pass a copy
    sequence_length=LSTM_SEQUENCE_LENGTH,
    batch_size=LSTM_BATCH_SIZE,
    epochs=LSTM_EPOCHS,
    lstm_units=LSTM_UNITS,
    random_state=RANDOM_STATE # Note: LSTM random state also set globally in lstm_model.py
)
all_results.update(lstm_results)


# --- Summary of Results ---
print("\n" + "="*60)
print("--- Overall Model Performance Summary (2024 Test Data) ---")
print("="*60)

# Print results from the collected dictionary
if "HMM_RMSE" in all_results:
    print(f"Hidden Markov Model (Price Prediction):")
    # Use .get() with a default value like 'N/A' in case a model failed
    print(f"  RMSE: {all_results.get('HMM_RMSE', 'N/A'):.4f}" if isinstance(all_results.get('HMM_RMSE'), (int, float)) else f"  RMSE: {all_results.get('HMM_RMSE', 'N/A')}")
    print(f"  MAE: {all_results.get('HMM_MAE', 'N/A'):.4f}" if isinstance(all_results.get('HMM_MAE'), (int, float)) else f"  MAE: {all_results.get('HMM_MAE', 'N/A')}")
    print("-" * 30)

if "RF_Accuracy" in all_results:
    print(f"Random Forest (Direction Prediction):")
    print(f"  Accuracy: {all_results.get('RF_Accuracy', 'N/A'):.4f}" if isinstance(all_results.get('RF_Accuracy'), (int, float)) else f"  Accuracy: {all_results.get('RF_Accuracy', 'N/A')}")
    print(f"  Precision: {all_results.get('RF_Precision', 'N/A'):.4f}" if isinstance(all_results.get('RF_Precision'), (int, float)) else f"  Precision: {all_results.get('RF_Precision', 'N/A')}")
    print(f"  Recall: {all_results.get('RF_Recall', 'N/A'):.4f}" if isinstance(all_results.get('RF_Recall'), (int, float)) else f"  Recall: {all_results.get('RF_Recall', 'N/A')}")
    print(f"  F1-Score: {all_results.get('RF_F1-Score', 'N/A'):.4f}" if isinstance(all_results.get('RF_F1-Score'), (int, float)) else f"  F1-Score: {all_results.get('RF_F1-Score', 'N/A')}")
    print("-" * 30)

if "LSTM_RMSE" in all_results:
    print(f"LSTM (Price Prediction):")
    print(f"  RMSE: {all_results.get('LSTM_RMSE', 'N/A'):.4f}" if isinstance(all_results.get('LSTM_RMSE'), (int, float)) else f"  RMSE: {all_results.get('LSTM_RMSE', 'N/A')}")
    print(f"  MAE: {all_results.get('LSTM_MAE', 'N/A'):.4f}" if isinstance(all_results.get('LSTM_MAE'), (int, float)) else f"  MAE: {all_results.get('LSTM_MAE', 'N/A')}")
    print("-" * 30)

print("\nAnalysis Complete. Check individual model plots for visualizations.")
