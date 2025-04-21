# === File: main_code.py ===
# This script loads SPY data, calls functions from separate model files
# to train and evaluate HMM, Random Forest, and LSTM models,
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
# DATA_FILE should contain data from 2000 through 2024
DATA_FILE = 'spy_2000_2023.csv'
TEST_START_DATE = '2024-01-01'

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
print(f"--- Loading Data ---")
if not os.path.exists(DATA_FILE):
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure '{DATA_FILE}' is in the correct directory.")
    print("You can create this file using the data download script provided earlier.")
    exit()

try:
    # Attempt to read the CSV without setting the index column first
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data from {DATA_FILE}")
    print(f"Columns found in the CSV: {df.columns.tolist()}") # Print column names

    # --- Identify and Set Date Column ---
    # Common names for the date column when saving index=True from yfinance
    date_column_candidates = ['Date', 'date', 'DATE', 'Unnamed: 0']
    date_column_name = None

    for col in date_column_candidates:
        if col in df.columns:
            date_column_name = col
            break # Found a likely date column

    if date_column_name:
        print(f"Identified '{date_column_name}' as the potential date column.")
        # Set the identified column as the index and parse as dates
        df.set_index(date_column_name, inplace=True)
        df.index = pd.to_datetime(df.index) # Ensure the index is datetime objects
        print("Successfully set date column as index.")
    else:
        print("Error: Could not find a suitable date column ('Date', 'date', 'DATE', 'Unnamed: 0').")
        print("Please check your CSV file and update the 'date_column_candidates' list or the file.")
        exit()

    print(f"Full data shape after setting index: {df.shape}")
    print("First 5 dates in index:", df.index[:5].tolist())


except Exception as e:
    print(f"An error occurred during data loading or processing: {e}")
    exit()

# Ensure data is sorted by date
df.sort_index(inplace=True)

# --- Data Splitting (Master Split for Train/Test) ---
# Split the full DataFrame into training (before TEST_START_DATE) and testing (from TEST_START_DATE onwards)
# Use .copy() to avoid SettingWithCopyWarning
train_df_master = df.loc[df.index < TEST_START_DATE].copy()
test_df_master = df.loc[df.index >= TEST_START_DATE].copy()

# Check if test data exists
if test_df_master.empty:
    print(f"Error: No test data found on or after {TEST_START_DATE}.")
    print("Please ensure your data file includes data for 2024.")
    exit()

print(f"Master Training data shape: {train_df_master.shape}")
print(f"Master Testing data shape: {test_df_master.shape}")


# --- Run Models and Collect Results ---
all_results = {}

# Run HMM Model
print("\n" + "="*60)
print("--- Running HMM Model ---")
print("="*60)
# Pass copies of the master dataframes to model functions to prevent unintended modifications
hmm_results = train_and_evaluate_hmm(
    train_df=train_df_master.copy(),
    test_df=test_df_master.copy(),
    n_states=HMM_N_STATES,
    random_state=RANDOM_STATE
)
all_results.update(hmm_results)


# Run Random Forest Model
print("\n" + "="*60)
print("--- Running Random Forest Model ---")
print("="*60)
# Pass copies of the master dataframes to model functions
rf_results = train_and_evaluate_random_forest(
    train_df=train_df_master.copy(),
    test_df=test_df_master.copy(),
    sma_period=RF_SMA_PERIOD,
    n_estimators=RF_N_ESTIMATORS,
    random_state=RANDOM_STATE
)
all_results.update(rf_results)


# Run LSTM Model
print("\n" + "="*60)
print("--- Running LSTM Model ---")
print("="*60)
# Pass copies of the master dataframes to model functions
lstm_results = train_and_evaluate_lstm(
    train_df=train_df_master.copy(),
    test_df=test_df_master.copy(),
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
