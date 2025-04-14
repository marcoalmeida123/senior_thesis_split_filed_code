# === File: main_code.py ===
from data_loader import load_spy_data
from hmm_model import add_hmm_states
from random_forest_model import train_random_forest
from lstm_model import train_lstm_model

# 1. Load and preprocess data
data = load_spy_data()

# 2. Train HMM and add hidden states
data, hmm_model = add_hmm_states(data)

# 3. Train Random Forest model
rf_model, X_test_rf, y_test_rf = train_random_forest(data)

# 4. Train LSTM model on close price
lstm_model, lstm_scaler = train_lstm_model(data[['Close']].values)

print("Models trained successfully.")
 
