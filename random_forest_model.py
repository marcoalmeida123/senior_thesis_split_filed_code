# === File: random_forest_model.py ===
# Contains functions for training, predicting, and evaluating the Random Forest Classifier.

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

    Args:
        train_df (pd.DataFrame): Training data DataFrame (2000-2023).
                                 Must contain a 'Close' column.
        test_df (pd.DataFrame): Testing data DataFrame (2024).
                                Must contain a 'Close' column.
        sma_period (int): Period for Simple Moving Average feature.
        n_estimators (int): Number of trees in the Random Forest.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing classification evaluation metrics
              (Accuracy, Precision, Recall, F1-Score).
    """
    print("\n--- Running Random Forest Model ---")

    # --- Data Preparation for Random Forest ---
    # RF requires Daily_Return, SMA, and Target features
    train_df_rf = train_df.copy()
    test_df_rf = test_df.copy()

    # Calculate features for both train and test sets
    for df_rf in [train_df_rf, test_df_rf]:
        df_rf['Daily_Return'] = df_rf['Close'].pct_change()
        df_rf['SMA_' + str(sma_period)] = df_rf['Close'].rolling(window=sma_period).mean()
        # Target: 1 if the next day's close is higher, 0 otherwise
        df_rf['Target'] = (df_rf['Close'].shift(-1) > df_rf['Close']).astype(int)

    # Drop NaNs created by feature engineering and target shift
    train_df_rf.dropna(inplace=True)
    test_df_rf.dropna(inplace=True)

    if train_df_rf.empty or test_df_rf.empty:
        print("Not enough data after feature engineering and dropping NaNs for Random Forest.")
        return {"RF_Accuracy": np.nan, "RF_Precision": np.nan, "RF_Recall": np.nan, "RF_F1-Score": np.nan}

    # Define features and target for RF
    features_rf = ['Daily_Return', 'SMA_' + str(sma_period)]
    target_rf = 'Target'

    X_train_rf = train_df_rf[features_rf]
    y_train_rf = train_df_rf[target_rf]
    X_test_rf = test_df_rf[features_rf]
    y_test_rf = test_df_rf[target_rf]

    # Optional: Scale RF features (less critical for tree models)
    scaler_rf = StandardScaler()
    cols_to_scale_rf = [col for col in features_rf if col in X_train_rf.columns]
    if cols_to_scale_rf:
        X_train_rf[cols_to_scale_rf] = scaler_rf.fit_transform(X_train_rf[cols_to_scale_rf])
        X_test_rf[cols_to_scale_rf] = scaler_rf.transform(X_test_rf[cols_to_scale_rf])


    print("Training Random Forest Classifier...")
    model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model_rf.fit(X_train_rf, y_train_rf)
    print("Random Forest training complete.")

    # --- Random Forest Prediction ---
    print("Making Random Forest Predictions...")
    predictions_rf = model_rf.predict(X_test_rf)

    # --- Random Forest Evaluation (Classification Metrics) ---
    print("Evaluating Random Forest Predictions...")
    accuracy_rf = accuracy_score(y_test_rf, predictions_rf)
    precision_rf = precision_score(y_test_rf, predictions_rf)
    recall_rf = recall_score(y_test_rf, predictions_rf)
    f1_rf = f1_score(y_test_rf, predictions_rf)
    conf_matrix_rf = confusion_matrix(y_test_rf, predictions_rf)
    class_report_rf = classification_report(y_test_rf, predictions_rf)


    print(f"Random Forest Accuracy for test period: {accuracy_rf:.4f}")
    print(f"Random Forest Precision for test period: {precision_rf:.4f}")
    print(f"Random Forest Recall for test period: {recall_rf:.4f}")
    print(f"Random Forest F1-Score for test period: {f1_rf:.4f}")
    print("\nConfusion Matrix (RF):")
    print(conf_matrix_rf)
    print("\nClassification Report (RF):")
    print(class_report_rf)

    # --- Random Forest Visualization (Direction) ---
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_rf.index, y_test_rf.values, label='Actual Direction (1=Up, 0=Down/Flat)', color='blue', marker='.', linestyle='None', alpha=0.6)
    plt.plot(y_test_rf.index, predictions_rf, label='Predicted Direction (1=Up, 0=Down/Flat)', color='red', marker='x', linestyle='None', alpha=0.6)
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
#     df_full = pd.read_csv('spy_2000_2023.csv', index_col='Date', parse_dates=True)
#     df_full.sort_index(inplace=True)
#     train_df_test = df_full.loc[df_full.index < '2024-01-01'].copy()
#     test_df_test = df_full.loc[df_full.index >= '2024-01-01'].copy()
#     rf_results = train_and_evaluate_random_forest(train_df_test, test_df_test)
#     print("\nStandalone RF Results:", rf_results)
