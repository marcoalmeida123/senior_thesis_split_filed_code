import yfinance as yf
import pandas as pd
import os # Import the os module to check if the file exists

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2000-01-01"
END_DATE = "2023-12-31" # Note: yfinance end date is exclusive, so this gets data up to Dec 30, 2023
FILE_NAME = "spy_2000_2023.csv"

# --- Download Data ---
print(f"Attempting to download {TICKER} data from {START_DATE} to {END_DATE}...")

try:
    # Download the data
    # The download function returns a pandas DataFrame
    spy_data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    if not spy_data.empty:
        print("Data downloaded successfully.")
        print(f"Downloaded data shape: {spy_data.shape}")
        print("First 5 rows of downloaded data:")
        print(spy_data.head())
        print("\nLast 5 rows of downloaded data:")
        print(spy_data.tail())

        # --- Save Data to CSV ---
        print(f"\nAttempting to save data to {FILE_NAME}...")

        # Save the DataFrame to a CSV file
        # index=True saves the Date index as a column in the CSV
        spy_data.to_csv(FILE_NAME, index=True)

        # Verify if the file was created
        if os.path.exists(FILE_NAME):
            print(f"Data successfully saved to {FILE_NAME}")
        else:
            print(f"Error: File {FILE_NAME} was not created.")

    else:
        print(f"No data downloaded for {TICKER} in the specified date range.")
        print("Please check the ticker symbol and date range.")

except Exception as e:
    print(f"An error occurred during data download or saving: {e}")

