"""
Multi task lstm data preprocessing script

This script prforms global synch of stock price data across multiple market cap categories
This prepares a multi dimensional feature set

key steps:
1. synchronized merging: using full outer join strategy to align all stocks by date
2. log transformation: applies natural log scaling to normalize price variance 
3. temporal windowing: generates sliding window sequences, length=60 for lstm input
4. global splitting: maintaings order with a 60/20/20 train/val/test split 
"""
import os
import numpy as np
import pandas as pd

# Path to where download scripts saved CSVs
base_path = '../DownloadCSV/'
seq_length = 60 

TARGET_TICKERS = [
    "CLS.TO", "GIB-A.TO", "OTEX.TO", "DSG.TO", "BB.TO", # iShare
    "AMD", "IBM", "CSCO", "AAPL", "MSFT", "ORCL", "INTC", "CRM",  # Large
    "NOVT", "VSAT", "BDC", "SLAB", "OLED", "ACIW", "BMI", # Medium
    "BELFB", "BLKB", "NTCT", "UCTT", "PLAB" # Small
]

global_df = None

print(f"Starting synchronized merge for {len(TARGET_TICKERS)} target stocks")

# data merging
# check the base_path directly since scripts save files there
if not os.path.exists(base_path):
    print(f"!!! ERROR: Folder '{base_path}' not found.")
    print(f"Current Directory: {os.getcwd()}")
    exit()

all_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
print(f"Files found in {base_path}: {all_files[:5]}... (Total: {len(all_files)})")

for ticker in TARGET_TICKERS:
    # use a case-insensitive, flexible start-match
    match = [f for f in all_files if f.upper().startswith(ticker.upper())]
    
    if match:
        file_name = match[0]
        path = os.path.join(base_path, file_name)
        try:
            # Try reading without headers first to see struct
            df = pd.read_csv(path)
        
            if 'Date' not in df.columns:
                df = pd.read_csv(path, names=['Date', 'Close'], skiprows=1)
            
            # Standardizing cols
            df = df.iloc[:, [0, 1]] # Take first 2 columns regardless of name
            df.columns = ['Date', 'Price']
            
            # Log transform
            df[ticker] = np.log(pd.to_numeric(df['Price'], errors='coerce'))
            df = df[['Date', ticker]].dropna()
            
            if global_df is None:
                global_df = df
            else:
                global_df = pd.merge(global_df, df, on='Date', how='outer')
            
            print(f"  [OK] Merged: {ticker}")
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
    else:
        print(f"  [MISSING] No file found starting with '{ticker}'")

# cleaning
if global_df is None:
    print("\n!!! ERROR: No data was merged. Check if the CSVs are in " + os.path.abspath(base_path))
    exit()

# Synchronize: keep only dates where ALL stocks have data
global_df.dropna(inplace=True)
global_df = global_df.sort_values('Date')

# Final ticker list (only those that acc had data)
final_tickers = [t for t in TARGET_TICKERS if t in global_df.columns]
global_df = global_df[['Date'] + final_tickers]

print(f"\nPreprocessing Complete: {len(final_tickers)} stocks synchronized.")
print(f"Final Data Shape: {global_df.shape}")

# Save tickers for the model
with open('tickers.txt', 'w') as f:
    f.write(",".join(final_tickers))

# sequence creation and splitting 
final_data = global_df[final_tickers].values
x_list, y_list = [], []

for i in range(len(final_data) - seq_length):
    x_list.append(final_data[i:(i + seq_length), :]) 
    y_list.append(final_data[i + seq_length, :]) 
    
X = np.array(x_list)
y = np.array(y_list)

# 60/20/20 Split
train_split = int(0.6 * len(X))
val_split = int(0.8 * len(X))

# Export binaries
np.save('X_train_global.npy', X[:train_split])
np.save('y_train_global.npy', y[:train_split])
np.save('X_val_global.npy', X[train_split:val_split])
np.save('y_val_global.npy', y[train_split:val_split])
np.save('X_test_global.npy', X[val_split:])
np.save('y_test_global.npy', y[val_split:])

print(f"Successfully exported data for {len(final_tickers)} stocks.")