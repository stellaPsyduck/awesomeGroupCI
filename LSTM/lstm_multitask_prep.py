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

#setup
# define categories and base path
base_path = '../DownloadCSV/'
categories = ["IShare", "LargeCompany", "MediumCompany", "SmallCompany"]
seq_length = 60 # 60 business days of lookback

# container for merged synchronized dataset 
global_df = None

print("Starting global data merging (full outer join startegy)")

#data merging and alignment 
for cat in categories:
    folder_path = os.path.join(base_path, cat)
    
    if not os.path.exists(folder_path):
        print(f"Skipping {cat}: Folder not found")
        continue

    # Get all CSV files across all folders
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file_name in all_files:
        path = os.path.join(folder_path, file_name)
        
        try:
            # standardizing input: skip headers and renaming cols
            df = pd.read_csv(path, skiprows=2)
            df.columns = ['Date', 'Price']
            
            # extract ticker from filename, like IBM_unadjusted_prices_2007_2018.csv  -> IBM
            ticker = file_name.split('_')[0]
            
            #apply  natural log to stabalize variance 
            df[ticker] = np.log(df['Price'])
            df = df[['Date', ticker]]
            
            # gsynch with global timeline 
            if global_df is None:
                global_df = df
            else:
                # merge on date to ensure all stocks align on same trading days 
                global_df = pd.merge(global_df, df, on='Date', how='outer')
            
            print(f" Merged {ticker} (Outer)")
        except Exception as e:
            print(f"Error: {e}")

# clean up, rmeove any dates where not all stocks have avaibale date  (synchronized window)
global_df.dropna(inplace=True)
global_df = global_df.sort_values('Date')

# identify all synchronized tickers (features)
tickers = [col for col in global_df.columns if col != 'Date']
print(f"\nPreprocessing Complete: {len(tickers)} stocks synchronized.")
print(f"Final Data Shape: {global_df.shape}")

# Save ticker names for the models output heads
with open('tickers.txt', 'w') as f:
    f.write(",".join(tickers))

# Sequence creation
# transform dataframe into multi-dimenosional numpy arr [Days, Stocks]
final_data = global_df[tickers].values
x_list, y_list = [], []

# create sliding windows: X = [t-60 to t-1], y=[t]
for i in range(len(final_data) - seq_length):
    x_list.append(final_data[i:(i + seq_length), :]) # multi stock input window
    y_list.append(final_data[i + seq_length, :]) # next day prices for all stocks 
    
X = np.array(x_list)
y = np.array(y_list)

#chronological data split
# Split 60/20/20 train/valid//test
train_split = int(0.6 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

#export deliverables
# Save as NumPy binaries for loading in train script
np.save('X_train_global.npy', X_train)
np.save('y_train_global.npy', y_train)
np.save('X_val_global.npy', X_val)
np.save('y_val_global.npy', y_val)
np.save('X_test_global.npy', X_test)
np.save('y_test_global.npy', y_test)

print(f"preprocessing complete. train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")