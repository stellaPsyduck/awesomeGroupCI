#LSTM Data Prep (single task)
# script takes CSV and transforms it into [Samples, Time Steps, Features] format

# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

#load and clean
ticker = "BB.TO"
dataset = pd.read_csv(f'../DownloadCSV/IShare/{ticker}_unadjusted_prices_2007_2018.csv', skiprows=2)
dataset.rename(columns={"Unnamed: 1": "Price"}, inplace=True)

# scaling
#use natural log
dataset['Log_Price'] = np.log(dataset['Price'])
#shift data so centered around 0
log_data = dataset['Log_Price'].values.reshape(-1, 1)

# create sliding windows
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)]) # look back window
        y.append(data[i + seq_length])    # target (next days price)
    return np.array(x), np.array(y)

window_size = 120 # 120 day lag goal
X, y = create_sequences(log_data, window_size)

# chronological split (no shuffling)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# verify 3d shape
print(f"X_train shape: {X_train.shape}") 
# should be: (samples, 120, 1)

# save processed arrays as NumPy files
np.save(f'X_train_{ticker}.npy', X_train)
np.save(f'X_test_{ticker}.npy', X_test)
np.save(f'y_train_{ticker}.npy', y_train)
np.save(f'y_test_{ticker}.npy', y_test)

print(f"Pre-processing complete. Processed data saved as: X_train_{ticker}.npy")