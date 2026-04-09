import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Need to test on IBM and with LSTM

# https://docs.pytorch.org/docs/stable/generated/torch.reshape.html

# Need to be able to take in the data on a timeseries-window based way
# log scaling

# Stealing stella code for this, as it basically does what we talked about
# LSTM is different from RF, so will need to be slightly different

# ticker = "IBM"
# Should skip 3 rows
# base_dir = Path(__file__).resolve().parent.parent
# file_path = base_dir / "DownloadCSV" / "LargeCompany" / f"{ticker}_unadjusted_prices_2007_2018.csv"

# dataset = pd.read_csv(file_path, skiprows=2, names=['Date', 'Price'])

# dataset = dataset.dropna().reset_index(drop=True)

# log the values
# prices = np.log(dataset["Price"].values)

# Stella used ranges of 60-65 (4 months) --> business days not actual months
# seq_length = 60

# Need to create the windows
def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i : (i + seq_len)] 
        y = data[i + seq_len]      
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def get_dataloaders(ticker="IBM", seq_length=60, batch_size=10):
    base_dir = Path(__file__).resolve().parent.parent

    if ticker == 'BB.TO':
        companyFolder = "IShare"
    else:
        companyFolder = "LargeCompany"

    file_path = base_dir / "DownloadCSV" / companyFolder / f"{ticker}_unadjusted_prices_2007_2018.csv"

    dataset = pd.read_csv(file_path, skiprows=2, names=['Date', 'Price'])
    dataset = dataset.dropna().reset_index(drop=True)

    prices = np.log(dataset["Price"].values)
    X, y = create_sequences(prices, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    total_samples = len(X)
    train_end = int(total_samples * 0.6) 
    val_end = int(total_samples * 0.8) 

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float32)

    train_data = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_data = TensorDataset(to_tensor(X_val), to_tensor(y_val))
    test_data = TensorDataset(to_tensor(X_test), to_tensor(y_test))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# Good information, just in case


# X, y = create_sequences(prices, seq_length)

# Takes in data as such:
    # (num_samples, seq_length, input_dim)
    # This is because batch_first = true
# Therefore, need to change to a tensor, only have one real feature which is price
# X = X.reshape(X.shape[0], X.shape[1], 1)

# convert to tensor
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)

# Batch size is currently unknown? How many months do we want to train on?
# batch_size = 10

# dataset_test = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# print(f"X_tensor shape: {X_tensor.shape}") 
# print(f"y_tensor shape: {y_tensor.shape}")

# X_tensor shape: torch.Size([2709, 60, 1])
# y_tensor shape: torch.Size([2709])

# So total of 45.15 windows at the moment, which should be the case of all models,
# might need to handle the differences in endings, as there might be corrupt data within the 
# CSVs, which is fine

