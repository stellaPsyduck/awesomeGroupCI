import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
TARGET_STOCK = 'BB.TO'  # 'IBM' 'BB.TO'
MODEL_PATH = 'final_2layer_multitask.pth'

# model definition
class DeepMultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, tickers, dropout=0.2):
        super().__init__()
        self.tickers = tickers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)
        # Create 1 output head for each stock
        self.heads = nn.ModuleDict({
            t.replace('.', '_'): nn.Linear(hidden_size, 1) for t in tickers
        })

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :] # Take last time step
        h = self.dropout_layer(h)
        # Return preds for all stocks as a dictionary
        return {t: self.heads[t.replace('.', '_')](h) for t in self.tickers}

# load data and model
# Load ticker list used during train
with open('tickers.txt', 'r') as f:
    current_tickers = f.read().split(',')

# Initialize and load saved weights
model = DeepMultiTaskLSTM(len(current_tickers), 128, 2, current_tickers)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Combine all data splits to get a full timeline for 2017
X_all = np.concatenate([np.load('X_train_global.npy'), 
                        np.load('X_val_global.npy'), 
                        np.load('X_test_global.npy')], axis=0)

y_all = np.concatenate([np.load('y_train_global.npy'), 
                        np.load('y_val_global.npy'), 
                        np.load('y_test_global.npy')], axis=0)

# Slice last 261 days (approx. 1 trading year)
full_year_X = torch.from_numpy(X_all[-261:]).float()
full_year_y = y_all[-261:]

# run preds
with torch.no_grad():
    predictions = model(full_year_X)
    
    # find right key ( BB.TO -> BB_TO)
    key_options = [TARGET_STOCK, TARGET_STOCK.replace('.', '_')]
    target_key = next((k for k in key_options if k in predictions), None)
    
    if not target_key:
        print(f"Error: {TARGET_STOCK} not found. Check: {list(predictions.keys())}")
        exit()
        
    preds_logged = predictions[target_key].numpy().flatten()

# Get actual values for target stock
target_idx = current_tickers.index(TARGET_STOCK)
true_logged = full_year_y[:, target_idx]

# prep datadrame
df = pd.DataFrame({
    'Time Step': np.arange(len(true_logged)),
    'Actual': true_logged,
    'Predicted': preds_logged,
    'Naive': pd.Series(true_logged).shift(1) # yesterdays price
})

# caclulate daily error 
df['error'] = df['Actual'] - df['Predicted']
# calculate global standard deviaition, non rolling
global_std = df['error'].std()
# use 1x standard deviation
df['lower'] = df['Predicted'] - global_std
df['upper'] = df['Predicted'] + global_std

# plotting
plt.figure(figsize=(10, 6), dpi=300)

# gray ribbon (uncertainty)
plt.fill_between(df['Time Step'], df['lower'], df['upper'], color='gray', alpha=0.1, label='95% CI (Variance)')

# actual price (black)
plt.plot(df['Time Step'], df['Actual'], color='black', lw=1.2, label='Actual Price', zorder=4)

# Naive forecast (red)
plt.plot(df['Time Step'], df['Naive'], color='red', ls=':', lw=0.8, label='Naive Forecast (1-Day Shift)', zorder=3)

# LSTM pred (blue)
plt.plot(df['Time Step'], df['Predicted'], color='blue', ls='--', lw=1.0, label='LSTM Predicted Price', zorder=2)

plt.title('LSTM Model - Actual vs LSTM vs Naive', fontsize=12)
plt.xlabel('Time Step', fontsize=10)
plt.ylabel('Price (Log)', fontsize=10)
plt.grid(True, alpha=0.15)
plt.legend(loc='upper left', frameon=True, fontsize=8)
plt.ylim(df['Actual'].min() - 0.1, df['Actual'].max() + 0.3)

# Save
plt.tight_layout()
plt.savefig(f'multi_lstm_{TARGET_STOCK}_EVALGRAPH.png')
print(f"Saved: multi_lstm_{TARGET_STOCK}_EVALGRAPH.png")