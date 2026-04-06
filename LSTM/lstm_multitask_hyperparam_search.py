"""
Multi-task LSTM hyperparameter ablation search

Description:
This script performs systematic grid search across 3 key hyperparameters:
1. number of LSTM layers, testing depth vs vanishing gradients
2. hidden layer size, test model capacity vs overifitting
3. learning rate, optimizig convergence speed and stability 

Methodology:
- each configuration is trained for 50 epochs on a syncronized 26 stock dataset.
- final choices: 2 layers, 128 hidden, 0.0005 lt chose based on miimal validation MAE to ensure max generalization
"""
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# model architecture
class DeepMultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, tickers, dropout=0.2):
        super().__init__()
        self.tickers = tickers
        # multi-task: shared lstm layers to learn corss stock correlations
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)
        # multitask heads: independent layers for each stock (ticker)
        self.heads = nn.ModuleDict({
            ticker.replace('.', '_'): nn.Linear(hidden_size, 1) for ticker in tickers
        })

    def forward(self, x):
        #pass through shared LSTM backbone
        lstm_out, _ = self.lstm(x)
        # use hidden state of final time step 
        h = lstm_out[:, -1, :] 
        h = self.dropout_layer(h)
        #pass shared representation through ticker-specific heads
        return {ticker: self.heads[ticker.replace('.', '_')](h) for ticker in self.tickers}

# DATA PREP
# loading pre synchronized numpy files from global data merge
X_train = torch.from_numpy(np.load('X_train_global.npy')).float()
y_train = torch.from_numpy(np.load('y_train_global.npy')).float()
X_val = torch.from_numpy(np.load('X_val_global.npy')).float()
y_val = torch.from_numpy(np.load('y_val_global.npy')).float()

with open('tickers.txt', 'r') as f:
    tickers = f.read().split(',')

num_stocks = len(tickers)

# HYPERPARAM CONFIG
# to test layers: configs = [1, 2, 4, 6]
# to test hidden: configs = [25, 50, 75, 100, 125, 150, 200]
lr_configs = [0.01, 0.005, 0.001, 0.0005, 0.0001]

train_maes = []
val_maes = []

# winners picked from prior ablation passes
FIXED_LAYERS = 2
FIXED_HIDDEN = 128

print(f"Starting Learning Rate Tuning for {len(lr_configs)} configurations")

try:
    for lr in lr_configs:
        print(f"\nTesting: Learning Rate: {lr}")
        
        model = DeepMultiTaskLSTM(num_stocks, FIXED_HIDDEN, FIXED_LAYERS, tickers, dropout=0.2)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        
        # 50 epochs picked cuz sufficent to observe convergence trends without over training 
        for epoch in range(50):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                # aggregate MAE loss across all 26 specific heads
                loss = sum(criterion(predictions[t], batch_y[:, i].unsqueeze(1)) for i, t in enumerate(tickers)) / num_stocks
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch+1}/50 | Current Loss: {epoch_loss/len(train_loader):.4f}")

        # capture final train performance
        train_maes.append(epoch_loss/len(train_loader))

        # evaluate on valid set, generalization check 
        model.eval()
        with torch.no_grad():
            v_preds = model(X_val)
            v_loss = sum(criterion(v_preds[t], y_val[:, i].unsqueeze(1)) for i, t in enumerate(tickers)) / num_stocks
            val_maes.append(v_loss.item())
            print(f">> LR {lr} Finished | Val MAE: {v_loss.item():.4f}")

except KeyboardInterrupt:
    print("\nexecution halted, compile partial results") # so my laptop doesnt explode

# visualization
if len(val_maes) > 0:
    plt.figure(figsize=(12, 7))
    x = np.arange(len(val_maes))
    width = 0.35

    plt.bar(x - width/2, train_maes, width, label='Train MAE', color='royalblue', alpha=0.7)
    plt.bar(x + width/2, val_maes, width, label='Validation MAE', color='orange', alpha=0.7)

    plt.title('MAE by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(x, [str(lr_configs[i]) for i in range(len(val_maes))])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # fig caption
    caption = "Figure: A visualization of the training and validation MAE across different learning rates for the Multi-Task LSTM."
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10, style='italic')
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('Learning_Rates_Comparison_Multitask.png')
    print("\nSuccess! Graph saved as: Learning_Rates_Comparison_Multitask.png")
    plt.show()