"""
Final optimized multi task lstm train script

description:
Trains a deep multi task lstm to predicit synchronized stocks simultaneously.
This architecture leverages shared temporal features to improve generalization 
across market.

Hyperparameters:
- Layers: 2 (optimized with ablation search)
- Hidden dim: 128 (optimized with ablation search and standard power of 2 for GPU efficency )
- Learning rate: 0.0005 ( optimal for learning clif convergence)
- epochs: 200 (ensures foll loss stabaliztaion )
"""
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#model architectire
class DeepMultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, tickers, dropout=0.2):
        super().__init__()
        self.tickers = tickers
        
        #shared lstm backbone, learns global market dependencies 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        
        # dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # task specific heads: indeoendnet output layers for eachh synchronized stock
        # sanitizes '.' to '_' for PyTorch module compatibility
        self.heads = nn.ModuleDict({
            ticker.replace('.', '_'): nn.Linear(hidden_size, 1) for ticker in tickers
        })

    def forward(self, x):
        # x shape: (batch, seq_length, num_stocks)
        lstm_out, _ = self.lstm(x)
        
        # feature pooling: extract represnetatio from final time step 
        h = lstm_out[:, -1, :] 
        h = self.dropout_layer(h)

        # multi-task inference, each head predicts its specific stock price
        return {ticker: self.heads[ticker.replace('.', '_')](h) for ticker in self.tickers}

#data loading and configuration 
# load synchronized 60/20/20 datasets
X_train = torch.from_numpy(np.load('X_train_global.npy')).float()
y_train = torch.from_numpy(np.load('y_train_global.npy')).float()
X_val = torch.from_numpy(np.load('X_val_global.npy')).float()
y_val = torch.from_numpy(np.load('y_val_global.npy')).float()
X_test = torch.from_numpy(np.load('X_test_global.npy')).float()
y_test = torch.from_numpy(np.load('y_test_global.npy')).float()

with open('tickers.txt', 'r') as f:
    tickers = f.read().split(',')

# oprimied hyperparameters from search results 
num_stocks = len(tickers)
hidden_dim = 128 #optimal capacity for 26 stocks without extreme overfittig 
layers_count = 2 
learning_rate = 0.0005
batch_size = 64
epochs = 200

# initialization 
model = DeepMultiTaskLSTM(num_stocks, hidden_dim, layers_count, tickers)
criterion = nn.L1Loss() # MAE 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
train_losses = []
val_losses = []

print(f"Starting final training: {layers_count}-Layer Model | {num_stocks} Stocks | {epochs} Epochs")

#train loop 
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0

    for batch_X, batch_y in train_loader: 
        optimizer.zero_grad()
        predictions = model(batch_X)
        
        # aggregate loss: avg MAE across all task specific heads
        loss = sum(criterion(predictions[t], batch_y[:, i].unsqueeze(1)) 
                   for i, t in enumerate(tickers)) / num_stocks
    
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # log train and valid loss for epoch 
    avg_epoch_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    model.eval()
    with torch.no_grad():
        v_preds = model(X_val)
        val_loss = sum(criterion(v_preds[t], y_val[:, i].unsqueeze(1)) 
                       for i, t in enumerate(tickers)) / num_stocks
        val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/200] | Train MAE: {avg_epoch_loss:.4f} | Val MAE: {val_loss.item():.4f}")


# final evaluation and export 
model.eval()
with torch.no_grad():
    # Validation still uses the whole X_val set at once
    t_preds = model(X_test)
    test_mae = sum(criterion(t_preds[t], y_test[:, i].unsqueeze(1)) 
                   for i, t in enumerate(tickers)) / num_stocks

print(f"\n FINAL EVALUATION RESULT (Test MAE): {test_mae.item():.4f}")

# save state dictionary
torch.save(model.state_dict(), 'final_4layer_multitask.pth')

# output detailed per stock erformance metrics 
print("\nTEST MAE PER STOCK")
for i, t in enumerate(tickers):
    stock_mae = criterion(t_preds[t], y_test[:, i].unsqueeze(1))
    print(f"{t:10} | MAE: {stock_mae.item():.4f}")

# generate and save final learning curve deliverable 
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss (MAE)', color='blue')
plt.plot(val_losses, label='Validation Loss (MAE)', color='orange')
plt.title('Multi-Task LSTM Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
caption_text = "Figure: Final training and validation MAE for the optimized 2-layer Multi-Task LSTM over 200 epochs."
plt.figtext(0.5, 0.01, caption_text, ha="center", fontsize=10, wrap=True, style='italic')
plt.savefig('lstm_multitask_train_vs_val_loss.png') 
plt.show()