import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error

# load data prepared by the lstm_dataprep.py script
ticker = "BB.TO" 
X_train = torch.from_numpy(np.load(f'X_train_{ticker}.npy')).float()
y_train = torch.from_numpy(np.load(f'y_train_{ticker}.npy')).float()
X_test = torch.from_numpy(np.load(f'X_test_{ticker}.npy')).float()
y_test = torch.from_numpy(np.load(f'y_test_{ticker}.npy')).float()

# define LSTM Model Architecture
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape (batch, seq_len, features)
        out, _ = self.lstm(x)
        # only care about last time steps output for prediction
        out = self.fc(out[:, -1, :]) 
        return out

# initialize Mmdel, loss, and optimizer
# input_size is 1 (price), hidden_size is a hyperparameter (trying 50)
model = SimpleLSTM(input_size=1, hidden_size=50, output_size=1)
criterion = nn.L1Loss() #mean absolute error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# simple training loop
print("starting training")
for epoch in range(10):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

print("training finished.")