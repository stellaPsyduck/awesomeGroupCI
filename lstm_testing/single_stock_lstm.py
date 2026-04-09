import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch.nn as nn

# https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/
# https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://github.com/senadkurtisi/Multivariate-Time-Series-Forecast/blob/main/src/model.py

# Notes:
    # Something called time-aware LSTM, or T-LSTM
    # seems really cool, but given the time constraints doesn't seem like we would have the time to implement this:
    # https://github.com/illidanlab/T-LSTM/blob/master/TLSTM.py

# Initialization coming from automatic setting from pytorch

# Single layer LSTM, can expand, or make larger depending on time constraints and group wants
# Takes in data as such:
    # (num_samples, seq_length, input_dim)
    # This is because batch_first = true
class UnivariantLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2, output_size=1):
        super().__init__()

        if num_layers == 1:
            dropout = 0.0

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, (hidden, _) = self.lstm(input)

        out = hidden[-1, :, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out