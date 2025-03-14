import torch
import torch.nn as nn

class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, text_dim, hidden_dim, output_dim):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + text_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x_financial, x_text):
        lstm_out, _ = self.lstm(x_financial)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_out, x_text), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x