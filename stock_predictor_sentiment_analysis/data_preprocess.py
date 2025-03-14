import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load historical stock data and macroeconomic indicators
def load_financial_data(stock_path, macro_path):
    stock_data = pd.read_csv(stock_path, parse_dates=['Date'], index_col='Date')
    macro_data = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
    data = stock_data.join(macro_data, how='inner')
    return data

# Load and preprocess news & social media data
def load_text_data(news_path, social_path):
    news_data = pd.read_csv(news_path)
    social_data = pd.read_csv(social_path)
    text_data = pd.concat([news_data, social_data])
    return text_data['content'].tolist()

# Tokenize text using FinBERT or Llama 3.2
def tokenize_text(texts, model_name='ProsusAI/finbert'): 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return inputs

# Extract embeddings using FinBERT or Llama 3.2
def extract_embeddings(texts, model_name='ProsusAI/finbert'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Normalize numerical data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Custom dataset class for LSTM
class StockDataset(Dataset):
    def __init__(self, financial_data, text_embeddings, target, seq_length=10):
        self.financial_data = financial_data
        self.text_embeddings = text_embeddings
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.financial_data) - self.seq_length

    def __getitem__(self, index):
        x_financial = self.financial_data[index:index+self.seq_length]
        x_text = self.text_embeddings[index]
        y = self.target[index+self.seq_length]
        return torch.tensor(x_financial, dtype=torch.float32), torch.tensor(x_text, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define LSTM model
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

# Train model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_financial, x_text, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x_financial, x_text)
            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_financial, x_text, y in test_loader:
            outputs = model(x_financial, x_text)
            predictions.append(outputs.numpy())
            actuals.append(y.numpy())
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")
    return mse, mae, r2

# Main function to run the model
def main():
    # Load data
    financial_data = load_financial_data('stock_data.csv', 'macro_data.csv')
    text_data = load_text_data('news_data.csv', 'social_data.csv')
    
    # Preprocess data
    normalized_data = normalize_data(financial_data.values)
    text_embeddings = extract_embeddings(text_data)
    
    # Split data
    split_idx = int(0.8 * len(normalized_data))
    train_data, test_data = normalized_data[:split_idx], normalized_data[split_idx:]
    train_text, test_text = text_embeddings[:split_idx], text_embeddings[split_idx:]
    target = financial_data['Close'].values
    train_target, test_target = target[:split_idx], target[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(train_data, train_text, train_target)
    test_dataset = StockDataset(test_data, test_text, test_target)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = StockPriceLSTM(input_dim=train_data.shape[1], text_dim=train_text.shape[1], hidden_dim=128, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
