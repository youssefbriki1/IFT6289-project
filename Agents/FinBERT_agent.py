import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import argparse
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sequence_length = 10
# 4. Create PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, sequences, targets, company_ids):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.company_ids = torch.tensor(company_ids, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.company_ids[idx]
    
def __prepare_data():
    # Load stock price data
    df = pd.read_csv("AUGMENTED_entityMask_merged_news_stock(finbert).csv")

    # Define the features to use (add your features here)
    features = ['close', 'finbert_score', 'volume']
    target_feature = 'close'

    # 1. Separate data by company
    grouped = df.groupby('ticker')
    company_data = {company: group.sort_values('date').copy() for company, group in grouped}

    # Define sequence length (window size)
    sequence_length = 10

    # 2. Preprocessing and creating sequences
    processed_data = {}
    scalers = {}

    for company, data in company_data.items():
        scaler = MinMaxScaler()
        # Scale all features at once
        scaled_features = scaler.fit_transform(data[features])

        # Create scaled columns
        for i, feature in enumerate(features):
            data[f'{feature}_scaled'] = scaled_features[:, i]

        scalers[company] = scaler

        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            # Create sequence for all features
            feature_seqs = [data[f'{feature}_scaled'].iloc[i:i+sequence_length].values
                        for feature in features]
            sequence = np.stack(feature_seqs, axis=-1)  # shape (seq_len, num_features)
            target = data['close_scaled'].iloc[i + sequence_length]

            if np.isnan(sequence).any() or np.isnan(target):
                print("NaN found in sequence or target!")
            sequences.append(sequence)
            targets.append(target)

        processed_data[company] = (np.array(sequences), np.array(targets))

    # 3. Create company ID mapping
    unique_companies = list(company_data.keys())
    company_to_id = {company: i for i, company in enumerate(unique_companies)}
    num_companies = len(unique_companies)
    num_features = len(features)  # Number of input features


    # Prepare datasets
    train_size = 0.8
    val_size = 0.1

    train_sequences_all = []
    train_targets_all = []
    train_company_ids_all = []
    val_sequences_all = []
    val_targets_all = []
    val_company_ids_all = []
    test_sequences_all = []
    test_targets_all = []
    test_company_ids_all = []

    for company, (sequences, targets) in processed_data.items():
        n = len(sequences)
        train_idx = int(n * train_size)
        val_idx = int(n * (train_size + val_size))

        # Split data
        train_sequences = sequences[:train_idx]
        train_targets = targets[:train_idx]
        val_sequences = sequences[train_idx:val_idx]
        val_targets = targets[train_idx:val_idx]
        test_sequences = sequences[val_idx:]
        test_targets = targets[val_idx:]

        # Create company IDs
        company_id = company_to_id[company]
        train_company_ids = np.full(len(train_sequences), company_id)
        val_company_ids = np.full(len(val_sequences), company_id)
        test_company_ids = np.full(len(test_sequences), company_id)

        # Add to combined datasets
        train_sequences_all.extend(train_sequences)
        train_targets_all.extend(train_targets)
        train_company_ids_all.extend(train_company_ids)
        val_sequences_all.extend(val_sequences)
        val_targets_all.extend(val_targets)
        val_company_ids_all.extend(val_company_ids)
        test_sequences_all.extend(test_sequences)
        test_targets_all.extend(test_targets)
        test_company_ids_all.extend(test_company_ids)

    # Create datasets
    train_dataset = StockDataset(train_sequences_all, train_targets_all, train_company_ids_all)
    val_dataset = StockDataset(val_sequences_all, val_targets_all, val_company_ids_all)
    test_dataset = StockDataset(test_sequences_all, test_targets_all, test_company_ids_all)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_companies, num_features, scalers

# 5. Define the Model
class SharedStockModel(nn.Module):
    def __init__(self, num_companies, embedding_dim, sequence_length, hidden_units, num_features):
        super(SharedStockModel, self).__init__()
        self.embedding = nn.Embedding(num_companies, embedding_dim)
        self.lstm = nn.LSTM(input_size=num_features + embedding_dim,
                           hidden_size=hidden_units,
                           batch_first=True)
        self.linear = nn.Linear(hidden_units, 1)

    def forward(self, x, company_ids):
        embedded_companies = self.embedding(company_ids).unsqueeze(1)
        embedded_companies = embedded_companies.repeat(1, x.size(1), 1)
        combined_input = torch.cat((x, embedded_companies), dim=-1)
        out, _ = self.lstm(combined_input)
        predictions = self.linear(out[:, -1, :])
        return predictions


def train_model(save_path):
    train_loader, val_loader, test_loader, num_companies, num_features, scalers = __prepare_data()
    # 6. Model Training Setup
    embedding_dim = 10
    hidden_units = 50
    model = SharedStockModel(num_companies, embedding_dim, sequence_length, hidden_units, num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    # 7. Training Loop
    epochs = 20
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, targets, company_ids in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            company_ids = company_ids.to(device)

            optimizer.zero_grad()
            predictions = model(sequences, company_ids)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets, company_ids in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                company_ids = company_ids.to(device)
                predictions = model(sequences, company_ids)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - ORG Entity Masking')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), save_path)
    return model

def __moving_average_smoothing(preds, window_size=3):
    preds = np.array(preds).flatten()
    smoothed_preds = np.convolve(preds, np.ones(window_size)/window_size, mode='valid')
    # Pad to original length to avoid the ValueError
    pad_width = (window_size - 1, 0)  # Pad only at the beginning
    smoothed_preds = np.pad(smoothed_preds, pad_width=pad_width, mode='edge')
    return smoothed_preds

# 9. Make Predictions and Evaluate

def predict_model_ticker(model_path, company_name, target_date_str, full_data_csv, window_size=3, max_future_days=7):
    """
    Args:
        model: trained PyTorch model
        company_name: str, e.g., 'NVDA'
        target_date_str: str, e.g., '2025-04-30'
        full_data_csv: str, path to your full historical csv
        scalers: dict of scalers per company
        window_size: int, smoothing window
        max_future_days: int, maximum number of days into future allowed
    """

    # Load full historical data
    df = pd.read_csv(full_data_csv)
    df['date'] = pd.to_datetime(df['date'])
    target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date()
    latest_date = df['date'].max()

    # Safety checks
    if (target_date - latest_date).days > max_future_days:
        raise ValueError(f"Can only predict up to {max_future_days} days later than the last updated date of model: {latest_date}.")

    # Filter for the company
    company_df = df[df['ticker'] == company_name].sort_values('date')

    if company_df.empty:
        raise ValueError(f"No data found for {company_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        # First, you must instantiate the model architecture
        train_loader, val_loader, test_loader, num_companies, num_features, scalers = __prepare_data()
        # 6. Model Training Setup
        embedding_dim = 10
        hidden_units = 50
        model = SharedStockModel(
            num_companies=num_companies, 
            embedding_dim=embedding_dim, 
            sequence_length=sequence_length, 
            hidden_units=hidden_units,
            num_features=num_features)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"No model found at {model_path}. Training a new model...")
        model = train_model(save_path=model_path)
    
    # Get last sequence_length rows
    feature_cols = model.feature_columns  # make sure model has this attribute
    sequence_length = model.sequence_length
    device = next(model.parameters()).device

    # Handle edge cases if not enough data
    if len(company_df) < sequence_length:
        raise ValueError(f"Not enough data to predict (need {sequence_length} days)")

    sequence = company_df[feature_cols].values[-sequence_length:]

    # Convert to tensor
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    company_id_tensor = torch.tensor([0], dtype=torch.long).to(device)  # If your model expects company_id, else skip

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(sequence_tensor, company_id_tensor).cpu().numpy().flatten()

    # Smoothing (if needed)
    pred_smoothed = __moving_average_smoothing(pred, window_size=window_size)[-1]

    # Inverse scale
    dummy_array = np.zeros((1, len(feature_cols)))
    dummy_array[:, 0] = pred_smoothed
    pred_final = scalers[company_name].inverse_transform(dummy_array)[:, 0][0]

    print(f"Prediction for {company_name} on {target_date_str}: {pred_final:.2f}")

    # Plot
    # Historical true prices
    close_prices = company_df['close'].values[-sequence_length:]
    dates = company_df['date'].values[-sequence_length:]

    plt.figure(figsize=(12,6))
    plt.plot(dates, close_prices, label='Historical Close Prices', marker='o')
    plt.scatter(target_date, pred_final, color='red', label=f'Predicted ({target_date_str})', s=100, marker='*')
    plt.title(f"{company_name} Stock Price Prediction ({target_date_str})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    return pred_final

def main():
    parser = argparse.ArgumentParser(description="Train or Predict with FinBERT+LSTM model")
    parser.add_argument('command', choices=['train', 'predict'], help='Command: train or predict')
    parser.add_argument('model_path', type=str, help='Path to save/load the model')
    parser.add_argument('ticker', nargs='?', default=None, help='Ticker symbol (only for predict)')
    parser.add_argument('target_date', nargs='?', default=None, help='Date to predict (only for predict)')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.model_path)
    elif args.command == 'predict':
        if args.ticker is None:
            print("Ticker symbol required for prediction.")
            return
        if args.target_date is None:
            print("Target prediction date required.")
            return
        predict_model_ticker(args.model_path, args.ticker)

if __name__ == "__main__":
    main()