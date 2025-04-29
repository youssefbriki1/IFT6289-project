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
from pandas.tseries.offsets import BDay
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
    
def __prepare_data(datafile,sentiment_analysis_model):
    # Load stock price data
    df = pd.read_csv(datafile)

    # Define the features to use 
    #features = ['close', 'finbert_score', 'volume']
    if sentiment_analysis_model == 'news-only-finbert':
        features = ['close', 'finbert_score', 'volume']
    elif sentiment_analysis_model == 'news-socialmedia-finbert':
        features = ['close', 'news_score_finbert','social_score_finbert', 'volume']
    
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
    
    train_sequences_all = []
    train_targets_all = []
    train_company_ids_all = []

    for company, (sequences, targets) in processed_data.items():
        
        # Create company IDs
        company_id = company_to_id[company]
        train_company_ids = np.full(len(sequences), company_id)

        # Add to combined datasets
        train_sequences_all.extend(sequences)
        train_targets_all.extend(targets)
        train_company_ids_all.extend(train_company_ids)
    
    # Create datasets
    train_dataset = StockDataset(train_sequences_all, train_targets_all, train_company_ids_all)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, num_companies, num_features, scalers, unique_companies

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


def train_model(save_path, datafile,sentiment_analysis_model):
    train_loader, num_companies, num_features, scalers, unique_companies = __prepare_data(datafile,sentiment_analysis_model)
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


    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - ORG Entity Masking')
    plt.legend()
    plt.savefig("TrainingLoss.png")
    plt.show()

    torch.save(model.state_dict(), save_path)
    return model


def moving_average_smoothing(preds, window_size=3):
    preds = np.array(preds).flatten()
    smoothed_preds = np.convolve(preds, np.ones(window_size)/window_size, mode='valid')
    # Pad to original length to avoid the ValueError
    pad_width = (window_size - 1, 0)  # Pad only at the beginning
    smoothed_preds = np.pad(smoothed_preds, pad_width=pad_width, mode='edge')
    return smoothed_preds

def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    true_direction = np.diff(y_true)
    pred_direction = np.diff(y_pred)
    correct = ((true_direction * pred_direction) > 0).sum()
    total = len(true_direction)
    return correct / total if total > 0 else 0

# 10. Enhanced Visualization with Future Prediction
def predict_model_ticker(company_name, sentiment_analysis_model):

    ##### Add additional possible models for sentiment analysis here ####
    
    if sentiment_analysis_model == 'news-only-finbert':
        model_path = "news_only_finbert.pt"
        datafile = "AUGMENTED_entityMask_merged_news_stock(finbert).csv"
    elif sentiment_analysis_model == 'news-socialmedia-finbert':
        model_path = "news_socialmedia_finbert.pt"
        datafile = "news_socialmedia_merged_data(finbert).csv"
    ###############


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, num_companies, num_features, scalers, unique_companies = __prepare_data(datafile,sentiment_analysis_model)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        # First, you must instantiate the model architecture
        
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
        model = train_model(save_path=model_path, datafile=datafile, sentiment_analysis_model=sentiment_analysis_model)
        model.eval()

    all_predictions = []
    all_targets = []
    all_companies = []
    
    with torch.no_grad():
        for sequences, targets, company_ids in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            company_ids = company_ids.to(device)

            predictions = model(sequences, company_ids)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_companies.extend([unique_companies[i] for i in company_ids.cpu().numpy()])

    predictions = moving_average_smoothing(all_predictions)
    
    targets = moving_average_smoothing(all_targets)

   
    if sentiment_analysis_model == 'news-only-finbert':
        features = ['close', 'finbert_score', 'volume']
    elif sentiment_analysis_model == 'news-socialmedia-finbert':        
        features = ['close', 'news_score_finbert','social_score_finbert', 'volume']
    target_feature = 'close'
    

    # Get indices for this company
    company_indices = [i for i, company in enumerate(all_companies) if company == company_name]

    if not company_indices:
        print(f"No data found for company: {company_name}")
        return

    # Prepare data for inverse transform
    dummy_array = np.zeros((len(company_indices), len(features)))

    # Get predictions and targets for this company
    y_pred = np.array([predictions[i] for i in company_indices])
    y_true = np.array([targets[i] for i in company_indices])

    # Inverse transform
    dummy_array[:, 0] = y_pred
    preds = scalers[company_name].inverse_transform(dummy_array)[:, 0]

    dummy_array[:, 0] = y_true
    trues = scalers[company_name].inverse_transform(dummy_array)[:, 0]

    # Calculate metrics
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    direction_acc = directional_accuracy(trues, preds)

    df = pd.read_csv(datafile)
    df['date'] = pd.to_datetime(df['date'])

    # Filter only the rows for the current company
    df_company = df[df['ticker'] == company_name]

    if df_company.empty:
        print(f"No historical data found for company {company_name} in the CSV.")
        return

    # Find the last date available
    last_date = df_company['date'].max()
    next_business_day = last_date + BDay(1)

    # Create plot
    plt.figure(figsize=(15, 6))

    # Plot true and predicted values
    plt.plot(trues, 'b-', label='True Values', linewidth=2)
    plt.plot(preds, 'r--', label='Predictions', linewidth=2)

    # Add future prediction point
    if len(preds) > 0:
        future_point = len(trues)
        plt.scatter(future_point, preds[-1], c='g', s=200,
                   label=f'Next Day {next_business_day} Prediction: {preds[-1]:.2f}',
                   marker='*', edgecolors='k')
        plt.axvline(x=len(trues)-0.5, color='gray', linestyle='--')

    # Add metrics to plot
    metrics_text = (f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}\n"
                   f"Direction Acc: {direction_acc:.2%}")
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f'{company_name} Stock Price Prediction - ORG Entity Masking', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{company_name}_Predictions.png")
    plt.show()

    print(f"Predicted next-day {next_business_day} close price for {company_name}: {preds[-1]:.2f}")


############################################################################
###### LET'S PREDICT!! #####################################################
def main():
    parser = argparse.ArgumentParser(description="Train or Predict with FinBERT+LSTM model")
    parser.add_argument('command', choices=['predict'], help='Command: train or predict')
    parser.add_argument('ticker', nargs='?', default=None, help='Ticker symbol (only for predict)')
    ### this is the argument where we specify which sentiment analysis model/dataset csv file we should use
    ### FOR NOW: ONLY news-only-finbert WORKS!!!! ADD news+socialmedia finbert and Deepseek predicted sentiment CSV when available!!!!
    parser.add_argument('sentiment_analysis_model', choices=['news-only-finbert', 'news-socialmedia-finbert'], help='Choose the sentiment analysis model for prediction')

    args = parser.parse_args()

    if args.command == 'predict':
        if args.ticker is None:
            print("Ticker symbol required for prediction.")
            return
        if args.sentiment_analysis_model is None:
            print("Using default sentiment analysis model: news-only-finbert")
            args.sentiment_analysis_model = 'news-only-finbert'
       

    predict_model_ticker(args.ticker, args.sentiment_analysis_model)

if __name__ == "__main__":
    main()