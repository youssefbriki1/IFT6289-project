import torch
from torch import optim
from torch.utils.data import DataLoader
from data_loader import load_financial_data, load_text_data
from preprocessing import normalize_data, extract_embeddings
from dataset import StockDataset
from model import StockPriceLSTM
from train import train_model, evaluate_model


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
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()