import torch
import numpy as np
from torch import optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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