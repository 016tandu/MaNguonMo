import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Data Preparation
def prepare_data(data, lookback=60):
    """Prepare time series data for LSTM"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Training Function
def train_model(ticker='AAPL', lookback=60, epochs=50):
    """
    Train LSTM model for stock price prediction
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        lookback: Number of days to look back for prediction
        epochs: Number of training epochs
    """
    print(f"Downloading data for {ticker}...")
    
    # Download stock data
    stock_data = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    data = stock_data['Close'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare sequences
    X, y = prepare_data(scaled_data, lookback)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # Initialize model
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        print(f'Test Loss: {test_loss.item():.6f}')
    
    # Save model and scaler
    torch.save(model.state_dict(), f'stock_model_{ticker}.pth')
    joblib.dump(scaler, f'scaler_{ticker}.pkl')
    
    print(f"Model and scaler saved successfully!")
    return model, scaler

# Prediction Function
def predict_future(ticker='AAPL', days=30, lookback=60):
    """
    Predict future stock prices
    
    Args:
        ticker: Stock symbol
        days: Number of days to predict into the future
        lookback: Number of days to look back
    """
    # Load model and scaler
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(f'stock_model_{ticker}.pth'))
    scaler = joblib.load(f'scaler_{ticker}.pkl')
    model.eval()
    
    # Get recent data
    stock_data = yf.download(ticker, period='1y')
    data = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    # Get last lookback days
    last_sequence = scaled_data[-lookback:]
    predictions = []
    
    # Predict future prices
    with torch.no_grad():
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            x = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
            pred = model(x)
            predictions.append(pred.item())
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred.cpu().numpy())
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

# Main execution
if __name__ == "__main__":
    # Example: Train model for Apple stock
    ticker = 'AAPL'
    print(f"Training stock prediction model for {ticker}...")
    model, scaler = train_model(ticker, lookback=60, epochs=50)
    
    # Predict next 30 days
    print(f"\nPredicting next 30 days...")
    future_prices = predict_future(ticker, days=30)
    print(f"Predicted prices (next 30 days):\n{future_prices}")
    
    print("\nTo use this model in your web app, you'll need:")
    print("1. Save the model and scaler files")
    print("2. Create a Flask/FastAPI backend to serve predictions")
    print("3. Use the React frontend to display the results")
