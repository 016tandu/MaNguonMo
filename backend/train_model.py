from __future__ import annotations

from datetime import datetime
from typing import Tuple
import argparse

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from app import StockLSTM, MODELS_DIR, _fetch_history, UpstreamRateLimitError  # reuse architecture and paths


def _prepare_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_model(ticker: str = "AAPL", lookback: int = 60, epochs: int = 50) -> Tuple[StockLSTM, MinMaxScaler]:
    """Train an LSTM model for a single ticker and persist the artefacts to disk."""
    MODELS_DIR.mkdir(exist_ok=True)

    try:
        stock_data = _fetch_history(ticker, "5y")
    except UpstreamRateLimitError as exc:
        raise RuntimeError(f"Rate limited while retrieving data for {ticker}: {exc}") from exc

    if stock_data is None or stock_data.empty:
        raise ValueError(f"No data received for ticker {ticker}")

    data = stock_data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = _prepare_sequences(scaled_data, lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)

    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_prediction = model(X_test_tensor)
        test_loss = criterion(test_prediction, y_test_tensor)
        print(f"Validation Loss: {test_loss.item():.6f}")

    model_path = MODELS_DIR / f"stock_model_{ticker}.pth"
    scaler_path = MODELS_DIR / f"scaler_{ticker}.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM stock model.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to train on (default: AAPL)")
    parser.add_argument("--lookback", type=int, default=60, help="Days of history per training sample (default: 60)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    args = parser.parse_args()

    train_model(ticker=args.ticker.upper(), lookback=args.lookback, epochs=args.epochs)
