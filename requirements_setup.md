# Stock Price Prediction System - Complete Setup Guide

## 📦 Project Structure

```
stock-prediction/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── train_model.py         # Model training script
│   ├── requirements.txt       # Python dependencies
│   └── models/                # Saved models directory
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx           # React application
│   │   └── index.js
│   └── public/
│
└── README.md
```

## 🐍 Backend Setup (Python)

### 1. Install Python Dependencies

Create `requirements.txt`:

```txt
flask==3.0.0
flask-cors==4.0.0
torch==2.1.0
scikit-learn==1.3.2
yfinance==0.2.32
numpy==1.24.3
pandas==2.1.3
joblib==1.3.2
```

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Your First Model

```bash
python train_model.py
```

This will:
- Download historical data for AAPL
- Train an LSTM model
- Save the model to `models/` directory

### 3. Start the Flask API

```bash
python app.py
```

API will run on `http://localhost:5000`

## ⚛️ Frontend Setup (React)

### 1. Create React App

```bash
npx create-react-app frontend
cd frontend
```

### 2. Install Dependencies

```bash
npm install recharts lucide-react axios
```

### 3. Update React App

Replace the content of `src/App.jsx` with the React code provided, but update it to call the real API:

```javascript
// Add this to the fetchStockData function:
const fetchStockData = async (symbol) => {
  setLoading(true);
  setError(null);
  
  try {
    // Fetch historical data
    const histResponse = await fetch(`http://localhost:5000/api/stock/${symbol}?period=3mo`);
    const histData = await histResponse.json();
    
    // Fetch predictions
    const predResponse = await fetch(`http://localhost:5000/api/predict/${symbol}?days=30`);
    const predData = await predResponse.json();
    
    setStockData(histData.data);
    setPredictions(predData.predictions);
  } catch (err) {
    setError('Failed to fetch stock data. Please try again.');
  } finally {
    setLoading(false);
  }
};
```

### 4. Start React App

```bash
npm start
```

App will run on `http://localhost:3000`

## 🚀 API Endpoints

### Get Historical Data
```
GET /api/stock/<ticker>?period=1y

Response:
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "data": [
    {
      "date": "2024-01-01",
      "price": 180.50,
      "open": 179.20,
      "high": 181.00,
      "low": 178.50,
      "volume": 50000000
    }
  ]
}
```

### Get Predictions
```
GET /api/predict/<ticker>?days=30

Response:
{
  "ticker": "AAPL",
  "predictions": [
    {
      "date": "2025-01-01",
      "price": 185.20
    }
  ],
  "current_price": 180.50,
  "predicted_price": 185.20
}
```

### Get Stock Info
```
GET /api/info/<ticker>

Response:
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "marketCap": 2800000000000
}
```

## 🎯 Using Free APIs

### Yahoo Finance (yfinance)
- **Free tier**: Unlimited requests
- **No API key required**
- **Data**: Historical prices, company info, real-time quotes
- **Limitations**: Rate limiting may apply, no SLA

### Alternative Free APIs:

1. **Alpha Vantage**
   - Free tier: 25 requests/day
   - Get API key: https://www.alphavantage.co/support/#api-key

2. **Twelve Data**
   - Free tier: 800 requests/day
   - Get API key: https://twelvedata.com/pricing

3. **Financial Modeling Prep**
   - Free tier: 250 requests/day
   - Get API key: https://site.financialmodelingprep.com/developer/docs

## 🧠 Model Details

### LSTM Architecture
- **Input**: 60 days of historical prices
- **Hidden layers**: 2 LSTM layers with 50 units each
- **Output**: Next day's price prediction
- **Training**: MSE loss with Adam optimizer

### Features
- Automatic model training on first request
- Model caching for faster predictions
- Supports any stock ticker from Yahoo Finance

## 🔧 Customization Options

### Train with Different Parameters

```python
# In train_model.py
model, scaler = train_model(
    ticker='TSLA',      # Stock symbol
    lookback=60,        # Days to look back
    epochs=100          # Training iterations
)
```

### Adjust Prediction Window

```python
# Predict 60 days instead of 30
predictions = predict_future(ticker='AAPL', days=60)
```

### Use Different Model Architectures

Replace LSTM with GRU or add more layers:

```python
class StockGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
```

## 📊 Pretrained Models

If you want to use pretrained models instead:

1. **FinBERT** (Financial sentiment analysis)
   - Hugging Face: `yiyanghkust/finbert-tone`
   
2. **Time Series Transformer**
   - Hugging Face: `huggingface/time-series-transformer-tourism-monthly`

Example with FinBERT:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
```

## ⚠️ Important Notes

1. **Not Financial Advice**: This is for educational purposes only
2. **Model Accuracy**: Stock predictions are inherently uncertain
3. **Data Quality**: Free APIs may have delays or limitations
4. **Rate Limits**: Implement caching to avoid hitting API limits

## 🐛 Troubleshooting

### Model Not Training
- Check internet connection (yfinance needs to download data)
- Verify ticker symbol is valid
- Ensure enough historical data exists

### CORS Issues
- Make sure Flask-CORS is installed
- Check that backend is running on port 5000

### Out of Memory
- Reduce lookback window (60 → 30 days)
- Reduce batch size during training
- Use CPU instead of GPU for small models

## 📚 Next Steps

1. Add more features (volume, moving averages, technical indicators)
2. Implement ensemble models (combine multiple predictions)
3. Add sentiment analysis from news articles
4. Deploy to cloud (Heroku, Railway, or Render)
5. Add user authentication and save predictions
6. Implement backtesting to evaluate model performance

## 🌐 Deployment

### Deploy Backend (Render/Railway)
```bash
# Add Procfile
web: python app.py
```

### Deploy Frontend (Vercel/Netlify)
```bash
npm run build
# Upload build/ directory
```

Good luck with your stock prediction system! 📈