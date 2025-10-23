import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Search, TrendingUp, Calendar, DollarSign } from 'lucide-react';

const StockPredictionApp = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);

  const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'];

  const fetchStockData = async (symbol) => {
    setLoading(true);
    setError(null);
    
    try {
      const historicalData = generateMockHistoricalData(symbol);
      const predictionData = generateMockPredictions(historicalData);
      
      setStockData(historicalData);
      setPredictions(predictionData);
    } catch (err) {
      setError('Failed to fetch stock data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const generateMockHistoricalData = (symbol) => {
    const data = [];
    const basePrice = { 'AAPL': 180, 'GOOGL': 140, 'MSFT': 380, 'AMZN': 170, 'TSLA': 250, 'META': 480, 'NVDA': 880, 'JPM': 190 }[symbol] || 100;
    const today = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const randomWalk = Math.sin(i / 10) * 10 + (Math.random() - 0.5) * 8;
      
      data.push({
        date: date.toISOString().split('T')[0],
        price: basePrice + randomWalk,
        type: 'historical'
      });
    }
    
    return data;
  };

  const generateMockPredictions = (historicalData) => {
    const predictions = [];
    const lastPrice = historicalData[historicalData.length - 1].price;
    const lastDate = new Date(historicalData[historicalData.length - 1].date);
    
    for (let i = 1; i <= 30; i++) {
      const date = new Date(lastDate);
      date.setDate(date.getDate() + i);
      const trend = Math.sin(i / 5) * 5 + (Math.random() - 0.3) * 3;
      
      predictions.push({
        date: date.toISOString().split('T')[0],
        price: lastPrice + trend + i * 0.2,
        type: 'prediction'
      });
    }
    
    return predictions;
  };

  const getChartData = () => {
    if (!stockData || !predictions) return [];
    
    const lastHistorical = stockData[stockData.length - 1];
    const allData = [
      ...stockData.slice(-60),
      { ...lastHistorical, predictedPrice: lastHistorical.price },
      ...predictions.map(p => ({
        date: p.date,
        predictedPrice: p.price,
        type: 'prediction'
      }))
    ];
    
    return allData;
  };

  useEffect(() => {
    fetchStockData(ticker);
  }, []);

  const handleSearch = () => {
    fetchStockData(ticker.toUpperCase());
  };

  const currentPrice = stockData ? stockData[stockData.length - 1].price : 0;
  const predictedPrice = predictions ? predictions[predictions.length - 1].price : 0;
  const priceChange = predictedPrice - currentPrice;
  const percentChange = (priceChange / currentPrice) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">Stock Price Predictor</h1>
          </div>
          
          <div className="flex gap-3 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Enter stock ticker (e.g., AAPL)"
                className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition-colors font-semibold"
            >
              {loading ? 'Loading...' : 'Analyze'}
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-gray-600 py-2">Popular:</span>
            {popularStocks.map(stock => (
              <button
                key={stock}
                onClick={() => { setTicker(stock); fetchStockData(stock); }}
                className="px-3 py-1 bg-gray-100 hover:bg-indigo-100 rounded-full text-sm font-medium transition-colors"
              >
                {stock}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {stockData && predictions && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center gap-3 mb-2">
                  <DollarSign className="w-6 h-6 text-green-600" />
                  <h3 className="text-gray-600 font-semibold">Current Price</h3>
                </div>
                <p className="text-3xl font-bold text-gray-800">${currentPrice.toFixed(2)}</p>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center gap-3 mb-2">
                  <Calendar className="w-6 h-6 text-blue-600" />
                  <h3 className="text-gray-600 font-semibold">30-Day Prediction</h3>
                </div>
                <p className="text-3xl font-bold text-gray-800">${predictedPrice.toFixed(2)}</p>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className={`w-6 h-6 ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`} />
                  <h3 className="text-gray-600 font-semibold">Expected Change</h3>
                </div>
                <p className={`text-3xl font-bold ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {priceChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%
                </p>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">
                Price History & Predictions for {ticker}
              </h2>
              
              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={getChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip 
                    formatter={(value) => [`$${value.toFixed(2)}`, '']}
                    labelFormatter={(date) => new Date(date).toLocaleDateString()}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#4f46e5" 
                    strokeWidth={2}
                    name="Historical Price"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="predictedPrice" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Predicted Price"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>

              <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
                <p className="text-sm text-yellow-800">
                  <strong>Note:</strong> This is a demonstration using simulated data. In production, connect to a Python backend 
                  running the PyTorch model with real yfinance data. Predictions are for educational purposes only and 
                  should not be used for actual trading decisions.
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default StockPredictionApp;