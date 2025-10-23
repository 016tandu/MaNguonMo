# Stock Prediction App

End-to-end stock price prediction project that pairs a Flask backend (PyTorch LSTM) with a Vite + React web frontend.

## Project Layout

```
.
├── backend/              # Flask API, model training utilities, Python virtual env helper
├── frontend/             # Vite React client with Recharts visualisations
├── flask_backend_api.txt # Original reference implementation (kept for context)
├── stock_prediction_model.py
└── stock_prediction_webapp.tsx
```

The top-level Python/React snippets from the original prompt remain for reference but the production-ready code now lives inside `backend/` and `frontend/`.

---

## Backend (Flask + PyTorch)

All Python dependencies should be installed inside the local virtual environment.

```bash
cd backend

# Create a Python 3.11 virtual environment (scikit-learn wheels are not yet available for Python 3.13)
python3.11 -m venv .venv

# Activate the environment (POSIX)
source .venv/bin/activate
# On Windows (PowerShell) use: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt  # installs PyTorch 2.5.1 (CPU build) + dependencies

# Configure API keys (copy template once)
cp .env.example .env
# then edit .env with your Finnhub & Alpha Vantage keys

# (Optional) pre-train a model so the first request is fast
python train_model.py --ticker AAPL

# Tip: The first few requests trigger provider downloads. Kick these off one at a time (or pre-train
# popular tickers) to avoid hitting upstream rate limits.

# Start the API server
python app.py
```

### Key Endpoints

- `GET /api/stock/<ticker>?period=1y` – historical OHLCV data
- `GET /api/predict/<ticker>?days=30` – forward price forecasts
- `GET /api/info/<ticker>` – company metadata
- `GET /api/search?q=aap` – quick search against a curated list

Models are stored under `backend/models/` and cached in-memory for repeat calls. If you use Python 3.13 you will hit compilation errors because scikit-learn does not publish wheels for that version yet—stick to Python 3.11 (or 3.10) for now. The requirements file also points pip at the PyTorch **CPU** wheel index so installs avoid pulling the massive CUDA toolchain, and the API now caches upstream responses, rotating across Finnhub → Alpha Vantage while surfacing graceful rate-limit messages when providers throttle requests.

---

## Frontend (Vite + React)

```bash
cd frontend

# Provide API base URL (optional, defaults to http://localhost:5000)
cp .env.example .env

npm install
npm run dev
```

Open the dev server URL (typically `http://localhost:5173`) once the Flask API is running. The dashboard will fetch live data, render historical prices, and overlay the LSTM forecasts.

Run `npm run build` to produce a production bundle (output in `frontend/dist/`).

---

## Verification Checklist

1. `python app.py` (inside the activated virtual environment) starts the backend on port 5000.
2. `npm run dev` (with the frontend `.env` configured) serves the React dashboard.
3. Navigating to the frontend shows current price, 30-day projection, and charted history/predictions for the chosen ticker.

---

## Notes

- The LSTM trains automatically the first time a new ticker is requested; pretraining with `train_model.py` can shorten this delay.
- Forecasts are illustrative. They do **not** constitute financial advice.
- For production deployment, consider externalising the model store (S3, database) and adding authentication/caching around the prediction endpoints.

codex resume 019a055a-48f8-77a2-896b-929c37960196

