from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch.nn as nn
import joblib


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
POLYGON_BASE_URL = "https://api.polygon.io"

_HISTORY_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_INFO_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_MODEL_CACHE: Dict[str, Tuple[float, Any, Any]] = {}

CACHE_TTL_SECONDS = 300
POPULAR_TICKERS = {
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "V",
    "WMT",
}
RATE_LIMIT_TOKENS = [
    "too many requests",
    "http error 429",
    "http 429",
    "expecting value",
    "jsondecodeerror",
    "temporary ban",
    "max retries exceeded",
    "timed out",
    "captcha",
    "standard api call frequency",
    "thank you for using alpha vantage",
]
NOT_FOUND_TOKENS = [
    "no price data found",
    "no timezone found",
    "possibly delisted",
    "ticker not found",
    "not found",
]


class UpstreamRateLimitError(Exception):
    """Raised when the upstream data provider responds with a rate-limit."""


def _normalize_message(message: str) -> str:
    return (message or "").strip()


def _message_indicates_rate_limit(message: str, ticker: str = "") -> bool:
    normalized = _normalize_message(message).lower()
    if not normalized:
        return False

    if any(token in normalized for token in RATE_LIMIT_TOKENS):
        return True

    if any(token in normalized for token in NOT_FOUND_TOKENS):
        # Treat delisting-style messages as rate-limit for very well-known tickers.
        return ticker.upper() in POPULAR_TICKERS

    return False


def _is_rate_limited_error(exc: Exception, ticker: str = "") -> bool:
    return _message_indicates_rate_limit(str(exc), ticker)


def _cache_is_fresh(timestamp: float) -> bool:
    return (time.time() - timestamp) <= CACHE_TTL_SECONDS

def _period_to_days(period: str) -> int:
    period_norm = (period or "").strip().lower()
    predefined = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 182,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "max": 3650,
    }
    if period_norm.endswith("d") and period_norm[:-1].isdigit():
        return max(int(period_norm[:-1]), 1)
    return predefined.get(period_norm, 365)


def _period_bounds(period: str) -> Tuple[datetime, datetime]:
    end = datetime.utcnow()
    days = _period_to_days(period)
    start = end - timedelta(days=days)
    return start, end


REQUEST_TIMEOUT_SECONDS = 10
HTTP_HEADERS = {"User-Agent": "stock-predictor/1.0"}


def _safe_period_key(period: str) -> str:
    raw = (period or "default").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in raw) or "default"


def _history_cache_path(ticker: str, period: str) -> Path:
    return DATA_DIR / f"history_{ticker.upper()}_{_safe_period_key(period)}.json"


def _prediction_cache_path(ticker: str, days: int) -> Path:
    return DATA_DIR / f"predictions_{ticker.upper()}_{days}d.json"


def _info_cache_path(ticker: str) -> Path:
    return DATA_DIR / f"info_{ticker.upper()}.json"


def _search_cache_path(query: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in query.lower().strip())
    safe = safe or "empty"
    return DATA_DIR / f"search_{safe}.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    timestamp = payload.get("_cached_at")
    if not isinstance(timestamp, (int, float)):
        return None
    if not _cache_is_fresh(timestamp):
        return None
    return payload


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    to_store = dict(payload)
    to_store["_cached_at"] = time.time()
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(to_store, handle)
    except OSError:
        return


def _frame_to_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in frame.iterrows():
        record = {
            "date": idx.strftime("%Y-%m-%d"),
            "Open": float(row.get("Open", 0.0)),
            "High": float(row.get("High", 0.0)),
            "Low": float(row.get("Low", 0.0)),
            "Close": float(row.get("Close", 0.0)),
            "Volume": float(row.get("Volume", 0.0)),
        }
        records.append(record)
    return records


def _records_to_frame(records: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not records:
        return None
    try:
        frame = pd.DataFrame(records)
    except ValueError:
        return None

    if frame.empty or "date" not in frame:
        return None

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame.dropna(subset=["date"], inplace=True)
    if frame.empty:
        return None

    frame.set_index("date", inplace=True)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.dropna(subset=["Close"], inplace=True)
    frame.sort_index(inplace=True)
    if frame.empty:
        return None
    return frame


def _load_cached_history(ticker: str, period: str) -> Optional[pd.DataFrame]:
    cached = _load_json(_history_cache_path(ticker, period))
    if not cached:
        return None
    records = cached.get("records")
    if not isinstance(records, list):
        return None
    return _records_to_frame(records)


def _save_cached_history(ticker: str, period: str, frame: pd.DataFrame) -> None:
    payload = {
        "ticker": ticker.upper(),
        "period": period,
        "records": _frame_to_records(frame),
    }
    _save_json(_history_cache_path(ticker, period), payload)


def _load_cached_predictions(ticker: str, days: int) -> Optional[Dict[str, Any]]:
    cached = _load_json(_prediction_cache_path(ticker, days))
    if not cached:
        return None
    cached.pop("_cached_at", None)
    return cached


def _save_cached_predictions(ticker: str, days: int, payload: Dict[str, Any]) -> None:
    _save_json(_prediction_cache_path(ticker, days), payload)


def _load_cached_info(ticker: str) -> Optional[Dict[str, Any]]:
    cached = _load_json(_info_cache_path(ticker))
    if not cached:
        return None
    cached.pop("_cached_at", None)
    return cached


def _save_cached_info(ticker: str, info: Dict[str, Any]) -> None:
    _save_json(_info_cache_path(ticker), info)


def _store_info_cache(ticker: str, info: Dict[str, Any]) -> None:
    _INFO_CACHE[ticker.upper()] = (time.time(), info)


def _load_cached_search(query: str) -> Optional[List[Dict[str, Any]]]:
    cached = _load_json(_search_cache_path(query))
    if not cached:
        return None
    results = cached.get("results")
    if not isinstance(results, list):
        return None
    return results


def _save_cached_search(query: str, results: List[Dict[str, Any]]) -> None:
    payload = {"query": query, "results": results}
    _save_json(_search_cache_path(query), payload)


class StockLSTM(nn.Module):
    """LSTM for stock prediction."""

    def __init__(self, input_size: int = 1, hidden_layer_size: int = 100, output_size: int = 1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def _build_company_stub(ticker: str, latest_price: Optional[float] = None) -> Dict[str, Any]:
    info = {
        "ticker": ticker.upper(),
        "name": ticker.upper(),
        "currency": "USD",
        "sector": "N/A",
        "industry": "N/A",
        "marketCap": 0.0,
    }
    if latest_price is not None:
        info["lastClose"] = float(latest_price)
    return info


def _fetch_history_polygon(ticker: str, period: str) -> Optional[pd.DataFrame]:
    if not POLYGON_API_KEY:
        return None

    start, end = _period_bounds(period)
    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=HTTP_HEADERS,
        )
    except requests.RequestException as exc:  # pragma: no cover - network guard
        if _is_rate_limited_error(exc, ticker):
            raise UpstreamRateLimitError(str(exc)) from exc
        return None

    if response.status_code == 429:
        raise UpstreamRateLimitError("Polygon rate limit reached")

    if response.status_code >= 500:
        return None

    data = response.json()
    status = data.get("status")
    if status == "ERROR":
        message = data.get("error") or str(data)
        if _message_indicates_rate_limit(message, ticker):
            raise UpstreamRateLimitError(message)
        return None

    if status not in {"OK", "DELAYED"}:
        return None

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None

    frame = pd.DataFrame(results)
    if frame.empty:
        return None

    rename_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    frame.rename(columns=rename_map, inplace=True)

    if "t" not in frame:
        return None
    frame["t"] = pd.to_datetime(frame["t"], unit="ms", errors="coerce")
    frame.dropna(subset=["t"], inplace=True)
    if frame.empty:
        return None
    frame.set_index("t", inplace=True)
    frame.index.name = None

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.dropna(subset=["Close"], inplace=True)
    frame = frame[[col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame]]
    frame.sort_index(inplace=True)
    return frame


def _search_tickers_polygon(query: str, limit: int = 10) -> List[Dict[str, str]]:
    if not POLYGON_API_KEY or not query:
        return []

    params = {
        "search": query.upper(),
        "active": "true",
        "limit": limit,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        response = requests.get(
            f"{POLYGON_BASE_URL}/v3/reference/tickers",
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=HTTP_HEADERS,
        )
    except requests.RequestException:
        return []

    if response.status_code == 429:
        raise UpstreamRateLimitError("Polygon rate limit reached")

    if response.status_code >= 500:
        return []

    data = response.json() or {}
    results = data.get("results")
    if not isinstance(results, list):
        return []

    suggestions: List[Dict[str, str]] = []
    for entry in results:
        ticker = entry.get("ticker")
        name = entry.get("name") or entry.get("description")
        if not ticker:
            continue
        suggestions.append({"ticker": ticker.upper(), "name": name or ticker.upper()})
        if len(suggestions) >= limit:
            break
    return suggestions


def _search_tickers_alpha_vantage(query: str, limit: int = 10) -> List[Dict[str, str]]:
    if not ALPHA_VANTAGE_API_KEY or not query:
        return []

    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=HTTP_HEADERS,
        )
    except requests.RequestException:
        return []

    if response.status_code == 429:
        raise UpstreamRateLimitError("Alpha Vantage rate limit reached")

    payload = response.json() or {}
    note = payload.get("Note") or payload.get("Information")
    if note:
        raise UpstreamRateLimitError(note)

    matches = payload.get("bestMatches")
    if not isinstance(matches, list):
        return []

    suggestions: List[Dict[str, str]] = []
    for entry in matches:
        symbol = entry.get("1. symbol")
        name = entry.get("2. name")
        if not symbol:
            continue
        suggestions.append({"ticker": symbol.upper(), "name": name or symbol.upper()})
        if len(suggestions) >= limit:
            break
    return suggestions


def _fetch_history(ticker: str, period: str) -> Optional[pd.DataFrame]:
    key = (ticker.upper(), period)
    cached = _HISTORY_CACHE.get(key)
    if cached and _cache_is_fresh(cached[0]):
        return cached[1]

    disk_frame = _load_cached_history(ticker, period)
    if disk_frame is not None and not disk_frame.empty:
        _HISTORY_CACHE[key] = (time.time(), disk_frame)
        return disk_frame

    frame = _fetch_history_polygon(ticker, period)
    if frame is not None and not frame.empty:
        _HISTORY_CACHE[key] = (time.time(), frame)
        _save_cached_history(ticker, period, frame)
    return frame


def _load_model(ticker: str) -> Tuple[Optional[StockLSTM], Optional[Any]]:
    """Load the model and scaler for a given ticker."""
    key = ticker.upper()
    cached = _MODEL_CACHE.get(key)
    if cached and _cache_is_fresh(cached[0]):
        return cached[1], cached[2]

    model_path = DATA_DIR / f"stock_model_{key}.pth"
    scaler_path = DATA_DIR / f"scaler_{key}.pkl"

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    try:
        model = StockLSTM()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load(scaler_path)
    except Exception:
        return None, None

    _MODEL_CACHE[key] = (time.time(), model, scaler)
    return model, scaler


@app.route("/api/stock/<ticker>", methods=["GET"])
def get_stock_data(ticker: str):
    """Return historical OHLCV data along with basic metadata."""
    try:
        period = request.args.get("period", "1y")
        try:
            history = _fetch_history(ticker, period)
        except UpstreamRateLimitError:
            return jsonify({"error": "Polygon rate-limited the request. Please wait a moment and try again."}), 429

        if history is None or history.empty:
            return jsonify({"error": "No historical data available for that ticker."}), 404

        data = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "price": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]),
            }
            for idx, row in history.iterrows()
        ]

        latest_price = data[-1]["price"] if data else None
        ticker_upper = ticker.upper()

        info: Dict[str, Any]
        cached_entry = _INFO_CACHE.get(ticker_upper)
        if cached_entry and _cache_is_fresh(cached_entry[0]):
            info = dict(cached_entry[1])
        else:
            disk_info = _load_cached_info(ticker)
            info = dict(disk_info) if isinstance(disk_info, dict) else _build_company_stub(ticker)

        if latest_price is not None:
            info["lastClose"] = latest_price

        _store_info_cache(ticker, info)
        _save_cached_info(ticker, info)

        return jsonify(
            {
                "ticker": info.get("ticker", ticker_upper),
                "name": info.get("name", ticker_upper),
                "currency": info.get("currency", "USD"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", 0.0),
                "data": data,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict/<ticker>", methods=["GET"])
def predict_stock(ticker: str):
    """Predict the next N days of closing prices for a given ticker."""
    try:
        days = int(request.args.get("days", 30))
        model, scaler = _load_model(ticker)

        if not model or not scaler:
            return jsonify({"error": "Model for this ticker is not available."}), 404

        try:
            stock_data = _fetch_history(ticker, "1y")
        except UpstreamRateLimitError:
            return jsonify({"error": "Polygon rate-limited the request. Please wait a moment and try again."}), 429

        if stock_data is None or stock_data.empty:
            return jsonify({"error": "Stock not found"}), 404

        closes = stock_data["Close"].values.astype(float)
        if closes.size < 60:
            return jsonify({"error": "Insufficient historical data to generate a forecast."}), 404

        # Prepare the data for prediction
        last_60_days = closes[-60:]
        scaled_data = scaler.transform(last_60_days.reshape(-1, 1))
        X_test = torch.FloatTensor(scaled_data).unsqueeze(0)

        # Make predictions
        predictions = []
        for _ in range(days):
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
                prediction = model(X_test)
                predictions.append(prediction.item())
                # Update X_test to include the new prediction
                new_sequence = torch.cat((X_test[0, 1:], prediction.unsqueeze(0).unsqueeze(0)), 0)
                X_test = new_sequence.unsqueeze(0)

        # Inverse transform the predictions
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        last_date = stock_data.index[-1]
        prediction_data = [
            {"date": (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"), "price": float(price)}
            for i, price in enumerate(predicted_prices.flatten())
        ]

        payload = {
            "ticker": ticker.upper(),
            "predictions": prediction_data,
            "current_price": float(closes[-1]),
            "predicted_price": float(predicted_prices[-1]),
        }

        _save_cached_predictions(ticker, days, payload)

        info = _load_cached_info(ticker) or _build_company_stub(ticker, payload["current_price"])
        info["lastClose"] = payload["current_price"]
        if prediction_data:
            info["nextPrediction"] = prediction_data[-1]["price"]
        _store_info_cache(ticker, info)
        _save_cached_info(ticker, info)

        return jsonify(payload)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/info/<ticker>", methods=["GET"])
def get_stock_info(ticker: str):
    """Return high-level company metadata for the given ticker."""
    try:
        ticker_upper = ticker.upper()
        cached_entry = _INFO_CACHE.get(ticker_upper)
        if cached_entry and _cache_is_fresh(cached_entry[0]):
            info = dict(cached_entry[1])
        else:
            disk_info = _load_cached_info(ticker)
            info = dict(disk_info) if isinstance(disk_info, dict) else _build_company_stub(ticker)
            _store_info_cache(ticker, info)
            _save_cached_info(ticker, info)

        if "lastClose" not in info:
            history = _load_cached_history(ticker, "1y")
            if history is not None and not history.empty:
                info["lastClose"] = float(history["Close"].iloc[-1])
                _store_info_cache(ticker, info)
                _save_cached_info(ticker, info)

        return jsonify(
            {
                "ticker": info.get("ticker", ticker_upper),
                "name": info.get("name", ticker_upper),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", 0.0),
                "currency": info.get("currency", "USD"),
                "lastClose": info.get("lastClose"),
                "nextPrediction": info.get("nextPrediction"),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search", methods=["GET"])
def search_stocks():
    """Return a small curated list of tickers that match the provided query."""
    query_raw = request.args.get("q", "")
    query = query_raw.strip()
    popular = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "WMT": "Walmart Inc.",
    }

    if not query:
        results = [{"ticker": ticker, "name": name} for ticker, name in popular.items()]
    else:
        cached_results = _load_cached_search(query)
        if cached_results is not None:
            results = cached_results
        else:
            alpha_rate_limited = False
            polygon_rate_limited = False
            try:
                results = _search_tickers_alpha_vantage(query)
            except UpstreamRateLimitError:
                alpha_rate_limited = True
                results = []

            if not results:
                try:
                    results = _search_tickers_polygon(query)
                except UpstreamRateLimitError:
                    polygon_rate_limited = True
                    results = []

            if results:
                _save_cached_search(query, results)
            elif alpha_rate_limited and polygon_rate_limited:
                return jsonify({"error": "All symbol providers are rate-limited. Please wait and try again."}), 429
            elif alpha_rate_limited:
                return jsonify({"error": "Alpha Vantage rate-limited the request. Please wait a moment and try again."}), 429
            elif polygon_rate_limited:
                return jsonify({"error": "Polygon rate-limited the request. Please wait a moment and try again."}), 429
            else:
                lowered = query.lower()
                results = [
                    {"ticker": ticker, "name": name}
                    for ticker, name in popular.items()
                    if lowered in ticker.lower() or lowered in name.lower()
                ]

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})


application = app
