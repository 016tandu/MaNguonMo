import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import axios from "axios";
import {
  TrendingUp,
  Search,
  Calendar,
  DollarSign,
  Loader2,
  AlertCircle,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

import "./App.css";

type HistoricalPoint = {
  date: string;
  price: number;
  open: number;
  high: number;
  low: number;
  volume: number;
};

type PredictionPoint = {
  date: string;
  price: number;
};

type CompanyInfo = {
  ticker: string;
  name: string;
  sector?: string;
  industry?: string;
  marketCap?: number;
  currency?: string;
};

type SearchResult = {
  ticker: string;
  name: string;
};

const LOCAL_API_BASE_URL = "http://127.0.0.1:5000";
const PRIMARY_REMOTE_API_BASE_URL = "https://manguonmo-cksy.onrender.com";
const LEGACY_REMOTE_API_BASE_URL = "https://stock-predictor-server.onrender.com";

function sanitizeBaseUrl(url?: string | null): string | null {
  if (typeof url !== "string") {
    return null;
  }
  const trimmed = url.trim();
  if (!trimmed) {
    return null;
  }
  return trimmed.replace(/\/+$/, "");
}

function buildApiBaseUrlCandidates(): string[] {
  const candidates: string[] = [];

  const envUrl = sanitizeBaseUrl(import.meta.env.VITE_API_BASE_URL);
  if (envUrl) {
    candidates.push(envUrl);
  }

  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      candidates.push(LOCAL_API_BASE_URL);
    }
  }

  candidates.push(PRIMARY_REMOTE_API_BASE_URL, LEGACY_REMOTE_API_BASE_URL);

  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    if (!candidate || seen.has(candidate)) {
      return false;
    }
    seen.add(candidate);
    return true;
  });
}

async function pingApiBase(url: string, externalSignal?: AbortSignal, timeoutMs = 1500): Promise<boolean> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  const handleAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", handleAbort);

  try {
    if (externalSignal?.aborted) {
      return false;
    }
    const response = await fetch(`${url}/health`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
    externalSignal?.removeEventListener("abort", handleAbort);
  }
}

const popularTickers = [
  "AAPL",
  "GOOGL",
  "MSFT",
  "AMZN",
  "TSLA",
  "META",
  "NVDA",
  "JPM",
];

function formatCurrency(value: number, currency = "USD") {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function App() {
  const apiCandidates = useMemo(() => buildApiBaseUrlCandidates(), []);
  const [apiBaseUrl, setApiBaseUrl] = useState(() => apiCandidates[0] ?? "");
  const [ticker, setTicker] = useState("AAPL");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalPoint[]>([]);
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null);
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const suggestionsCache = useRef<Record<string, SearchResult[]>>({});

  useEffect(() => {
    let isMounted = true;
    const abortController = new AbortController();

    const selectApiBase = async () => {
      for (const candidate of apiCandidates) {
        if (!candidate) {
          continue;
        }
        const reachable = await pingApiBase(candidate, abortController.signal);
        if (!isMounted) {
          return;
        }
        if (reachable) {
          setApiBaseUrl(candidate);
          return;
        }
      }
    };

    selectApiBase();

    return () => {
      isMounted = false;
      abortController.abort();
    };
  }, [apiCandidates]);

  const fetchStockData = async (symbol: string) => {
    const trimmedSymbol = symbol.trim().toUpperCase();
    if (!trimmedSymbol || !apiBaseUrl) {
      return;
    }

    setTicker(trimmedSymbol);
    setLoading(true);
    setError(null);

    try {
      const [historyRes, predictionsRes, infoRes] = await Promise.all([
        axios.get(`${apiBaseUrl}/api/stock/${trimmedSymbol}`, {
          params: { period: "1y" },
        }),
        axios.get(`${apiBaseUrl}/api/predict/${trimmedSymbol}`, {
          params: { days: 30 },
        }),
        axios.get(`${apiBaseUrl}/api/info/${trimmedSymbol}`),
      ]);

      setHistoricalData(historyRes.data.data ?? []);
      setPredictions(predictionsRes.data.predictions ?? []);
      setCompanyInfo(infoRes.data);
    } catch (err) {
      console.error(err);

      let message =
        "We were unable to retrieve data for that ticker. Please verify the symbol and ensure the backend is running.";
      let shouldClearData = true;

      if (axios.isAxiosError(err)) {
        const serverMessage = err.response?.data?.error;
        if (typeof serverMessage === "string" && serverMessage.trim().length > 0) {
          message = serverMessage;
        } else if (err.response?.status === 404) {
          message = "We could not find data for that ticker. Please double-check the symbol.";
        }

        if (err.response?.status === 429) {
          shouldClearData = false;
        }
      }

      if (shouldClearData) {
        setHistoricalData([]);
        setPredictions([]);
        setCompanyInfo(null);
      }

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!apiBaseUrl) {
      return;
    }
    fetchStockData(ticker);
  }, [apiBaseUrl]);

  useEffect(() => {
    const query = ticker.trim();
    if (!showSuggestions || !query || query.length < 1) {
      setSuggestions([]);
      setSuggestionsLoading(false);
      return;
    }

    if (suggestionsCache.current[query]) {
      setSuggestions(suggestionsCache.current[query]);
      return;
    }

    const controller = new AbortController();
    setSuggestionsLoading(true);

    const timeoutId = window.setTimeout(async () => {
      try {
        if (!apiBaseUrl) {
          setSuggestionsLoading(false);
          return;
        }
        const response = await axios.get(`${apiBaseUrl}/api/search`, {
          params: { q: query },
          signal: controller.signal,
        });
        const results = response.data.results ?? [];
        suggestionsCache.current[query] = results;
        setSuggestions(results);
      } catch (err) {
        if (!axios.isCancel(err)) {
          console.error("Search suggestion error", err);
          setSuggestions([]);
        }
      } finally {
        setSuggestionsLoading(false);
      }
    }, 200);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [ticker, showSuggestions, apiBaseUrl]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    fetchStockData(ticker);
  };

  const handleSelectSuggestion = (symbol: string) => {
    setTicker(symbol);
    setShowSuggestions(false);
  };

  const latestHistorical = historicalData[historicalData.length - 1];
  const latestPrediction = predictions[predictions.length - 1];
  const currentPrice = latestHistorical?.price ?? 0;
  const predictedPrice = latestPrediction?.price ?? 0;
  const priceChange = predictedPrice - currentPrice;
  const percentChange = currentPrice
    ? (priceChange / currentPrice) * 100
    : 0;

  const chartData = useMemo(() => {
    if (!historicalData.length) return [];

    const recentHistory = historicalData.slice(-60).map((item) => ({
      date: item.date,
      historical: item.price,
      predicted: null as number | null,
    }));

    const forecastData = predictions.map((item, index) => ({
      date: item.date,
      historical: index === 0
        ? recentHistory[recentHistory.length - 1]?.historical ?? null
        : null,
      predicted: item.price,
    }));

    return [...recentHistory, ...forecastData];
  }, [historicalData, predictions]);

  return (
    <div className="app">
      <header className="header">
        <div className="title-block">
          <TrendingUp className="title-icon" />
          <div>
            <h1>Stock Price Predictor</h1>
            <p>Forecast market trends with an LSTM model powered backend.</p>
          </div>
        </div>

        <form className="search-bar" onSubmit={handleSubmit}>
          <div className="search-input">
            <Search className="search-icon" />
            <input
              value={ticker}
              onChange={(event) => setTicker(event.target.value)}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => {
                window.setTimeout(() => setShowSuggestions(false), 150);
              }}
              placeholder="Enter ticker symbol (e.g. AAPL)"
            />
            {showSuggestions && ticker.trim() && (
              <div className="search-suggestions">
                {suggestionsLoading ? (
                  <div className="suggestion-item muted">Searching…</div>
                ) : suggestions.length > 0 ? (
                  suggestions.map((item) => (
                    <button
                      key={item.ticker}
                      type="button"
                      className="suggestion-item"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => handleSelectSuggestion(item.ticker)}
                    >
                      <span className="suggestion-symbol">{item.ticker}</span>
                      <span className="suggestion-name">{item.name}</span>
                    </button>
                  ))
                ) : (
                  <div className="suggestion-item muted">No matches found</div>
                )}
              </div>
            )}
          </div>
          <button type="submit" disabled={loading}>
            {loading ? <Loader2 className="spinner" /> : "Analyze"}
          </button>
        </form>

        <div className="popular-chips">
          <span>Popular:</span>
          <div className="chip-list">
            {popularTickers.map((symbol) => (
              <button
                key={symbol}
                type="button"
                onClick={() => handleSelectSuggestion(symbol)}
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle className="error-icon" />
          <p>{error}</p>
        </div>
      )}

      {!error && (
        <>
          <section className="stats-grid">
            <article className="stat-card">
              <div className="stat-header">
                <DollarSign />
                <h3>Current Price</h3>
              </div>
              <p className="stat-value">
                {latestHistorical
                  ? formatCurrency(
                      currentPrice,
                      companyInfo?.currency ?? "USD"
                    )
                  : "--"}
              </p>
              {latestHistorical && (
                <span className="stat-subtext">
                  Updated {new Date(latestHistorical.date).toDateString()}
                </span>
              )}
            </article>

            <article className="stat-card">
              <div className="stat-header">
                <Calendar />
                <h3>30-Day Outlook</h3>
              </div>
              <p className="stat-value">
                {latestPrediction
                  ? formatCurrency(
                      predictedPrice,
                      companyInfo?.currency ?? "USD"
                    )
                  : "--"}
              </p>
              {latestPrediction && (
                <span className="stat-subtext">
                  Target {new Date(latestPrediction.date).toDateString()}
                </span>
              )}
            </article>

            <article className="stat-card">
              <div className="stat-header">
                <TrendingUp />
                <h3>Expected Change</h3>
              </div>
              <p
                className={`stat-value ${
                  priceChange >= 0 ? "positive" : "negative"
                }`}
              >
                {priceChange >= 0 ? "+" : ""}
                {percentChange.toFixed(2)}%
              </p>
              <span className="stat-subtext">
                {priceChange >= 0 ? "Bullish" : "Bearish"} signal
              </span>
            </article>

            <article className="stat-card info-card">
              <h3>{companyInfo?.name ?? ticker}</h3>
              <dl>
                <div>
                  <dt>Ticker</dt>
                  <dd>{companyInfo?.ticker ?? ticker}</dd>
                </div>
                <div>
                  <dt>Sector</dt>
                  <dd>{companyInfo?.sector ?? "N/A"}</dd>
                </div>
                <div>
                  <dt>Industry</dt>
                  <dd>{companyInfo?.industry ?? "N/A"}</dd>
                </div>
                <div>
                  <dt>Market Cap</dt>
                  <dd>
                    {companyInfo?.marketCap
                      ? `${formatNumber(companyInfo.marketCap)} ${
                          companyInfo.currency ?? "USD"
                        }`
                      : "N/A"}
                  </dd>
                </div>
              </dl>
            </article>
          </section>

          <section className="chart-section">
            <header>
              <h2>Price History & Forecast</h2>
              <p>
                Historical closing prices are shown in indigo, with model
                projections in amber.
              </p>
            </header>

            <div className="chart-wrapper">
              {loading ? (
                <div className="chart-loading">
                  <Loader2 className="spinner large" />
                  <span>Fetching latest data…</span>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={420}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) =>
                        new Date(value).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        })
                      }
                      tick={{ fontSize: 12 }}
                      minTickGap={20}
                    />
                    <YAxis
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip
                      formatter={(
                        value: number | string | Array<number | string>
                      ) =>
                        typeof value === "number"
                          ? formatCurrency(value, companyInfo?.currency ?? "USD")
                          : value
                      }
                      labelFormatter={(value) =>
                        new Date(value).toLocaleDateString()
                      }
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="historical"
                      name="Historical"
                      stroke="#4f46e5"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="Predicted"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      dot={false}
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default App;
