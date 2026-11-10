# Cryptocurrency-future-price-prediction-
Cryptocurrency has become a popular form of digital asset, but its prices are highly  volatile and can change frequently within a short period of time. Therefore, collecting  cryptocurrency market data and analyzing historical price information can help in  understanding price trends and estimating future price movements.
import warnings
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

COINS = {"bitcoin": "BTC", "ethereum": "ETH", "cardano": "ADA", "litecoin": "LTC"}
WINDOW = 30
DAYS = 365

def fetch_coin(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    try:
        r = requests.get(url, params={"vs_currency": "usd", "days": DAYS,
                                      "interval": "daily"}, timeout=10)
        r.raise_for_status()
        d = r.json()
        df = pd.DataFrame({
            "date": pd.to_datetime([p[0] for p in d["prices"]], unit="ms"),
            "price": [p[1] for p in d["prices"]],
            "volume": [v[1] for v in d["total_volumes"]],
        })
        return coin_id, df, "live"
    except Exception:
        n = DAYS
        price = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, n)))
        df = pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=n),
            "price": price,
            "volume": np.random.uniform(1e6, 5e6, n),
        })
        return coin_id, df, "synthetic"

def collect_all():
    with ThreadPoolExecutor(max_workers=len(COINS)) as ex:
        return {cid: (df, src) for cid, df, src in ex.map(fetch_coin, COINS)}

def add_features(df):
    df = df.copy().drop_duplicates("date").sort_values("date")
    df["price"] = df["price"].interpolate()
    df["sma_7"] = df["price"].rolling(7).mean()
    df["ema_12"] = df["price"].ewm(span=12, adjust=False).mean()
    delta = df["price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    df["macd"] = (df["price"].ewm(span=12, adjust=False).mean()
                  - df["price"].ewm(span=26, adjust=False).mean())
    df["weekday"] = df["date"].dt.weekday
    return df.dropna().reset_index(drop=True)

def make_sequences(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

def build_rnn(kind, shape):
    from tensorflow.keras import layers, models
    Layer = layers.LSTM if kind == "LSTM" else layers.GRU
    m = models.Sequential([
        Layer(64, return_sequences=True, input_shape=shape),
        layers.Dropout(0.2),
        Layer(32),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def get_models():
    out = {}
    try:
        import tensorflow
        out["LSTM"] = "rnn"
        out["GRU"] = "rnn"
    except ImportError:
        print("[info] TensorFlow missing -> skipping LSTM / GRU")
    try:
        import xgboost
        out["XGBoost"] = "tree"
    except ImportError:
        out["GradientBoosting"] = "tree"
    return out

def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE%": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }

def run_coin(coin_id, df):
    df = add_features(df)
    feats = ["price", "volume", "sma_7", "ema_12", "rsi", "macd", "weekday"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[feats])
    X, y = make_sequences(data, WINDOW)
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    pmin, pmax = scaler.data_min_[0], scaler.data_max_[0]
    inv = lambda v: v * (pmax - pmin) + pmin
    y_true = inv(yte)

    results, preds = {}, {}
    for name, kind in get_models().items():
        if kind == "rnn":
            m = build_rnn(name, (WINDOW, len(feats)))
            m.fit(Xtr, ytr, epochs=30, batch_size=16,
                  validation_split=0.1, verbose=0)
            p = m.predict(Xte, verbose=0).ravel()
        else:
            if name == "XGBoost":
                from xgboost import XGBRegressor
                m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
            else:
                m = GradientBoostingRegressor(n_estimators=300,
                                              learning_rate=0.05, max_depth=4)
            m.fit(Xtr.reshape(len(Xtr), -1), ytr)
            p = m.predict(Xte.reshape(len(Xte), -1))
        preds[name] = inv(p)
        results[name] = metrics(y_true, preds[name])
    return results, y_true, preds

print("Collecting data with multithreading...")
data = collect_all()
for cid, (_, src) in data.items():
    print(f"  {COINS[cid]:4s} -> {src} data")

summary = []
for cid, (df, _) in data.items():
    res, y_true, preds = run_coin(cid, df)
    for model, mt in res.items():
        summary.append({"Coin": COINS[cid], "Model": model, **mt})
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, "k", label="Actual Price")
    for name, p in preds.items():
        plt.plot(p, label=f"{name} Predicted")
    plt.title(f"{COINS[cid]} Price Prediction")
    plt.xlabel("Time (Days)"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout()
    plt.show()

table = pd.DataFrame(summary)
print("\n=== Model Comparison (lower is better) ===")
print(table.round(4).to_string(index=False))
best = table.loc[table.groupby("Coin")["MAPE%"].idxmin()]
print("\n=== Best model per coin (by MAPE) ===")
print(best[["Coin", "Model", "MAPE%"]].round(3).to_string(index=False))
