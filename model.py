"""
model.py - Unified LSTM & GRU Stock Price Prediction
Saham: BBRI, BMRI, BBTN, BBNI
Target: Harga Penutupan Disesuaikan (Adj Close)
Metrik: MAE, MSE, MAPE
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
STOCKS       = ["BBRI.JK", "BMRI.JK", "BBTN.JK", "BBNI.JK"]
START_DATE   = "2015-01-01"
END_DATE     = "2025-12-31"
TIMESTEPS    = 30          # lookback window
TEST_SIZE    = 0.2
EPOCHS       = 50
BATCH_SIZE   = 32
FUTURE_DAYS  = 3           # prediksi ke depan
LEARNING_RATE = 0.0001


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_data(ticker: str) -> pd.DataFrame:
    """Download data historis dari Yahoo Finance.
    Kompatibel dengan yfinance lama (Adj Close) dan baru (MultiIndex / Close).
    """
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)

    # Flatten MultiIndex columns jika ada (yfinance >= 0.2.38)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Pilih kolom harga: utamakan 'Adj Close', fallback ke 'Close'
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
    elif "Close" in df.columns:
        df = df[["Close"]].copy()
    else:
        raise ValueError(f"Kolom harga tidak ditemukan untuk {ticker}. "
                         f"Kolom tersedia: {df.columns.tolist()}")

    df.columns = ["ha"]
    df = df.dropna()
    df.index.name = "Tanggal"
    return df


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    """Buat sequence untuk model RNN."""
    x_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        x_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(x_seq), np.array(y_seq)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_lstm(input_shape):
    """Bangun model LSTM."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_gru(input_shape):
    """Bangun model GRU."""
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_and_evaluate(ticker: str) -> dict:
    """
    Pipeline lengkap untuk satu saham:
    - Load data
    - Scaling
    - Split train/test
    - Latih LSTM & GRU
    - Evaluasi MAE, MSE, MAPE
    - Prediksi 3 hari ke depan
    """
    print(f"\n{'='*55}")
    print(f"  MEMPROSES: {ticker}")
    print(f"{'='*55}")

    # 1. Load & Preprocess
    df = load_data(ticker)
    values = df["ha"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(values)
    y_scaled = scaler_y.fit_transform(values)

    # 2. Train/Test Split (tanpa shuffle – time series)
    split = int(len(X_scaled) * (1 - TEST_SIZE))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    # 3. Sequence Creation
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIMESTEPS)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  TIMESTEPS)

    input_shape = (TIMESTEPS, 1)
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    results = {}

    for model_name, build_fn in [("LSTM", build_lstm), ("GRU", build_gru)]:
        print(f"\n  ▶ Melatih {model_name}...")
        model = build_fn(input_shape)
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        # Prediksi test
        pred_scaled = model.predict(X_test_seq, verbose=0)
        pred_actual = scaler_y.inverse_transform(pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test_seq)

        # Metrik
        mae_val  = mean_absolute_error(y_test_actual, pred_actual)
        mse_val  = mean_squared_error(y_test_actual, pred_actual)
        mape_val = mape(y_test_actual, pred_actual)

        print(f"    MAE : {mae_val:.4f}")
        print(f"    MSE : {mse_val:.4f}")
        print(f"    MAPE: {mape_val:.4f}%")

        # ── Prediksi 3 hari ke depan ──
        last_sequence = X_scaled[-TIMESTEPS:].reshape(1, TIMESTEPS, 1)
        future_preds = []
        current_seq = last_sequence.copy()

        for _ in range(FUTURE_DAYS):
            next_pred = model.predict(current_seq, verbose=0)
            future_preds.append(next_pred[0, 0])
            # Geser sequence
            current_seq = np.append(
                current_seq[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )

        future_preds_actual = scaler_y.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        results[model_name] = {
            "predictions"     : pred_actual.flatten(),
            "actuals"         : y_test_actual.flatten(),
            "mae"             : mae_val,
            "mse"             : mse_val,
            "mape"            : mape_val,
            "future_3days"    : future_preds_actual,
            "history"         : history,
            "test_dates"      : df.index[split + TIMESTEPS :],
            "all_dates"       : df.index,
            "all_prices"      : values.flatten(),
        }

    return results


def run_all_stocks() -> dict:
    """Jalankan pipeline untuk semua saham."""
    all_results = {}
    for ticker in STOCKS:
        all_results[ticker] = train_and_evaluate(ticker)
    return all_results


# ─────────────────────────────────────────────
# MAIN (standalone)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all_stocks()

    print("\n\n" + "="*55)
    print("  RINGKASAN PERBANDINGAN LSTM vs GRU")
    print("="*55)
    for ticker, res in results.items():
        print(f"\n  {ticker}")
        print(f"  {'Model':<8} {'MAE':>10} {'MSE':>14} {'MAPE':>10}")
        print(f"  {'-'*46}")
        for m in ["LSTM", "GRU"]:
            print(f"  {m:<8} {res[m]['mae']:>10.2f} {res[m]['mse']:>14.2f} {res[m]['mape']:>9.2f}%")

        print(f"\n  Prediksi 3 Hari ke Depan:")
        for m in ["LSTM", "GRU"]:
            fp = res[m]["future_3days"]
            print(f"    {m}: Hari+1={fp[0]:.2f}  Hari+2={fp[1]:.2f}  Hari+3={fp[2]:.2f}")
