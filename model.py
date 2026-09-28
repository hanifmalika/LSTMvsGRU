"""
model.py - Unified LSTM & GRU Stock Price Prediction
Saham: BBRI, BMRI, BBTN, BBNI
Target: Harga Penutupan
Metrik: MAE, MSE, MAPE

CATATAN:
- Mendukung DUA sumber data:
    1. Dataset otomatis dari Yahoo Finance (yfinance), rentang 2015-2025
       (STOCKS, START_DATE, END_DATE di bawah) -> load_yahoo_multistock()
    2. Data manual yang diinput user lewat file Excel (.xlsx)
       -> load_excel_multisheet()
- Kedua sumber dinormalisasi ke format yang sama:
  DataFrame dengan index bernama "Tanggal" dan satu kolom "ha" (harga),
  sehingga pipeline training (train_and_evaluate) bisa dipakai untuk
  keduanya tanpa perbedaan kode.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
STOCKS        = ["BBRI.JK", "BMRI.JK", "BBTN.JK", "BBNI.JK"]
START_DATE    = "2015-01-01"
END_DATE      = "2025-12-31"

TIMESTEPS     = 30          # lookback window
TEST_SIZE     = 0.2
EPOCHS        = 50
BATCH_SIZE    = 32
FUTURE_DAYS   = 3           # prediksi ke depan
LEARNING_RATE = 0.0001

# Nama kolom yang dikenali otomatis saat membaca Excel (tidak case-sensitive)
DATE_COL_CANDIDATES  = ["tanggal", "date", "tgl", "waktu"]
PRICE_COL_CANDIDATES = ["adj close", "close", "harga penutupan", "harga", "ha", "closing price"]


# ─────────────────────────────────────────────
# 1a. SUMBER DATA: YAHOO FINANCE (otomatis, 2015-2025)
# ─────────────────────────────────────────────

def load_data_yahoo(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Download data historis dari Yahoo Finance untuk satu ticker.
    Kompatibel dengan yfinance lama (Adj Close) dan baru (MultiIndex / Close).
    Hasil distandarisasi ke format yang sama dengan data Excel manual:
    index bernama "Tanggal", satu kolom "ha".
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError(
            "Library 'yfinance' tidak tersedia di environment ini. "
            "Gunakan mode input Excel manual sebagai gantinya."
        )

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(
            f"Tidak ada data yang diterima untuk {ticker}. "
            f"Kemungkinan akses ke Yahoo Finance diblokir atau ticker salah."
        )

    # Flatten MultiIndex columns jika ada (yfinance >= 0.2.38)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Pilih kolom harga: utamakan 'Adj Close', fallback ke 'Close'
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
    elif "Close" in df.columns:
        df = df[["Close"]].copy()
    else:
        raise ValueError(
            f"Kolom harga tidak ditemukan untuk {ticker}. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    df.columns = ["ha"]
    df = df.dropna()
    df.index.name = "Tanggal"

    if len(df) < TIMESTEPS + 10:
        raise ValueError(
            f"Data {ticker} hanya {len(df)} baris, minimal butuh sekitar "
            f"{TIMESTEPS + 10} baris data historis."
        )

    return df


def load_yahoo_multistock(tickers: list = None, start: str = START_DATE,
                           end: str = END_DATE) -> dict:
    """
    Download data historis untuk beberapa saham sekaligus dari Yahoo Finance.
    Return: dict {ticker: DataFrame_bersih}  -- format sama dengan
    load_excel_multisheet(), sehingga bisa langsung dipakai run_all_stocks().

    Jika satu/lebih ticker gagal diambil, ticker lain tetap diproses;
    error dikumpulkan dan dilempar sebagai ValueError hanya jika SEMUA
    ticker gagal.
    """
    tickers = tickers or STOCKS
    cleaned = {}
    errors = {}

    for ticker in tickers:
        try:
            cleaned[ticker] = load_data_yahoo(ticker, start=start, end=end)
        except Exception as e:
            errors[ticker] = str(e)

    if not cleaned:
        pesan = "\n".join([f"  - {k}: {v}" for k, v in errors.items()])
        raise ValueError(
            f"Gagal mengambil data dari Yahoo Finance untuk semua saham:\n{pesan}\n\n"
            f"Kemungkinan akses ke Yahoo Finance diblokir di jaringan ini. "
            f"Gunakan mode Upload Excel Manual sebagai alternatif."
        )

    return cleaned, errors


# ─────────────────────────────────────────────
# 1b. SUMBER DATA: EXCEL MANUAL
# ─────────────────────────────────────────────

def load_excel_multisheet(file) -> dict:
    """
    Membaca satu file Excel yang berisi beberapa sheet (1 sheet = 1 saham).
    `file` bisa berupa path string ATAU file-like object (mis. hasil
    st.file_uploader di Streamlit).

    Return: dict {nama_sheet: DataFrame_bersih}
    """
    raw_sheets = pd.read_excel(file, sheet_name=None)  # dict semua sheet
    cleaned = {}
    errors = {}

    for sheet_name, df_raw in raw_sheets.items():
        try:
            cleaned[sheet_name] = validate_and_clean(df_raw, sheet_name)
        except ValueError as e:
            errors[sheet_name] = str(e)

    if errors:
        pesan = "\n".join([f"  - Sheet '{k}': {v}" for k, v in errors.items()])
        raise ValueError(
            f"Beberapa sheet gagal diproses:\n{pesan}\n\n"
            f"Pastikan setiap sheet punya kolom tanggal (mis. 'Tanggal'/'Date') "
            f"dan kolom harga (mis. 'Close'/'Harga Penutupan')."
        )

    if not cleaned:
        raise ValueError("Tidak ada sheet valid yang bisa diproses dari file Excel ini.")

    return cleaned


def validate_and_clean(df_raw: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Validasi & bersihkan satu DataFrame mentah hasil input manual (Excel):
    - Deteksi otomatis kolom tanggal & kolom harga (case-insensitive).
    - Parse tanggal, urutkan ascending (wajib untuk time series).
    - Buang baris kosong/duplikat.
    - Pastikan kolom harga numerik.
    - Standarisasi nama kolom jadi ["ha"], index bernama "Tanggal".
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("sheet kosong.")

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower().strip(): c for c in df.columns}

    # cari kolom tanggal
    date_col = next((lower_map[c] for c in DATE_COL_CANDIDATES if c in lower_map), None)
    if date_col is None:
        raise ValueError(
            f"kolom tanggal tidak ditemukan. Kolom tersedia: {df.columns.tolist()}"
        )

    # cari kolom harga
    price_col = next((lower_map[c] for c in PRICE_COL_CANDIDATES if c in lower_map), None)
    if price_col is None:
        raise ValueError(
            f"kolom harga tidak ditemukan. Kolom tersedia: {df.columns.tolist()}"
        )

    df = df[[date_col, price_col]].rename(columns={date_col: "Tanggal", price_col: "ha"})

    # parsing & pembersihan
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=False)
    df["ha"] = pd.to_numeric(df["ha"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["Tanggal", "ha"])
    df = df.drop_duplicates(subset=["Tanggal"])
    df = df.sort_values("Tanggal")
    n_after = len(df)

    if n_after < TIMESTEPS + 10:
        raise ValueError(
            f"data valid hanya {n_after} baris (dari {n_before}). "
            f"Minimal butuh sekitar {TIMESTEPS + 10} baris data historis."
        )

    df = df.set_index("Tanggal")
    return df[["ha"]]


# ─────────────────────────────────────────────
# 2. NORMALISASI DATA
# ─────────────────────────────────────────────

def normalize_series(values: np.ndarray):
    """
    Normalisasi data harga ke rentang [0, 1] dengan MinMaxScaler
    (dibutuhkan LSTM/GRU agar training stabil & cepat konvergen).

    Return: (data_scaled, scaler) — scaler disimpan untuk inverse_transform nanti.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    """Ubah data menjadi sequence (sliding window) untuk input RNN."""
    x_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        x_seq.append(X[i: i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(x_seq), np.array(y_seq)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ─────────────────────────────────────────────
# 3. MODEL LSTM & GRU
# ─────────────────────────────────────────────

def build_lstm(input_shape):
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


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUASI
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, label: str = "",
                        timesteps: int = TIMESTEPS,
                        epochs: int = EPOCHS,
                        future_days: int = FUTURE_DAYS,
                        progress_cb=None) -> dict:
    """
    Pipeline lengkap untuk satu saham. Bekerja untuk DataFrame dari sumber
    manapun (Yahoo Finance atau Excel manual), selama formatnya sudah
    distandarisasi: index "Tanggal", kolom "ha".
    - Normalisasi
    - Split train/test
    - Latih LSTM & GRU
    - Evaluasi MAE, MSE, MAPE
    - Prediksi N hari ke depan
    """
    print(f"\n{'='*55}")
    print(f"  MEMPROSES: {label}")
    print(f"{'='*55}")

    values = df["ha"].values.reshape(-1, 1)

    # 1. Normalisasi
    scaled, scaler = normalize_series(values)

    # 2. Train/Test Split (tanpa shuffle - time series)
    split = int(len(scaled) * (1 - TEST_SIZE))
    train_scaled, test_scaled = scaled[:split], scaled[split:]

    # 3. Sequence Creation
    X_train_seq, y_train_seq = create_sequences(train_scaled, train_scaled, timesteps)
    X_test_seq, y_test_seq   = create_sequences(test_scaled, test_scaled, timesteps)

    input_shape = (timesteps, 1)
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    results = {}

    for model_name, build_fn in [("LSTM", build_lstm), ("GRU", build_gru)]:
        if progress_cb:
            progress_cb(model_name)
        print(f"\n  > Melatih {model_name}...")
        model = build_fn(input_shape)
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_split=0.1,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        pred_scaled   = model.predict(X_test_seq, verbose=0)
        pred_actual   = scaler.inverse_transform(pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test_seq)

        mae_val  = mean_absolute_error(y_test_actual, pred_actual)
        mse_val  = mean_squared_error(y_test_actual, pred_actual)
        mape_val = mape(y_test_actual, pred_actual)

        print(f"    MAE : {mae_val:.4f}")
        print(f"    MSE : {mse_val:.4f}")
        print(f"    MAPE: {mape_val:.4f}%")

        # Prediksi N hari ke depan
        last_sequence = scaled[-timesteps:].reshape(1, timesteps, 1)
        future_preds = []
        current_seq = last_sequence.copy()

        for _ in range(future_days):
            next_pred = model.predict(current_seq, verbose=0)
            future_preds.append(next_pred[0, 0])
            current_seq = np.append(
                current_seq[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )

        future_preds_actual = scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        results[model_name] = {
            "predictions"  : pred_actual.flatten(),
            "actuals"      : y_test_actual.flatten(),
            "mae"          : mae_val,
            "mse"          : mse_val,
            "mape"         : mape_val,
            "future"       : future_preds_actual,
            "train_loss"   : history.history["loss"],
            "val_loss"     : history.history.get("val_loss", []),
            "test_dates"   : df.index[split + timesteps: split + timesteps + len(pred_actual)],
            "all_dates"    : df.index,
            "all_prices"   : values.flatten(),
        }

    return results


def run_all_stocks(stock_data: dict, timesteps: int = TIMESTEPS,
                    epochs: int = EPOCHS, future_days: int = FUTURE_DAYS,
                    progress_cb=None) -> dict:
    """
    Jalankan pipeline untuk semua saham.
    `stock_data`: dict {nama_saham: DataFrame_bersih}, hasil dari
    `load_excel_multisheet()` ATAU `load_yahoo_multistock()`.
    """
    all_results = {}
    for label, df in stock_data.items():
        all_results[label] = train_and_evaluate(
            df, label=label, timesteps=timesteps,
            epochs=epochs, future_days=future_days, progress_cb=progress_cb
        )
    return all_results


# ─────────────────────────────────────────────
# MAIN (standalone / CLI)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "yahoo"

    if mode == "yahoo":
        print(f"Mengambil data dari Yahoo Finance ({START_DATE} s/d {END_DATE})...")
        stock_data, errors = load_yahoo_multistock()
        if errors:
            print("Peringatan, beberapa ticker gagal:")
            for k, v in errors.items():
                print(f"  - {k}: {v}")
    else:
        excel_path = mode
        print(f"Membaca data dari: {excel_path}")
        stock_data = load_excel_multisheet(excel_path)

    print(f"Saham terdeteksi: {list(stock_data.keys())}")

    results = run_all_stocks(stock_data)

    print("\n\n" + "="*55)
    print("  RINGKASAN PERBANDINGAN LSTM vs GRU")
    print("="*55)
    for label, res in results.items():
        print(f"\n  {label}")
        print(f"  {'Model':<8} {'MAE':>10} {'MSE':>14} {'MAPE':>10}")
        print(f"  {'-'*46}")
        for m in ["LSTM", "GRU"]:
            print(f"  {m:<8} {res[m]['mae']:>10.2f} {res[m]['mse']:>14.2f} {res[m]['mape']:>9.2f}%")

        print(f"\n  Prediksi ke Depan:")
        for m in ["LSTM", "GRU"]:
            fp = res[m]["future"]
            preview = "  ".join([f"H+{i+1}={v:.2f}" for i, v in enumerate(fp)])
            print(f"    {m}: {preview}")
