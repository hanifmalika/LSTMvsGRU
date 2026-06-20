"""
app.py  –  Streamlit Dashboard: LSTM vs GRU Stock Prediction
Saham: BBRI, BMRI, BBTN, BBNI  |  Metrik: MAE, MSE, MAPE
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight | LSTM vs GRU",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 40%, #0a1628 100%);
    color: #e2e8f0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111d35 100%);
    border-right: 1px solid #1e3a5f;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0f2040 0%, #162a4a 100%);
    border: 1px solid #1e4a7a;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-label {
    font-size: 12px; color: #64b5f6; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 700; color: #ffffff;
  }
  .metric-sub { font-size: 11px; color: #90caf9; margin-top: 4px; }

  /* Section header */
  .section-header {
    background: linear-gradient(90deg, #1565c0, #0d47a1);
    border-radius: 12px;
    padding: 14px 22px;
    margin: 24px 0 16px;
    font-size: 18px; font-weight: 700; color: #ffffff;
    border-left: 4px solid #42a5f5;
    box-shadow: 0 4px 15px rgba(21,101,192,0.3);
  }

  /* Winner badge */
  .badge-winner {
    display: inline-block;
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: #a5d6a7; border-radius: 20px;
    padding: 4px 14px; font-size: 12px; font-weight: 700;
    border: 1px solid #43a047; letter-spacing: 1px;
    text-transform: uppercase;
  }
  .badge-runner {
    display: inline-block;
    background: linear-gradient(135deg, #1a237e, #283593);
    color: #9fa8da; border-radius: 20px;
    padding: 4px 14px; font-size: 12px; font-weight: 700;
    border: 1px solid #3949ab; letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* Hero title */
  .hero-title {
    font-size: 48px; font-weight: 700;
    background: linear-gradient(135deg, #42a5f5, #1976d2, #0d47a1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 8px;
  }
  .hero-sub { color: #78909c; font-size: 16px; margin-bottom: 24px; }

  /* Future table */
  .future-table { width: 100%; border-collapse: collapse; }
  .future-table th {
    background: #1565c0; color: white; padding: 10px 14px;
    font-size: 13px; text-align: center; letter-spacing: 0.5px;
  }
  .future-table td {
    padding: 10px 14px; text-align: center;
    border-bottom: 1px solid #1e3a5f; color: #e2e8f0; font-size: 14px;
  }
  .future-table tr:hover td { background: rgba(21,101,192,0.15); }

  /* Progress */
  .stProgress > div > div { background: #1565c0; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white; border: none; border-radius: 10px;
    padding: 10px 28px; font-weight: 600; font-size: 15px;
    transition: all 0.2s; width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    box-shadow: 0 6px 20px rgba(21,101,192,0.4);
    transform: translateY(-2px);
  }

  /* Divider */
  hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
STOCKS_META = {
    "BBRI.JK": {"name": "Bank Rakyat Indonesia", "short": "BBRI", "color": "#42a5f5"},
    "BMRI.JK": {"name": "Bank Mandiri",          "short": "BMRI", "color": "#66bb6a"},
    "BBTN.JK": {"name": "Bank Tabungan Negara",  "short": "BBTN", "color": "#ffa726"},
    "BBNI.JK": {"name": "Bank Negara Indonesia", "short": "BBNI", "color": "#ef5350"},
    
}
TIMESTEPS    = 30
TEST_SIZE    = 0.2
EPOCHS       = 50
BATCH_SIZE   = 32
FUTURE_DAYS  = 3
LR           = 0.0001


# ─────────────────────────────────────────────
# ML FUNCTIONS
# ─────────────────────────────────────────────

def mape_score(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_sequences(X, y, ts):
    xs, ys = [], []
    for i in range(len(X) - ts):
        xs.append(X[i:i+ts])
        ys.append(y[i+ts])
    return np.array(xs), np.array(ys)


def build_model(model_type, input_shape):
    layer = LSTM if model_type == "LSTM" else GRU
    m = Sequential([
        layer(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        layer(50),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=LR, clipnorm=1.0), loss="mse", metrics=["mae"])
    return m


@st.cache_data(show_spinner=False)
def load_stock_data(ticker, start="2015-01-01", end="2025-12-31"):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    # Flatten MultiIndex columns (yfinance >= 0.2.38)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Pilih kolom harga
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
    elif "Close" in df.columns:
        df = df[["Close"]].copy()
    else:
        raise ValueError(f"Kolom harga tidak ditemukan. Tersedia: {df.columns.tolist()}")

    df.columns = ["ha"]
    df = df.dropna()
    return df


def train_stock(ticker, progress_cb=None):
    df = load_stock_data(ticker)
    values = df["ha"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(values)
    y_sc = scaler_y.fit_transform(values)

    split = int(len(X_sc) * (1 - TEST_SIZE))
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y_sc[:split], y_sc[split:]

    X_tr_s, y_tr_s = create_sequences(X_tr, y_tr, TIMESTEPS)
    X_te_s, y_te_s = create_sequences(X_te, y_te, TIMESTEPS)

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    results = {}

    for i, mtype in enumerate(["LSTM", "GRU"]):
        if progress_cb:
            progress_cb(mtype)
        model = build_model(mtype, (TIMESTEPS, 1))
        hist = model.fit(
            X_tr_s, y_tr_s,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )
        pred_sc = model.predict(X_te_s, verbose=0)
        pred    = scaler_y.inverse_transform(pred_sc)
        actual  = scaler_y.inverse_transform(y_te_s)

        # Future
        cur = X_sc[-TIMESTEPS:].reshape(1, TIMESTEPS, 1)
        fp  = []
        for _ in range(FUTURE_DAYS):
            nxt = model.predict(cur, verbose=0)
            fp.append(nxt[0, 0])
            cur = np.append(cur[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
        fp_actual = scaler_y.inverse_transform(np.array(fp).reshape(-1,1)).flatten()

        train_loss = hist.history["loss"]
        val_loss   = hist.history.get("val_loss", [])

        results[mtype] = {
            "predictions" : pred.flatten(),
            "actuals"     : actual.flatten(),
            "mae"         : mean_absolute_error(actual, pred),
            "mse"         : mean_squared_error(actual, pred),
            "mape"        : mape_score(actual, pred),
            "future"      : fp_actual,
            "train_loss"  : train_loss,
            "val_loss"    : val_loss,
            "test_dates"  : df.index[split + TIMESTEPS:],
            "all_dates"   : df.index,
            "all_prices"  : values.flatten(),
        }

    return results


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("---")

    selected_stocks = st.multiselect(
        "Pilih Saham Bank",
        options=list(STOCKS_META.keys()),
        default=list(STOCKS_META.keys()),
        format_func=lambda x: f"{STOCKS_META[x]['short']} – {STOCKS_META[x]['name']}"
    )

    st.markdown("---")
    st.markdown("### 🔧 Parameter Model")
    epochs_ui    = st.slider("Epochs",     10, 100, EPOCHS, 5)
    timesteps_ui = st.slider("Timesteps",  10, 60,  TIMESTEPS, 5)
    future_ui    = st.slider("Hari Prediksi", 1, 7,  FUTURE_DAYS)

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Model", use_container_width=True)

    st.markdown("---")
    st.caption("📊 Data: Yahoo Finance | Model: TensorFlow/Keras")
    st.caption("Saham perbankan Indonesia 2015–2025")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">StockSight</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">🏦 Prediksi Harga Saham Bank Indonesia · LSTM vs GRU · Evaluasi MAE · MSE · MAPE</div>', unsafe_allow_html=True)
st.markdown("---")

if not selected_stocks:
    st.warning("⚠️ Pilih minimal satu saham di sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────
tab_overview, tab_model, tab_forecast, tab_compare = st.tabs([
    "📊 Overview Harga",
    "🧠 Hasil Model",
    "🔮 Forecasting 3 Hari",
    "⚖️ Perbandingan Kinerja"
])

# ══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">📈 Histori Harga Penutupan Disesuaikan</div>', unsafe_allow_html=True)

    fig = go.Figure()
    for ticker in selected_stocks:
        meta = STOCKS_META[ticker]
        with st.spinner(f"Memuat {meta['short']}..."):
            df = load_stock_data(ticker)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ha"],
            name=meta["short"], line=dict(color=meta["color"], width=2),
            hovertemplate=f"<b>{meta['short']}</b><br>Tanggal: %{{x|%d %b %Y}}<br>Harga: Rp%{{y:,.0f}}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,32,64,0.4)",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f", borderwidth=1),
        xaxis=dict(gridcolor="#1e3a5f", showgrid=True),
        yaxis=dict(gridcolor="#1e3a5f", showgrid=True, title="Harga (IDR)"),
        hovermode="x unified",
        height=480,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats cards
    st.markdown('<div class="section-header">📋 Statistik Deskriptif</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected_stocks))
    for i, ticker in enumerate(selected_stocks):
        meta = STOCKS_META[ticker]
        df   = load_stock_data(ticker)
        with cols[i]:
            latest = df["ha"].iloc[-1]   # harga terbaru
            oldest = df["ha"].iloc[0]    # harga tertua (2015)
            pct    = (latest - oldest) / oldest * 100
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{meta['short']}</div>
              <div class="metric-value">Rp{latest:,.0f}</div>
              <div class="metric-sub">
                {meta['name']}<br>
                Min: Rp{df['ha'].min():,.0f} | Max: Rp{df['ha'].max():,.0f}<br>
                Return: {"+" if pct>=0 else ""}{pct:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# RUN MODEL
# ══════════════════════════════════════════════
if run_btn or ("model_results" in st.session_state):

    if run_btn:
        # Jalankan model dengan progress bar
        TIMESTEPS   = timesteps_ui
        FUTURE_DAYS = future_ui
        EPOCHS      = epochs_ui

        all_results = {}
        progress_bar = st.progress(0, text="Memulai pelatihan model...")
        total = len(selected_stocks) * 2
        step  = 0

        for ticker in selected_stocks:
            meta = STOCKS_META[ticker]

            def update_progress(mtype):
                global step
                step += 1
                progress_bar.progress(
                    step / total,
                    text=f"⚙️ Melatih {mtype} untuk {meta['short']}... ({step}/{total})"
                )

            all_results[ticker] = train_stock(ticker, progress_cb=update_progress)

        progress_bar.progress(1.0, text="✅ Pelatihan selesai!")
        st.session_state["model_results"] = all_results
        st.success("✅ Model LSTM & GRU berhasil dilatih untuk semua saham!")
    else:
        all_results = st.session_state["model_results"]

    # ══════════════════════════════════════════
    # TAB 2: HASIL MODEL
    # ══════════════════════════════════════════
    with tab_model:
        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]

            st.markdown(f'<div class="section-header">🏦 {meta["short"]} – {meta["name"]}</div>', unsafe_allow_html=True)

            # Metrics
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            metrics_data = [
                (c1, "LSTM MAE",  f"{res['LSTM']['mae']:,.2f}",  "Mean Absolute Error"),
                (c2, "LSTM MSE",  f"{res['LSTM']['mse']:,.2f}",  "Mean Squared Error"),
                (c3, "LSTM MAPE", f"{res['LSTM']['mape']:.2f}%", "Mean Abs % Error"),
                (c4, "GRU MAE",   f"{res['GRU']['mae']:,.2f}",   "Mean Absolute Error"),
                (c5, "GRU MSE",   f"{res['GRU']['mse']:,.2f}",   "Mean Squared Error"),
                (c6, "GRU MAPE",  f"{res['GRU']['mape']:.2f}%",  "Mean Abs % Error"),
            ]
            for col, label, value, sub in metrics_data:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value" style="font-size:22px">{value}</div>
                      <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            # Plot Prediksi vs Aktual
            fig2 = make_subplots(rows=1, cols=2,
                subplot_titles=["LSTM: Prediksi vs Aktual", "GRU: Prediksi vs Aktual"])

            for col_i, mtype in enumerate(["LSTM", "GRU"], 1):
                r = res[mtype]
                dates = r["test_dates"][:len(r["actuals"])]
                fig2.add_trace(go.Scatter(x=dates, y=r["actuals"], name="Aktual",
                    line=dict(color="#90caf9", width=2),
                    hovertemplate="Aktual: Rp%{y:,.0f}<extra></extra>"), row=1, col=col_i)
                fig2.add_trace(go.Scatter(x=dates, y=r["predictions"], name=f"{mtype} Prediksi",
                    line=dict(color=meta["color"], width=2, dash="dash"),
                    hovertemplate=f"{mtype}: Rp%{{y:,.0f}}<extra></extra>"), row=1, col=col_i)

            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,64,0.4)",
                font=dict(color="#e2e8f0"), height=400,
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
                xaxis2=dict(gridcolor="#1e3a5f"), yaxis2=dict(gridcolor="#1e3a5f"),
                showlegend=False, margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Training Loss
            fig3 = make_subplots(rows=1, cols=2,
                subplot_titles=["LSTM Training Loss", "GRU Training Loss"])
            for col_i, mtype in enumerate(["LSTM", "GRU"], 1):
                r = res[mtype]
                fig3.add_trace(go.Scatter(y=r["train_loss"], name="Train Loss",
                    line=dict(color=meta["color"])), row=1, col=col_i)
                if r["val_loss"]:
                    fig3.add_trace(go.Scatter(y=r["val_loss"], name="Val Loss",
                        line=dict(color="#ef9a9a", dash="dot")), row=1, col=col_i)
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,64,0.4)",
                font=dict(color="#e2e8f0"), height=300,
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
                xaxis2=dict(gridcolor="#1e3a5f"), yaxis2=dict(gridcolor="#1e3a5f"),
                showlegend=False, margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("---")

    # ══════════════════════════════════════════
    # TAB 3: FORECASTING 3 HARI
    # ══════════════════════════════════════════
    with tab_forecast:
        st.markdown('<div class="section-header">🔮 Prediksi Harga 3 Hari ke Depan</div>', unsafe_allow_html=True)

        import datetime
        today = datetime.date.today()
        future_dates = [
            (today + datetime.timedelta(days=i)).strftime("%d %b %Y")
            for i in range(1, FUTURE_DAYS + 1)
        ]

        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]
            df   = load_stock_data(ticker)
            last_price = float(df["ha"].iloc[-1])  # harga terbaru (data terakhir)

            st.markdown(f"#### 🏦 {meta['short']} – {meta['name']}")

            # Chart
            fig4 = go.Figure()
            # Historical tail
            hist_tail = df.head(30)
            fig4.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail["ha"],
                name="Histori", line=dict(color="#546e7a", width=2)
            ))

            colors_m = {"LSTM": "#42a5f5", "GRU": "#66bb6a"}
            today_str = today.strftime("%Y-%m-%d")
            for mtype in ["LSTM", "GRU"]:
                fp = res[mtype]["future"]
                x_future = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                            for i in range(1, len(fp)+1)]
                fig4.add_trace(go.Scatter(
                    x=[today_str] + x_future,
                    y=[last_price] + list(fp),
                    name=f"{mtype} Prediksi",
                    line=dict(color=colors_m[mtype], width=3, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=8)
                ))

            # Garis vertikal manual (lebih kompatibel lintas versi Plotly)
            fig4.add_shape(
                type="line",
                x0=today_str, x1=today_str,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#ffa726", dash="dot", width=2)
            )
            fig4.add_annotation(
                x=today_str, y=1, xref="x", yref="paper",
                text="Hari ini", showarrow=False,
                font=dict(color="#ffa726", size=12),
                xanchor="left", yanchor="top"
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,64,0.4)",
                font=dict(color="#e2e8f0"), height=360,
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f", title="Harga (IDR)"),
                legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f"),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Table
            rows = ""
            for i, date_str in enumerate(future_dates[:FUTURE_DAYS]):
                lstm_p = res["LSTM"]["future"][i] if i < len(res["LSTM"]["future"]) else "-"
                gru_p  = res["GRU"]["future"][i]  if i < len(res["GRU"]["future"])  else "-"
                lstm_ch = (lstm_p - last_price) / last_price * 100
                gru_ch  = (gru_p  - last_price) / last_price * 100
                lstm_arrow = "▲" if lstm_ch >= 0 else "▼"
                gru_arrow  = "▲" if gru_ch  >= 0 else "▼"
                lstm_col   = "#66bb6a" if lstm_ch >= 0 else "#ef5350"
                gru_col    = "#66bb6a" if gru_ch  >= 0 else "#ef5350"
                rows += f"""
                <tr>
                  <td><b>{date_str}</b></td>
                  <td>Rp{lstm_p:,.0f}</td>
                  <td style="color:{lstm_col}">{lstm_arrow} {abs(lstm_ch):.2f}%</td>
                  <td>Rp{gru_p:,.0f}</td>
                  <td style="color:{gru_col}">{gru_arrow} {abs(gru_ch):.2f}%</td>
                </tr>"""

            st.markdown(f"""
            <table class="future-table">
              <tr>
                <th>Tanggal</th>
                <th>LSTM Prediksi</th>
                <th>LSTM Δ%</th>
                <th>GRU Prediksi</th>
                <th>GRU Δ%</th>
              </tr>
              {rows}
            </table>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")

    # ══════════════════════════════════════════
    # TAB 4: PERBANDINGAN KINERJA
    # ══════════════════════════════════════════
    with tab_compare:
        st.markdown('<div class="section-header">⚖️ Perbandingan Kinerja LSTM vs GRU</div>', unsafe_allow_html=True)

        # Summary table
        rows_data = []
        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]
            for mtype in ["LSTM", "GRU"]:
                rows_data.append({
                    "Saham"  : meta["short"],
                    "Model"  : mtype,
                    "MAE"    : res[mtype]["mae"],
                    "MSE"    : res[mtype]["mse"],
                    "MAPE(%)" : res[mtype]["mape"],
                })

        df_cmp = pd.DataFrame(rows_data)

        # Bar chart comparison MAE
        fig_bar = make_subplots(rows=1, cols=3,
            subplot_titles=["MAE (lebih kecil = lebih baik)",
                            "MSE (lebih kecil = lebih baik)",
                            "MAPE % (lebih kecil = lebih baik)"])

        colors_m = {"LSTM": "#42a5f5", "GRU": "#66bb6a"}
        for col_i, metric in enumerate(["MAE", "MSE", "MAPE(%)"], 1):
            for mtype in ["LSTM", "GRU"]:
                d = df_cmp[df_cmp["Model"] == mtype]
                fig_bar.add_trace(go.Bar(
                    x=d["Saham"], y=d[metric], name=mtype,
                    marker_color=colors_m[mtype],
                    text=d[metric].apply(lambda v: f"{v:.2f}"),
                    textposition="outside",
                    showlegend=(col_i == 1)
                ), row=1, col=col_i)

        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,64,0.4)",
            font=dict(color="#e2e8f0"), height=420,
            legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            xaxis2=dict(gridcolor="#1e3a5f"), yaxis2=dict(gridcolor="#1e3a5f"),
            xaxis3=dict(gridcolor="#1e3a5f"), yaxis3=dict(gridcolor="#1e3a5f"),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart
        st.markdown('<div class="section-header">🕸 Radar Chart Perbandingan</div>', unsafe_allow_html=True)
        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]

            categories = ["MAE", "MSE", "MAPE"]
            fig_radar = go.Figure()
            radar_colors = {"LSTM": ("#42a5f5", "rgba(66,165,245,0.15)"),
                              "GRU":  ("#66bb6a", "rgba(102,187,106,0.15)")}
            for mtype in ["LSTM", "GRU"]:
                line_color, fill_color = radar_colors[mtype]
                vals = [
                    res[mtype]["mae"],
                    res[mtype]["mse"],
                    res[mtype]["mape"]
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    name=mtype, fill="toself",
                    line=dict(color=line_color),
                    fillcolor=fill_color
                ))
            fig_radar.update_layout(
                polar=dict(bgcolor="rgba(15,32,64,0.4)",
                    radialaxis=dict(visible=True, color="#90caf9"),
                    angularaxis=dict(color="#90caf9")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                title=dict(text=f"{meta['short']} – {meta['name']}", font=dict(size=15)),
                showlegend=True,
                legend=dict(bgcolor="rgba(13,21,38,0.8)"),
                height=380,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Winner Summary ──
        st.markdown('<div class="section-header">🏆 Kesimpulan: Model Terbaik per Saham</div>', unsafe_allow_html=True)
        win_cols = st.columns(len(selected_stocks))
        overall_wins = {"LSTM": 0, "GRU": 0}

        for i, ticker in enumerate(selected_stocks):
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]

            lstm_score = res["LSTM"]["mae"] + res["LSTM"]["mape"]
            gru_score  = res["GRU"]["mae"]  + res["GRU"]["mape"]
            winner     = "LSTM" if lstm_score < gru_score else "GRU"
            loser      = "GRU"  if winner == "LSTM" else "LSTM"
            overall_wins[winner] += 1

            with win_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                  <div class="metric-label">{meta['short']}</div>
                  <div style="font-size:32px; margin: 8px 0">🏆</div>
                  <span class="badge-winner">{winner}</span><br><br>
                  <span class="badge-runner">{loser} Runner-up</span>
                  <div class="metric-sub" style="margin-top:12px">
                    MAE: {winner}={res[winner]['mae']:,.0f} | {loser}={res[loser]['mae']:,.0f}<br>
                    MAPE: {winner}={res[winner]['mape']:.2f}% | {loser}={res[loser]['mape']:.2f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Overall winner
        st.markdown("---")
        overall_winner = max(overall_wins, key=overall_wins.get)
        other = "GRU" if overall_winner == "LSTM" else "LSTM"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-radius:16px;
            padding:28px 32px;text-align:center;border:1px solid #43a047;
            box-shadow:0 8px 32px rgba(27,94,32,0.4)">
          <div style="font-size:48px">🥇</div>
          <div style="font-size:28px;font-weight:700;color:#a5d6a7;margin:8px 0">
            {overall_winner} adalah Model Terbaik Secara Keseluruhan
          </div>
          <div style="color:#81c784;font-size:16px">
            Menang di {overall_wins[overall_winner]} dari {len(selected_stocks)} saham
            ({overall_wins[other]} kali {other} lebi h unggul)
          </div>
          <div style="color:#a5d6a7;margin-top:12px;font-size:14px">
            Berdasarkan MAE + MAPE terkecil sebagai kriteria utama evaluasi
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Prompt to run
    with tab_model, tab_forecast, tab_compare:
        st.info("👆 Klik tombol **🚀 Jalankan Model** di sidebar untuk memulai pelatihan LSTM & GRU.")