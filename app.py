"""
app.py  –  Streamlit Dashboard: LSTM vs GRU Stock Prediction
Saham: BBRI, BMRI, BBTN, BBNI  |  Metrik: MAE, MSE, MAPE
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* =========================
   GLOBAL
========================= */

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.stApp{
    background-color:#0d1117;
    color:#f0f6fc;
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"]{
    background-color:#161b22 !important;
    border-right:1px solid #30363d;
}

[data-testid="stSidebar"] *{
    color:#f0f6fc !important;
}

[data-testid="stSidebar"] label{
    color:#58a6ff !important;
    font-weight:600;
}

/* =========================
   TITLE
========================= */

.hero-title{
    font-size:46px;
    font-weight:800;
    color:#58a6ff;
    margin-bottom:6px;
}

.hero-sub{
    color:#8b949e;
    font-size:16px;
}

/* =========================
   SECTION
========================= */

.section-header{
    background:#1f6feb;
    color:white;
    padding:14px 20px;
    border-radius:10px;
    font-size:18px;
    font-weight:700;
    margin:20px 0px;
}

/* =========================
   CARD
========================= */

.metric-card{
    background:#161b22;
    border:1px solid #30363d;
    border-radius:14px;
    padding:20px;
    margin-bottom:15px;
    box-shadow:0 3px 12px rgba(0,0,0,.25);
    transition:.2s;
}

.metric-card:hover{
    border-color:#58a6ff;
    transform:translateY(-3px);
}

.metric-label{
    color:#8b949e;
    font-size:12px;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:1px;
}

.metric-value{
    font-family:'JetBrains Mono', monospace;
    color:#ffffff;
    font-size:30px;
    font-weight:700;
    margin-top:8px;
}

.metric-sub{
    color:#58a6ff;
    font-size:13px;
    margin-top:6px;
}

/* =========================
   BADGE
========================= */

.badge-winner{
    display:inline-block;
    padding:5px 14px;
    border-radius:20px;
    background:#238636;
    color:white;
    font-size:12px;
    font-weight:700;
}

.badge-runner{
    display:inline-block;
    padding:5px 14px;
    border-radius:20px;
    background:#1f6feb;
    color:white;
    font-size:12px;
    font-weight:700;
}

/* =========================
   TABLE
========================= */

.future-table{
    width:100%;
    border-collapse:collapse;
}

.future-table th{
    background:#1f6feb;
    color:white;
    padding:12px;
    font-weight:600;
}

.future-table td{
    background:#161b22;
    color:#f0f6fc;
    padding:10px;
    border-bottom:1px solid #30363d;
}

.future-table tr:hover td{
    background:#21262d;
}

/* =========================
   BUTTON
========================= */

.stButton>button{
    width:100%;
    background:#1f6feb;
    color:white;
    border:none;
    border-radius:10px;
    padding:10px;
    font-weight:600;
}

.stButton>button:hover{
    background:#388bfd;
}
.stDownloadButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: 0.3s;
}

.stDownloadButton > button:hover {
    background: #388bfd;
    color: white;
}

/* =========================
   INPUT
========================= */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stDateInput input,
.stTextArea textarea{
    background:#21262d !important;
    color:white !important;
    border:1px solid #30363d !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stDateInput label,
.stTextArea label{
    color:#f0f6fc !important;
}

/* =========================
   DATAFRAME
========================= */

[data-testid="stDataFrame"]{
    border:1px solid #30363d;
    border-radius:10px;
}

/* =========================
   METRIC
========================= */

[data-testid="metric-container"]{
    background:#161b22;
    border:1px solid #30363d;
    border-radius:12px;
    padding:15px;
}

[data-testid="metric-container"] *{
    color:white !important;
}

/* =========================
   EXPANDER
========================= */

.streamlit-expanderHeader{
    color:white !important;
}

/* =========================
   MARKDOWN
========================= */

p, li, span{
    color:#f0f6fc;
}

/* =========================
   HR
========================= */

hr{
    border-color:#30363d;
}
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
TIMESTEPS   = 30
TEST_SIZE   = 0.2
EPOCHS      = 50
BATCH_SIZE  = 32
FUTURE_DAYS = 3
LR          = 0.0001


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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
    elif "Close" in df.columns:
        df = df[["Close"]].copy()
    else:
        raise ValueError(f"Kolom harga tidak ditemukan. Tersedia: {df.columns.tolist()}")

    df.columns = ["ha"]
    df = df.dropna()
    return df


def train_stock(ticker, timesteps, epochs, future_days, progress_cb=None):
    df = load_stock_data(ticker)
    values = df["ha"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    split = int(len(scaled) * (1 - TEST_SIZE))
    X_tr, X_te = scaled[:split], scaled[split:]
    y_tr, y_te = scaled[:split], scaled[split:]

    X_tr_s, y_tr_s = create_sequences(X_tr, y_tr, timesteps)
    X_te_s, y_te_s = create_sequences(X_te, y_te, timesteps)

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    results = {}

    for mtype in ["LSTM", "GRU"]:
        if progress_cb:
            progress_cb(mtype)
        model = build_model(mtype, (timesteps, 1))
        hist = model.fit(
            X_tr_s, y_tr_s,
            validation_split=0.1,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )
        pred_sc = model.predict(X_te_s, verbose=0)
        pred    = scaler.inverse_transform(pred_sc)
        actual  = scaler.inverse_transform(y_te_s)

        # Future prediction
        cur = scaled[-timesteps:].reshape(1, timesteps, 1)
        fp  = []
        for _ in range(future_days):
            nxt = model.predict(cur, verbose=0)
            fp.append(nxt[0, 0])
            cur = np.append(cur[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
        fp_actual = scaler.inverse_transform(np.array(fp).reshape(-1, 1)).flatten()

        results[mtype] = {
            "predictions" : pred.flatten(),
            "actuals"     : actual.flatten(),
            "mae"         : mean_absolute_error(actual, pred),
            "mse"         : mean_squared_error(actual, pred),
            "mape"        : mape_score(actual, pred),
            "future"      : fp_actual,
            "train_loss"  : hist.history["loss"],
            "val_loss"    : hist.history.get("val_loss", []),
            "test_dates"  : df.index[split + timesteps : split + timesteps + len(pred)],
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
    epochs_ui    = st.slider("Epochs",        10, 100, EPOCHS,      5)
    timesteps_ui = st.slider("Timesteps",     10, 60,  TIMESTEPS,   5)
    future_ui    = st.slider("Hari Prediksi",  1,  7,  FUTURE_DAYS)

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Model", use_container_width=True)

    st.markdown("---")
    st.caption(" Data: Yahoo Finance | Model: TensorFlow/Keras")
    st.caption("Saham perbankan Indonesia 2015–2025")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">StockSight</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub"> Prediksi Harga Saham Bank Indonesia · LSTM vs GRU · Evaluasi MAE · MSE · MAPE</div>', unsafe_allow_html=True)
st.markdown("---")

if not selected_stocks:
    st.warning("⚠️ Pilih minimal satu saham di sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────
tab_overview, tab_model, tab_forecast, tab_compare = st.tabs([
    " Overview Harga",
    " Hasil Model",
    " Forecasting",
    " Perbandingan Kinerja"
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

    st.markdown('<div class="section-header">📋 Statistik Deskriptif</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected_stocks))
    for i, ticker in enumerate(selected_stocks):
        meta = STOCKS_META[ticker]
        df   = load_stock_data(ticker)
        with cols[i]:
            latest = df["ha"].iloc[-1]
            oldest = df["ha"].iloc[0]
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
        all_results  = {}
        progress_bar = st.progress(0, text="Memulai pelatihan model...")
        total        = len(selected_stocks) * 2
        step         = [0]

        for ticker in selected_stocks:
            meta = STOCKS_META[ticker]

            def update_progress(mtype, _meta=meta):
                step[0] += 1
                progress_bar.progress(
                    step[0] / total,
                    text=f"⚙️ Melatih {mtype} untuk {_meta['short']}... ({step[0]}/{total})"
                )

            all_results[ticker] = train_stock(
                ticker,
                timesteps=timesteps_ui,
                epochs=epochs_ui,
                future_days=future_ui,
                progress_cb=update_progress
            )

        progress_bar.progress(1.0, text="✅ Pelatihan selesai!")
        st.session_state["model_results"]  = all_results
        st.session_state["future_days_ui"] = future_ui
        st.success("✅ Model LSTM & GRU berhasil dilatih untuk semua saham!")

    else:
        all_results = st.session_state["model_results"]

    future_days_used = st.session_state.get("future_days_ui", FUTURE_DAYS)

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
                r     = res[mtype]
                dates = r["test_dates"][:len(r["actuals"])]
                fig2.add_trace(go.Scatter(
                    x=dates, y=r["actuals"], name="Aktual",
                    line=dict(color="#90caf9", width=2),
                    hovertemplate="Aktual: Rp%{y:,.0f}<extra></extra>"
                ), row=1, col=col_i)
                fig2.add_trace(go.Scatter(
                    x=dates, y=r["predictions"], name=f"{mtype} Prediksi",
                    line=dict(color=meta["color"], width=2, dash="dash"),
                    hovertemplate=f"{mtype}: Rp%{{y:,.0f}}<extra></extra>"
                ), row=1, col=col_i)

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
                fig3.add_trace(go.Scatter(
                    y=r["train_loss"], name="Train Loss",
                    line=dict(color=meta["color"])
                ), row=1, col=col_i)
                if r["val_loss"]:
                    fig3.add_trace(go.Scatter(
                        y=r["val_loss"], name="Val Loss",
                        line=dict(color="#ef9a9a", dash="dot")
                    ), row=1, col=col_i)
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
    # TAB 3: FORECASTING
    # ══════════════════════════════════════════
    with tab_forecast:
        st.markdown('<div class="section-header">🔮 Prediksi Harga ke Depan</div>', unsafe_allow_html=True)

        today        = datetime.date.today()
        future_dates = [
            (today + datetime.timedelta(days=i)).strftime("%d %b %Y")
            for i in range(1, future_days_used + 1)
        ]

        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta       = STOCKS_META[ticker]
            res        = all_results[ticker]
            df         = load_stock_data(ticker)
            last_price = float(df["ha"].iloc[-1])

            st.markdown(f"#### 🏦 {meta['short']} – {meta['name']}")

            fig4      = go.Figure()
            hist_tail = df.tail(30)
            fig4.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail["ha"],
                name="Histori", line=dict(color="#546e7a", width=2)
            ))

            colors_m  = {"LSTM": "#42a5f5", "GRU": "#66bb6a"}
            today_str = today.strftime("%Y-%m-%d")
            for mtype in ["LSTM", "GRU"]:
                fp       = res[mtype]["future"]
                x_future = [
                    (today + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, len(fp) + 1)
                ]
                fig4.add_trace(go.Scatter(
                    x=[today_str] + x_future,
                    y=[last_price] + list(fp),
                    name=f"{mtype} Prediksi",
                    line=dict(color=colors_m[mtype], width=3, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=8)
                ))

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
                xaxis=dict(gridcolor="#1e3a5f"),
                yaxis=dict(gridcolor="#1e3a5f", title="Harga (IDR)"),
                legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f"),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Tabel prediksi
            rows = ""
            for i, date_str in enumerate(future_dates):
                lstm_p     = res["LSTM"]["future"][i]
                gru_p      = res["GRU"]["future"][i]
                lstm_ch    = (lstm_p - last_price) / last_price * 100
                gru_ch     = (gru_p  - last_price) / last_price * 100
                lstm_arrow = "▲" if lstm_ch >= 0 else "▼"
                gru_arrow  = "▲" if gru_ch  >= 0 else "▼"
                lstm_col   = "#66bb6a" if lstm_ch >= 0 else "#ef5350"
                gru_col    = "#66bb6a" if gru_ch  >= 0 else "#ef5350"
                rows += (
                    f"<tr>"
                    f"<td><b>{date_str}</b></td>"
                    f"<td>Rp{lstm_p:,.0f}</td>"
                    f"<td style=\"color:{lstm_col}\">{lstm_arrow} {abs(lstm_ch):.2f}%</td>"
                    f"<td>Rp{gru_p:,.0f}</td>"
                    f"<td style=\"color:{gru_col}\">{gru_arrow} {abs(gru_ch):.2f}%</td>"
                    f"</tr>"
                )

            table_html = (
                '<table class="future-table">'
                '<tr><th>Tanggal</th><th>LSTM Prediksi</th><th>LSTM Δ%</th>'
                '<th>GRU Prediksi</th><th>GRU Δ%</th></tr>'
                f'{rows}'
                '</table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")

    # ══════════════════════════════════════════
    # TAB 4: PERBANDINGAN KINERJA
    # ══════════════════════════════════════════
    with tab_compare:
        st.markdown('<div class="section-header">⚖️ Perbandingan Kinerja LSTM vs GRU</div>', unsafe_allow_html=True)

        rows_data = []
        for ticker in selected_stocks:
            if ticker not in all_results:
                continue
            meta = STOCKS_META[ticker]
            res  = all_results[ticker]
            for mtype in ["LSTM", "GRU"]:
                rows_data.append({
                    "Saham"   : meta["short"],
                    "Model"   : mtype,
                    "MAE"     : res[mtype]["mae"],
                    "MSE"     : res[mtype]["mse"],
                    "MAPE(%)" : res[mtype]["mape"],
                })

        df_cmp = pd.DataFrame(rows_data)

        if df_cmp.empty or "Model" not in df_cmp.columns:
            st.warning("Tidak ada data. Jalankan model terlebih dahulu.")
        else:
            # Bar chart
            fig_bar = make_subplots(rows=1, cols=3,
                subplot_titles=[
                    "MAE (lebih kecil = lebih baik)",
                    "MSE (lebih kecil = lebih baik)",
                    "MAPE % (lebih kecil = lebih baik)"
                ])

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
                xaxis=dict(gridcolor="#1e3a5f"),  yaxis=dict(gridcolor="#1e3a5f"),
                xaxis2=dict(gridcolor="#1e3a5f"), yaxis2=dict(gridcolor="#1e3a5f"),
                xaxis3=dict(gridcolor="#1e3a5f"), yaxis3=dict(gridcolor="#1e3a5f"),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            # Download Data perbandingan LSTM_Prediksi dan GRU Prediksi
            all_export = []
            for ticker in selected_stocks:
                if ticker not in all_results:
                    continue
                meta = STOCKS_META[ticker]
                res  = all_results[ticker]
                dates = res["LSTM"]["test_dates"][:len(res["LSTM"]["actuals"])]
                
                for j, d in enumerate(dates):
                    all_export.append({
                        "Saham"         : meta["short"],
                        "Tanggal"       : d,
                        "Aktual"        : res["LSTM"]["actuals"][j],
                        "LSTM_Prediksi" : res["LSTM"]["predictions"][j],
                        "GRU_Prediksi"  : res["GRU"]["predictions"][j],
                    })

            df_all = pd.DataFrame(all_export)
            csv_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Semua Hasil Prediksi (.csv)",
                data=csv_all,
                file_name="semua_prediksi_saham.csv",
                mime="text/csv",
                use_container_width=True
            )
            # Radar chart
            st.markdown('<div class="section-header">🕸 Radar Chart Perbandingan</div>', unsafe_allow_html=True)
            for ticker in selected_stocks:
                if ticker not in all_results:
                    continue
                meta = STOCKS_META[ticker]
                res  = all_results[ticker]

                categories   = ["MAE", "MSE", "MAPE"]
                fig_radar    = go.Figure()
                radar_colors = {
                    "LSTM": ("#42a5f5", "rgba(66,165,245,0.15)"),
                    "GRU" : ("#66bb6a", "rgba(102,187,106,0.15)")
                }
                for mtype in ["LSTM", "GRU"]:
                    line_color, fill_color = radar_colors[mtype]
                    vals = [res[mtype]["mae"], res[mtype]["mse"], res[mtype]["mape"]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories + [categories[0]],
                        name=mtype, fill="toself",
                        line=dict(color=line_color),
                        fillcolor=fill_color
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(15,32,64,0.4)",
                        radialaxis=dict(visible=True, color="#90caf9"),
                        angularaxis=dict(color="#90caf9")
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    title=dict(text=f"{meta['short']} – {meta['name']}", font=dict(size=15)),
                    showlegend=True,
                    legend=dict(bgcolor="rgba(13,21,38,0.8)"),
                    height=380,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Winner Summary
            st.markdown('<div class="section-header">🏆 Kesimpulan: Model Terbaik per Saham</div>', unsafe_allow_html=True)
            win_cols     = st.columns(len(selected_stocks))
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
            other          = "GRU" if overall_winner == "LSTM" else "LSTM"
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
                ({overall_wins[other]} kali {other} lebih unggul)
              </div>
              <div style="color:#a5d6a7;margin-top:12px;font-size:14px">
                Berdasarkan MAE + MAPE terkecil sebagai kriteria utama evaluasi
              </div>
            </div>
            """, unsafe_allow_html=True)

else:
    with tab_model:
        st.info("👆 Klik tombol **🚀 Jalankan Model** di sidebar untuk memulai pelatihan LSTM & GRU.")
    with tab_forecast:
        st.info("👆 Klik tombol **🚀 Jalankan Model** di sidebar untuk memulai pelatihan LSTM & GRU.")
    with tab_compare:
        st.info("👆 Klik tombol **🚀 Jalankan Model** di sidebar untuk memulai pelatihan LSTM & GRU.")