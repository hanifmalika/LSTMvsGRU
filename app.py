"""
app.py  -  Streamlit Dashboard: LSTM vs GRU Stock Prediction
Saham: BBRI, BMRI, BBTN, BBNI  |  Metrik: MAE, MSE, MAPE

SUMBER DATA (dua pilihan, dipilih lewat sidebar):
1. Dataset Yahoo Finance otomatis, rentang 2015-01-01 s/d 2025-12-31,
   untuk BBRI/BMRI/BBTN/BBNI (tombol "Ambil Data Yahoo Finance").
2. Input MANUAL lewat upload file Excel (satu file, boleh berisi
   beberapa sheet - satu sheet per saham). Tersedia juga tombol download
   TEMPLATE Excel dan link untuk melihat contoh formatnya.

Kedua sumber divalidasi & dinormalisasi lewat model.py (validate_and_clean(),
normalize_series()) sebelum dipakai untuk training LSTM/GRU.
"""

import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings("ignore")

import model  # <-- semua logic load-excel, validasi, normalisasi, LSTM/GRU ada di sini

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

html, body, [class*="css"]{ font-family: 'Inter', sans-serif; }
.stApp{ background-color:#0d1117; color:#f0f6fc; }

[data-testid="stSidebar"]{ background-color:#161b22 !important; border-right:1px solid #30363d; }
[data-testid="stSidebar"] *{ color:#f0f6fc !important; }
[data-testid="stSidebar"] label{ color:#58a6ff !important; font-weight:600; }

.hero-title{ font-size:46px; font-weight:800; color:#58a6ff; margin-bottom:6px; }
.hero-sub{ color:#8b949e; font-size:16px; }

.section-header{
    background:#1f6feb; color:white; padding:14px 20px; border-radius:10px;
    font-size:18px; font-weight:700; margin:20px 0px;
}

.metric-card{
    background:#161b22; border:1px solid #30363d; border-radius:14px;
    padding:20px; margin-bottom:15px; box-shadow:0 3px 12px rgba(0,0,0,.25); transition:.2s;
}
.metric-card:hover{ border-color:#58a6ff; transform:translateY(-3px); }
.metric-label{ color:#8b949e; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
.metric-value{ font-family:'JetBrains Mono', monospace; color:#ffffff; font-size:30px; font-weight:700; margin-top:8px; }
.metric-sub{ color:#58a6ff; font-size:13px; margin-top:6px; }

.badge-winner{ display:inline-block; padding:5px 14px; border-radius:20px; background:#238636; color:white; font-size:12px; font-weight:700; }
.badge-runner{ display:inline-block; padding:5px 14px; border-radius:20px; background:#1f6feb; color:white; font-size:12px; font-weight:700; }

.future-table{ width:100%; border-collapse:collapse; }
.future-table th{ background:#1f6feb; color:white; padding:12px; font-weight:600; }
.future-table td{ background:#161b22; color:#f0f6fc; padding:10px; border-bottom:1px solid #30363d; }
.future-table tr:hover td{ background:#21262d; }

.stButton>button{ width:100%; background:#1f6feb; color:white; border:none; border-radius:10px; padding:10px; font-weight:600; }
.stButton>button:hover{ background:#388bfd; }
.stDownloadButton > button { background: #1f6feb; color: white; border: none; border-radius: 10px; padding: 10px 20px; font-weight: 600; transition: 0.3s; }
.stDownloadButton > button:hover { background: #388bfd; color: white; }

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"],
.stDateInput input, .stTextArea textarea{ background:#21262d !important; color:white !important; border:1px solid #30363d !important; }
.stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label, .stTextArea label{ color:#f0f6fc !important; }

[data-testid="stDataFrame"]{ border:1px solid #30363d; border-radius:10px; }
[data-testid="metric-container"]{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:15px; }
[data-testid="metric-container"] *{ color:white !important; }
.streamlit-expanderHeader{ color:white !important; }
p, li, span{ color:#f0f6fc; }
hr{ border-color:#30363d; }

.upload-box{
    background:#161b22; border:1px dashed #30363d; border-radius:12px;
    padding:16px; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# META WARNA/NAMA UNTUK SAHAM YANG SUDAH DIKENAL
# (dipakai kalau nama sheet cocok, kalau tidak akan pakai default)
# ─────────────────────────────────────────────
KNOWN_STOCKS_META = {
    "BBRI": {"name": "Bank Rakyat Indonesia", "color": "#42a5f5"},
    "BMRI": {"name": "Bank Mandiri",          "color": "#66bb6a"},
    "BBTN": {"name": "Bank Tabungan Negara",  "color": "#ffa726"},
    "BBNI": {"name": "Bank Negara Indonesia", "color": "#ef5350"},
}
DEFAULT_COLORS = ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc", "#26c6da"]

TIMESTEPS   = 30
EPOCHS      = 50
FUTURE_DAYS = 3


def build_meta(sheet_names):
    """Bangun metadata (nama tampilan + warna) untuk tiap sheet yang diupload."""
    meta = {}
    for i, name in enumerate(sheet_names):
        key = name.strip().upper()
        if key in KNOWN_STOCKS_META:
            meta[name] = {"short": key, **KNOWN_STOCKS_META[key]}
        else:
            meta[name] = {
                "short": name,
                "name": name,
                "color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            }
    return meta


def make_template_excel() -> bytes:
    """Buat file Excel template (contoh format) untuk diunduh user."""
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    rng = np.random.default_rng(42)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for i, sheet in enumerate(["BBRI", "BMRI", "BBTN", "BBNI"]):
            base = 3000 + i * 1500
            harga = base + np.cumsum(rng.normal(0, 20, size=len(dates)))
            df_tmpl = pd.DataFrame({
                "Tanggal": dates.strftime("%Y-%m-%d"),
                "Close": harga.round(0)
            })
            df_tmpl.to_excel(writer, sheet_name=sheet, index=False)
    buffer.seek(0)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR: SUMBER DATA (Yahoo Finance otomatis ATAU Excel manual)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📥 Sumber Data")
    st.markdown("---")

    data_source = st.radio(
        "Pilih sumber data",
        options=["📊 Dataset Yahoo Finance (2015-2025)", "📁 Upload Excel Manual"],
        key="data_source_choice"
    )

    # ── OPSI 1: DATASET YAHOO FINANCE (otomatis, 2015-2025) ──
    if data_source.startswith("📊"):
        st.caption(
            f"Data historis otomatis untuk **{', '.join(t.replace('.JK','') for t in model.STOCKS)}** "
            f"dari **{model.START_DATE}** s/d **{model.END_DATE}** via Yahoo Finance."
        )

        yahoo_btn = st.button("🚀 Ambil Data Yahoo Finance", use_container_width=True)

        if yahoo_btn:
            with st.spinner("Mengambil data dari Yahoo Finance..."):
                try:
                    stock_data, load_errors = model.load_yahoo_multistock(
                        model.STOCKS, model.START_DATE, model.END_DATE
                    )
                    st.session_state["stock_data"] = stock_data
                    st.session_state["stocks_meta"] = build_meta(list(stock_data.keys()))
                    st.session_state["data_source_label"] = "Yahoo Finance (2015-2025)"

                    if load_errors:
                        st.warning(
                            "⚠️ Sebagian saham gagal diambil:\n\n" +
                            "\n".join([f"- {k}: {v}" for k, v in load_errors.items()])
                        )
                    st.success(f"✅ {len(stock_data)} saham berhasil diambil dari Yahoo Finance.")
                except ValueError as e:
                    st.error(
                        f"❌ Gagal mengambil data dari Yahoo Finance:\n\n{e}\n\n"
                        f"Silakan coba mode **📁 Upload Excel Manual** sebagai alternatif."
                    )
                except Exception as e:
                    st.error(
                        f"❌ Terjadi kesalahan saat mengambil data: {e}\n\n"
                        f"Silakan coba mode **📁 Upload Excel Manual** sebagai alternatif."
                    )

    # ── OPSI 2: UPLOAD EXCEL MANUAL ──
    else:
        st.download_button(
            label="⬇️ Download Template Excel",
            data=make_template_excel(),
            file_name="template_data_saham.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        with st.expander("🔗 Lihat contoh format Excel"):
            st.caption(
                "Setiap SHEET = 1 saham (nama sheet bebas, misal BBRI/BMRI/BBTN/BBNI). "
                "Kolom wajib: **Tanggal** dan **Close** (atau 'Adj Close' / 'Harga')."
            )
            contoh_df = pd.DataFrame({
                "Tanggal": pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d"),
                "Close": [4520, 4535, 4510, 4560, 4575],
            })
            st.dataframe(contoh_df, use_container_width=True, hide_index=True)
            st.caption("Contoh di atas untuk 1 sheet (misal sheet bernama 'BBRI'). Ulangi pola yang sama di sheet lain.")

        uploaded_file = st.file_uploader(
            "Upload File Excel (.xlsx)",
            type=["xlsx"],
            help="Satu file, boleh berisi beberapa sheet (satu sheet per saham)."
        )

        if uploaded_file is not None:
            try:
                stock_data = model.load_excel_multisheet(uploaded_file)
                st.session_state["stock_data"] = stock_data
                st.session_state["stocks_meta"] = build_meta(list(stock_data.keys()))
                st.session_state["data_source_label"] = "Upload Excel Manual"
                st.success(f"✅ {len(stock_data)} sheet berhasil dimuat & divalidasi.")
            except ValueError as e:
                st.error(f"❌ Gagal memproses file:\n\n{e}")

    st.markdown("---")

    if "stock_data" in st.session_state:
        all_sheet_names = list(st.session_state["stock_data"].keys())
        stocks_meta = st.session_state["stocks_meta"]

        selected_stocks = st.multiselect(
            "Pilih Saham",
            options=all_sheet_names,
            default=all_sheet_names,
            format_func=lambda x: f"{stocks_meta[x]['short']} - {stocks_meta[x]['name']}"
        )

        st.markdown("---")
        st.markdown("### 🔧 Parameter Model")
        epochs_ui    = st.slider("Epochs",        10, 100, EPOCHS,      5)
        timesteps_ui = st.slider("Timesteps",     10, 60,  TIMESTEPS,   5)
        future_ui    = st.slider("Hari Prediksi",  1,  7,  FUTURE_DAYS)

        st.markdown("---")
        run_btn = st.button("🚀 Jalankan Model", use_container_width=True)
    else:
        selected_stocks = []
        run_btn = False
        epochs_ui, timesteps_ui, future_ui = EPOCHS, TIMESTEPS, FUTURE_DAYS

    st.markdown("---")
    st.caption(f"Data: {st.session_state.get('data_source_label', 'Belum dipilih')} | Model: TensorFlow/Keras")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">StockSight</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Prediksi Harga Saham Bank Indonesia | LSTM vs GRU | Evaluasi MAE, MSE, MAPE</div>', unsafe_allow_html=True)
st.markdown("---")

if "stock_data" not in st.session_state:
    st.info(
        "👋 Mulai dengan memilih **sumber data** di sidebar:\n\n"
        "- Klik **🚀 Ambil Data Yahoo Finance** untuk memakai dataset otomatis 2015-2025, atau\n"
        "- Pilih **📁 Upload Excel Manual** untuk memasukkan data sendiri "
        "(download **Template Excel** atau lihat **contoh formatnya** dulu di sidebar)."
    )
    st.stop()

if not selected_stocks:
    st.warning("⚠️ Pilih minimal satu saham di sidebar.")
    st.stop()

stocks_meta = st.session_state["stocks_meta"]
stock_data  = st.session_state["stock_data"]

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
    st.markdown('<div class="section-header">📈 Histori Harga</div>', unsafe_allow_html=True)

    fig = go.Figure()
    for ticker in selected_stocks:
        meta = stocks_meta[ticker]
        df = stock_data[ticker]
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
        meta = stocks_meta[ticker]
        df   = stock_data[ticker]
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
                Jumlah data: {len(df)} baris<br>
                Min: Rp{df['ha'].min():,.0f} | Max: Rp{df['ha'].max():,.0f}<br>
                Return: {"+" if pct>=0 else ""}{pct:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🔍 Lihat data mentah (setelah validasi & pembersihan)"):
        preview_ticker = st.selectbox("Pilih saham untuk preview", selected_stocks, key="preview_sel")
        st.dataframe(stock_data[preview_ticker], use_container_width=True)


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
            meta = stocks_meta[ticker]

            def update_progress(mtype, _meta=meta):
                step[0] += 1
                progress_bar.progress(
                    step[0] / total,
                    text=f"⚙️ Melatih {mtype} untuk {_meta['short']}... ({step[0]}/{total})"
                )

            all_results[ticker] = model.train_and_evaluate(
                stock_data[ticker],
                label=ticker,
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
            meta = stocks_meta[ticker]
            res  = all_results[ticker]

            st.markdown(f'<div class="section-header">🏦 {meta["short"]} - {meta["name"]}</div>', unsafe_allow_html=True)

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
            meta       = stocks_meta[ticker]
            res        = all_results[ticker]
            df         = stock_data[ticker]
            last_price = float(df["ha"].iloc[-1])

            st.markdown(f"#### 🏦 {meta['short']} - {meta['name']}")

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
                type="line", x0=today_str, x1=today_str, y0=0, y1=1,
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
            meta = stocks_meta[ticker]
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

            all_export = []
            for ticker in selected_stocks:
                if ticker not in all_results:
                    continue
                meta = stocks_meta[ticker]
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

            st.markdown('<div class="section-header">🕸 Radar Chart Perbandingan</div>', unsafe_allow_html=True)
            for ticker in selected_stocks:
                if ticker not in all_results:
                    continue
                meta = stocks_meta[ticker]
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
                    title=dict(text=f"{meta['short']} - {meta['name']}", font=dict(size=15)),
                    showlegend=True,
                    legend=dict(bgcolor="rgba(13,21,38,0.8)"),
                    height=380,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown('<div class="section-header">🏆 Kesimpulan: Model Terbaik per Saham</div>', unsafe_allow_html=True)
            win_cols     = st.columns(len(selected_stocks))
            overall_wins = {"LSTM": 0, "GRU": 0}

            for i, ticker in enumerate(selected_stocks):
                if ticker not in all_results:
                    continue
                meta = stocks_meta[ticker]
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
