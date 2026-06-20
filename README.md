# 📈 StockSight – LSTM vs GRU Stock Prediction

## File Struktur
```
├── model.py          ← Pipeline ML (standalone, bisa dijalankan sendiri)
├── app.py            ← Aplikasi Streamlit (dashboard utama)
├── requirements.txt  ← Dependensi Python
└── README.md
```

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

### 1. Hanya Model (tanpa UI)
```bash
python model.py
```
Output: Hasil MAE, MSE, MAPE + Prediksi 3 hari ke depan di terminal.

### 2. Aplikasi Streamlit (dengan Dashboard)
```bash
streamlit run app.py
```
Buka browser di: http://localhost:8501

## 🧠 Fitur Dashboard
| Tab | Fitur |
|-----|-------|
| 📊 Overview Harga | Grafik histori harga 2015–2025 semua saham |
| 🧠 Hasil Model | Metrik MAE/MSE/MAPE, grafik Aktual vs Prediksi, training loss |
| 🔮 Forecasting | Prediksi 3 hari ke depan LSTM & GRU |
| ⚖️ Perbandingan | Bar chart, radar chart, kesimpulan model terbaik |

## 📌 Detail Model

| Parameter | Nilai |
|-----------|-------|
| Timesteps (lookback) | 30 hari |
| Train/Test Split | 80% / 20% |
| Epochs | 50 (+ EarlyStopping) |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.0001, clipnorm=1.0) |
| Arsitektur | 2 layer (LSTM/GRU 50 units) + Dropout 0.2 |

## 📊 Saham yang Dianalisis
- **BBRI** – Bank Rakyat Indonesia
- **BMRI** – Bank Mandiri
- **BBTN** – Bank Tabungan Negara
- **BBNI** – Bank Negara Indonesia

## 📏 Metrik Evaluasi
- **MAE** – Mean Absolute Error (rata-rata kesalahan absolut)
- **MSE** – Mean Squared Error (rata-rata kuadrat kesalahan)
- **MAPE** – Mean Absolute Percentage Error (% kesalahan rata-rata)

Semakin kecil nilai → semakin akurat model.
