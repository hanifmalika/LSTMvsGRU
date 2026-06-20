import yfinance as yf

df = yf.download(
    "BBRI.JK",
    start="2015-01-01",
    end="2025-12-31",
    progress=False,
    auto_adjust=True
)

print(df.head())