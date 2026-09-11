import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Deep Learning (opsional, dibungkus try/except agar app tetap jalan
# walau TensorFlow belum terinstal)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Analisis Saham & IHSG Realtime",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. AUTO-REFRESH & SIDEBAR CONFIG
# ==========================================
st.sidebar.title("⚙️ Pengaturan Dashboard")

refresh_interval = st.sidebar.slider(
    "Interval Auto-Refresh (Detik):",
    min_value=5,
    max_value=60,
    value=10,
    step=5
)

count = st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

ticker_options = {
    "IHSG (^JKSE)": "^JKSE",
    "BBCA (Bank Central Asia)": "BBCA.JK",
    "BBRI (Bank Rakyat Indonesia)": "BBRI.JK",
    "BMRI (Bank Mandiri)": "BMRI.JK",
    "TLKM (Telkom Indonesia)": "TLKM.JK",
    "ASII (Astra International)": "ASII.JK"
}

selected_label = st.sidebar.selectbox("Pilih Saham / Indeks:", list(ticker_options.keys()))
selected_ticker = ticker_options[selected_label]

time_range = st.sidebar.selectbox(
    "Rentang Waktu (tampilan grafik):",
    options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=0
)

interval_map = {
    "1d": "1m",
    "5d": "5m",
    "1mo": "1d",
    "3mo": "1d",
    "6mo": "1d",
    "1y": "1wk"
}
selected_interval = interval_map.get(time_range, "1m")

custom_ticker = st.sidebar.text_input("Atau Ketik Ticker Lain (contoh: UNVR.JK):", value="")
if custom_ticker.strip():
    selected_ticker = custom_ticker.strip().upper()
    selected_label = selected_ticker

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Pengaturan Model AI")
enable_ml = st.sidebar.checkbox("Aktifkan Machine Learning (Random Forest)", value=True)
enable_dl = st.sidebar.checkbox(
    "Aktifkan Deep Learning (LSTM)" + ("" if TF_AVAILABLE else " — TensorFlow tidak terpasang"),
    value=False,
    disabled=not TF_AVAILABLE
)
ml_lookback_days = st.sidebar.slider("Data latih untuk scalping (hari, interval 1m):", 1, 7, 5)

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Disclaimer risiko**: model statistik/ML/DL di bawah ini memberi *probabilitas*, "
    "bukan kepastian. Pergerakan harga jangka sangat pendek (scalping) sangat bising (noisy) "
    "dan sebagian besar acak — tidak ada model yang bisa 100% akurat. Gunakan sebagai salah satu "
    "input, bukan satu-satunya dasar keputusan trading, dan selalu terapkan manajemen risiko "
    "(stop loss)."
)

# ==========================================
# 3. FUNGSI AMBIL DATA
# ==========================================
@st.cache_data(ttl=refresh_interval)
def fetch_stock_data(ticker, period="1d", interval="1m"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # fundamental hanya berubah berkala (bukan per detik)
def fetch_fundamental_data(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception as e:
        return {"_error": str(e)}


# ==========================================
# 4. ANALISIS TEKNIKAL
# ==========================================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # EMA
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands (20, 2 std)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std

    # Stochastic Oscillator (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ATR (14) - volatilitas, berguna untuk sizing stop-loss scalping
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # VWAP (kumulatif per sesi data yang tersedia)
    df["VWAP"] = (close * vol).cumsum() / vol.cumsum().replace(0, np.nan)

    return df


def generate_technical_signal(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 21:
        return {"label": "DATA KURANG", "score": 0, "detail": []}

    last = df.iloc[-1]
    score = 0
    detail = []

    if last["EMA9"] > last["EMA21"]:
        score += 1
        detail.append("EMA9 > EMA21 → tren jangka pendek naik")
    else:
        score -= 1
        detail.append("EMA9 < EMA21 → tren jangka pendek turun")

    if pd.notna(last["RSI14"]):
        if last["RSI14"] < 30:
            score += 1
            detail.append(f"RSI {last['RSI14']:.1f} → oversold")
        elif last["RSI14"] > 70:
            score -= 1
            detail.append(f"RSI {last['RSI14']:.1f} → overbought")
        else:
            detail.append(f"RSI {last['RSI14']:.1f} → netral")

    if last["MACD"] > last["MACD_signal"]:
        score += 1
        detail.append("MACD di atas signal line → momentum naik")
    else:
        score -= 1
        detail.append("MACD di bawah signal line → momentum turun")

    if pd.notna(last["BB_lower"]) and last["Close"] < last["BB_lower"]:
        score += 1
        detail.append("Harga di bawah Bollinger Band bawah → potensi rebound")
    elif pd.notna(last["BB_upper"]) and last["Close"] > last["BB_upper"]:
        score -= 1
        detail.append("Harga di atas Bollinger Band atas → potensi koreksi")

    if pd.notna(last["Stoch_K"]):
        if last["Stoch_K"] < 20:
            score += 1
            detail.append(f"Stochastic {last['Stoch_K']:.1f} → oversold")
        elif last["Stoch_K"] > 80:
            score -= 1
            detail.append(f"Stochastic {last['Stoch_K']:.1f} → overbought")

    if score >= 3:
        label = "STRONG BUY"
    elif score >= 1:
        label = "BUY"
    elif score <= -3:
        label = "STRONG SELL"
    elif score <= -1:
        label = "SELL"
    else:
        label = "NETRAL"

    return {"label": label, "score": score, "detail": detail}


# ==========================================
# 5. MACHINE LEARNING (Random Forest) - untuk scalping jangka sangat pendek
# ==========================================
@st.cache_resource(ttl=60)
def train_ml_model(ticker: str, df_hash: int, df: pd.DataFrame):
    data = calculate_technical_indicators(df).dropna().copy()
    if len(data) < 60:
        return None

    data["Return"] = data["Close"].pct_change()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    features = ["EMA9", "EMA21", "RSI14", "MACD", "MACD_hist",
                "Stoch_K", "Stoch_D", "ATR14", "Return"]
    X = data[features]
    y = data["Target"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) > 0 else np.nan

    latest_features = X.iloc[[-1]]
    proba_up = model.predict_proba(latest_features)[0][1]

    return {
        "model": model,
        "test_accuracy": acc,
        "proba_up": proba_up,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features,
    }


# ==========================================
# 6. DEEP LEARNING (LSTM) - opsional, prediksi harga close berikutnya
# ==========================================
@st.cache_resource(ttl=300)
def train_lstm_model(ticker: str, df_hash: int, closes: np.ndarray, window: int = 20):
    if not TF_AVAILABLE:
        return None
    if len(closes) < window + 30:
        return None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(closes.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=8, batch_size=16, verbose=0)

    if len(X_test) > 0:
        test_pred = model.predict(X_test, verbose=0)
        mae_scaled = float(np.mean(np.abs(test_pred.flatten() - y_test)))
    else:
        mae_scaled = np.nan

    last_window = scaled[-window:].reshape(1, window, 1)
    next_scaled = model.predict(last_window, verbose=0)[0][0]
    next_price = scaler.inverse_transform([[next_scaled]])[0][0]

    return {"next_price_pred": next_price, "mae_scaled": mae_scaled}


# ==========================================
# 7. TAMPILAN UTAMA
# ==========================================
st.title("📊 Dashboard Analisis Saham & IHSG — Teknikal, Fundamental & AI Scalping")

wib_tz = pytz.timezone("Asia/Jakarta")
waktu_sekarang = datetime.now(wib_tz).strftime("%Y-%m-%d %H:%M:%S WIB")
st.markdown(
    f"**Aset Terpilih:** `{selected_label}` | 🕒 **Waktu Terkini:** `{waktu_sekarang}` | "
    f"🔄 *Refresh Ke-#{count}*"
)

df = fetch_stock_data(selected_ticker, period=time_range, interval=selected_interval)

if df.empty or len(df) < 1:
    st.warning("Data tidak ditemukan atau pasar sedang tutup. Coba rentang waktu lebih luas (5d / 1mo).")
    st.stop()

df_ti = calculate_technical_indicators(df)

# --- METRIK UTAMA ---
harga_terakhir = df["Close"].iloc[-1]
harga_sebelumnya = df["Close"].iloc[-2] if len(df) >= 2 else df["Open"].iloc[0]
perubahan = harga_terakhir - harga_sebelumnya
persen_perubahan = (perubahan / harga_sebelumnya) * 100 if harga_sebelumnya != 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Terakhir", f"{harga_terakhir:,.2f}", f"{perubahan:+,.2f} ({persen_perubahan:+.2f}%)")
col2.metric("Harga Tertinggi", f"{df['High'].max():,.2f}")
col3.metric("Harga Terendah", f"{df['Low'].min():,.2f}")
col4.metric("Total Volume", f"{df['Volume'].sum():,.0f}")

st.markdown("---")

tab_chart, tab_teknikal, tab_fundamental, tab_ai = st.tabs(
    ["📈 Grafik Harga", "🔬 Analisis Teknikal", "🏦 Analisis Fundamental", "🤖 ML & DL — Scalping"]
)

# --- TAB 1: GRAFIK ---
with tab_chart:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
    ))
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["EMA9"], mode="lines",
                                  name="EMA 9", line=dict(color="orange", width=1.3)))
        fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["EMA21"], mode="lines",
                                  name="EMA 21", line=dict(color="blue", width=1.3)))
        fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["BB_upper"], mode="lines",
                                  name="BB Atas", line=dict(color="gray", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["BB_lower"], mode="lines",
                                  name="BB Bawah", line=dict(color="gray", width=1, dash="dot")))

    fig.update_layout(
        xaxis_rangeslider_visible=False, height=520,
        margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", yaxis_title="Harga (IDR)"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Lihat Data Histori Terakhir"):
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

# --- TAB 2: TEKNIKAL ---
with tab_teknikal:
    signal = generate_technical_signal(df_ti)

    color_map = {
        "STRONG BUY": "🟢🟢", "BUY": "🟢", "NETRAL": "🟡",
        "SELL": "🔴", "STRONG SELL": "🔴🔴", "DATA KURANG": "⚪"
    }
    st.subheader(f"Sinyal Gabungan: {color_map.get(signal['label'], '')} **{signal['label']}** (skor {signal['score']})")
    for d in signal["detail"]:
        st.write(f"- {d}")

    st.markdown("#### Indikator Momentum & Volatilitas")
    ind_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                             vertical_spacing=0.04,
                             subplot_titles=("RSI (14)", "MACD", "Stochastic Oscillator"))
    ind_fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["RSI14"], name="RSI14"), row=1, col=1)
    ind_fig.add_hline(y=70, line_dash="dot", line_color="red", row=1, col=1)
    ind_fig.add_hline(y=30, line_dash="dot", line_color="green", row=1, col=1)

    ind_fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["MACD"], name="MACD"), row=2, col=1)
    ind_fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["MACD_signal"], name="Signal"), row=2, col=1)
    ind_fig.add_trace(go.Bar(x=df_ti.index, y=df_ti["MACD_hist"], name="Histogram"), row=2, col=1)

    ind_fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["Stoch_K"], name="%K"), row=3, col=1)
    ind_fig.add_trace(go.Scatter(x=df_ti.index, y=df_ti["Stoch_D"], name="%D"), row=3, col=1)
    ind_fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
    ind_fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

    ind_fig.update_layout(height=650, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(ind_fig, use_container_width=True)

    st.markdown("#### ATR (Average True Range) — acuan jarak stop-loss")
    st.line_chart(df_ti["ATR14"])

# --- TAB 3: FUNDAMENTAL ---
with tab_fundamental:
    st.caption(
        "Data fundamental diperbarui berkala oleh penyedia data (biasanya harian/kuartalan), "
        "bukan per detik — berbeda dari harga yang bergerak real-time."
    )
    info = fetch_fundamental_data(selected_ticker)

    if not info or "_error" in info:
        st.info("Data fundamental tidak tersedia untuk ticker ini (umum terjadi pada indeks seperti ^JKSE).")
    else:
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("PER (Trailing)", f"{info.get('trailingPE', 'N/A')}")
        fcol2.metric("PBV", f"{info.get('priceToBook', 'N/A')}")
        fcol3.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}")
        fcol4.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A')}")

        fcol5, fcol6, fcol7, fcol8 = st.columns(4)
        fcol5.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,}" if isinstance(info.get('marketCap'), (int, float)) else "N/A")
        fcol6.metric("EPS (Trailing)", f"{info.get('trailingEps', 'N/A')}")
        fcol7.metric("Beta", f"{info.get('beta', 'N/A')}")
        fcol8.metric("Profit Margin", f"{info.get('profitMargins', 'N/A')}")

        st.write(f"**Sektor:** {info.get('sector', 'N/A')} | **Industri:** {info.get('industry', 'N/A')}")
        st.write(f"**52-Week High/Low:** {info.get('fiftyTwoWeekHigh', 'N/A')} / {info.get('fiftyTwoWeekLow', 'N/A')}")

        with st.expander("Ringkasan Bisnis"):
            st.write(info.get("longBusinessSummary", "Tidak ada ringkasan tersedia."))

# --- TAB 4: ML & DL SCALPING ---
with tab_ai:
    st.subheader("🤖 Prediksi Arah Harga Jangka Sangat Pendek (Scalping)")
    st.caption(
        "Model dilatih ulang secara periodik memakai data candle 1 menit terbaru. "
        "Output berupa PROBABILITAS, bukan jaminan — pasar bisa berubah arah kapan saja."
    )

    df_scalp = fetch_stock_data(selected_ticker, period=f"{ml_lookback_days}d", interval="1m")

    if df_scalp.empty or len(df_scalp) < 90:
        st.warning(
            "Data 1 menit tidak cukup (bursa mungkin tutup atau ticker tidak mendukung interval 1m "
            "untuk rentang ini). Coba lagi saat jam bursa aktif, atau naikkan 'Data latih' di sidebar."
        )
    else:
        df_hash = int(pd.util.hash_pandas_object(df_scalp["Close"]).sum())

        ml_col, dl_col = st.columns(2)

        with ml_col:
            st.markdown("##### 🌲 Random Forest (Machine Learning)")
            if enable_ml:
                result = train_ml_model(selected_ticker, df_hash, df_scalp)
                if result is None:
                    st.info("Data belum cukup untuk melatih model ML.")
                else:
                    proba_up = result["proba_up"]
                    arah = "NAIK ⬆️" if proba_up >= 0.5 else "TURUN ⬇️"
                    st.metric("Prediksi Candle Berikutnya", arah, f"{proba_up*100:.1f}% keyakinan naik")
                    acc_txt = f"{result['test_accuracy']*100:.1f}%" if pd.notna(result["test_accuracy"]) else "N/A"
                    st.write(f"Akurasi pada data uji (out-of-sample): **{acc_txt}** "
                             f"(latih: {result['n_train']} candle, uji: {result['n_test']} candle)")
                    st.caption(
                        "Akurasi ~50-55% pada skala menit sudah dianggap wajar untuk data pasar yang efisien; "
                        "waspadai overfitting bila akurasi tampak jauh lebih tinggi."
                    )
            else:
                st.info("Aktifkan 'Machine Learning' di sidebar untuk melihat prediksi.")

        with dl_col:
            st.markdown("##### 🧠 LSTM (Deep Learning)")
            if not TF_AVAILABLE:
                st.info(
                    "TensorFlow belum terpasang di environment ini. Jalankan "
                    "`pip install tensorflow` lalu restart aplikasi untuk mengaktifkan model LSTM."
                )
            elif enable_dl:
                with st.spinner("Melatih model LSTM (beberapa epoch, mohon tunggu)..."):
                    dl_result = train_lstm_model(
                        selected_ticker, df_hash,
                        df_scalp["Close"].values.astype(float)
                    )
                if dl_result is None:
                    st.info("Data belum cukup untuk melatih model LSTM.")
                else:
                    pred_price = dl_result["next_price_pred"]
                    delta_pred = pred_price - harga_terakhir
                    st.metric("Prediksi Harga Candle Berikutnya", f"{pred_price:,.2f}",
                              f"{delta_pred:+,.2f}")
                    st.caption(
                        "Model LSTM ringan (2 layer, ~8 epoch) demi kecepatan real-time. "
                        "Untuk akurasi lebih tinggi, latih model lebih besar secara offline "
                        "dan muat bobotnya, bukan melatih ulang tiap refresh."
                    )
            else:
                st.info("Aktifkan 'Deep Learning (LSTM)' di sidebar untuk melihat prediksi.")

        st.markdown("---")
        st.markdown("#### 🎯 Ringkasan Sinyal Scalping (Teknikal + AI)")
        signal_scalp = generate_technical_signal(calculate_technical_indicators(df_scalp))
        st.write(f"- Sinyal teknikal 1 menit: **{signal_scalp['label']}** (skor {signal_scalp['score']})")
        if enable_ml and 'result' in dir() and result is not None:
            st.write(f"- Model ML memperkirakan probabilitas naik: **{result['proba_up']*100:.1f}%**")
        st.info(
            "Untuk scalping nyata: kombinasikan sinyal di atas dengan spread bid-ask, likuiditas, "
            "dan berita terkini secara manual. Tidak ada sistem — statistik, ML, maupun DL — yang "
            "dapat menjamin profit konsisten pada horizon waktu semenit."
        )

st.markdown("---")
st.caption(
    "Data harga & fundamental disediakan oleh Yahoo Finance (yfinance). Pembaruan otomatis aktif "
    "melalui streamlit-autorefresh. Model ML/DL bersifat edukatif dan bukan nasihat keuangan."
)
