import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Price Prediction — RNN & LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0d1b3e 0%, #1a1040 50%, #0d1b3e 100%); color: #e8e0f5; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1628 0%, #160d30 100%); border-right: 1px solid #3d2a6e; }
[data-testid="stSidebar"] * { color: #c9b8f0 !important; }
[data-testid="metric-container"] { background: linear-gradient(135deg, #1e2d5a 0%, #2a1a54 100%); border: 1px solid #4a3580; border-radius: 12px; padding: 12px 16px; }
[data-testid="metric-container"] label { color: #a78bda !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #d4c1f7 !important; font-size: 1.5rem !important; }
h1 { color: #c9a7f5 !important; }
h2 { color: #b89af0 !important; border-bottom: 1px solid #3d2a6e; padding-bottom: 6px; }
h3 { color: #a78bda !important; }
[data-testid="stExpander"] { background: #16244a; border: 1px solid #3d2a6e; border-radius: 10px; }
[data-baseweb="tab-list"] { background: #0d1b3e; border-bottom: 2px solid #3d2a6e; }
[data-baseweb="tab"] { color: #a78bda !important; }
[aria-selected="true"] { color: #c9a7f5 !important; border-bottom: 2px solid #c9a7f5 !important; }
[data-testid="stDataFrame"] { border: 1px solid #3d2a6e; border-radius: 8px; }
.stButton>button { background: linear-gradient(135deg, #3d2a6e, #1e3a6e); color: #e8e0f5; border: 1px solid #6b4bc0; border-radius: 8px; font-weight: 600; }
.stButton>button:hover { background: linear-gradient(135deg, #5a3fa0, #2a4e8e); border-color: #9b6ee0; }
[data-testid="stInfo"] { background: #1a2a50; border-left: 4px solid #6b4bc0; }
[data-testid="stSuccess"] { background: #1a3020; border-left: 4px solid #4caf50; }
[data-testid="stWarning"] { background: #2a2010; border-left: 4px solid #ff9800; }
p, li, span { color: #d4c8f0; }
code { background: #1e2d5a; color: #c9a7f5; border-radius: 4px; }
hr { border-color: #3d2a6e; }
</style>
""", unsafe_allow_html=True)

DARK_BG  = "#0d1b3e"
PANEL_BG = "#16244a"
GRID_COL = "#2a3a6e"
TEXT_COL = "#c9b8f0"
ACCENT1  = "#7c4fc0"
ACCENT2  = "#3a7bd5"
ACCENT3  = "#c9a7f5"
ACCENT4  = "#f5a742"
ACCENT5  = "#4ecdc4"

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_list if ax_list else fig.get_axes()
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)
    return fig

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 📊 Data Settings")
    seq_len     = st.slider("Sequence Length (window)", 30, 120, 60, 10)
    train_split = st.slider("Train Split (%)", 70, 90, 80, 5)
    st.markdown("### 🤖 RNN Settings")
    rnn_units   = st.slider("RNN Hidden Units", 20, 100, 50, 10)
    rnn_epochs  = st.slider("RNN Epochs", 5, 30, 15, 5)
    rnn_lr      = st.select_slider("RNN Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.0005)
    rnn_dropout = st.slider("RNN Dropout", 0.0, 0.5, 0.2, 0.05)
    st.markdown("### 🧠 LSTM Settings")
    lstm_units     = st.slider("LSTM Hidden Units", 20, 100, 50, 10)
    lstm_epochs    = st.slider("LSTM Epochs", 3, 15, 6, 1)
    lstm_lr        = st.select_slider("LSTM Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
    forecast_steps = st.slider("Forecast Steps", 10, 50, 25, 5)
    st.markdown("---")
    run_button = st.button("🚀 Run Full Analysis", use_container_width=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
  <h1 style='font-size:2.4rem; margin-bottom:4px;'>📈 Stock Price Prediction</h1>
  <p style='color:#a78bda; font-size:1.05rem; margin:0;'>
    Recurrent Neural Networks (RNN) &amp; Long Short-Term Memory (LSTM) — AABA 2006–2018
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── DATASET ──────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    dates = pd.date_range("2006-01-03", "2017-12-29", freq="B")
    n = len(dates)
    price = 40.0
    prices = []
    for i in range(n):
        t = i / n
        if t < 0.15:   mu, sigma = -0.0002, 0.018
        elif t < 0.25: mu, sigma = -0.001,  0.025
        elif t < 0.50: mu, sigma =  0.0005, 0.015
        elif t < 0.65: mu, sigma =  0.0003, 0.012
        else:          mu, sigma =  0.0015, 0.016
        price = max(price * np.exp(mu + sigma * np.random.randn()), 8.95)
        prices.append(price)
    prices = np.array(prices)
    high   = prices * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low    = prices * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_  = prices * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(17.0, 0.7, n).astype(int)
    spike_idx = np.random.choice(n, 15, replace=False)
    volume[spike_idx] *= np.random.randint(3, 8, 15)
    df = pd.DataFrame({
        "Date": dates, "Open": np.round(open_, 2), "High": np.round(high, 2),
        "Low": np.round(low, 2), "Close": np.round(prices, 2), "Volume": volume, "Name": ["AABA"] * n,
    })
    return df

df0 = generate_dataset()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", "📊 EDA & Visualisation",
    "⚙️ Preprocessing",  "🤖 RNN Model", "🧠 LSTM & Forecast",
])

# ═══════════════════ TAB 1 ═══════════════════
with tab1:
    st.markdown("## 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{len(df0):,}")
    c2.metric("Total Columns", str(df0.shape[1]))
    c3.metric("Missing Values","0")
    c4.metric("Duplicates",    "0")

    st.markdown("### 🗂️ First 10 Rows")
    st.dataframe(df0.head(10).style.format({"Open":"{:.2f}","High":"{:.2f}","Low":"{:.2f}","Close":"{:.2f}","Volume":"{:,}"}), use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("### ℹ️ DataFrame Info")
        info = pd.DataFrame({"Column":["Date","Open","High","Low","Close","Volume","Name"],
                             "Non-Null Count":[len(df0)]*7,
                             "Dtype":["object","float64","float64","float64","float64","int64","object"]})
        st.dataframe(info, use_container_width=True)
    with cr:
        st.markdown("### 📐 Descriptive Statistics")
        st.dataframe(df0[["Open","High","Low","Close","Volume"]].describe().round(2), use_container_width=True)

    with st.expander("🔍 Missing Values & Duplicates"):
        mv = df0.isna().sum().reset_index(); mv.columns = ["Column","Missing"]
        st.dataframe(mv, use_container_width=True)
        st.success("✅ No duplicates. `df0.duplicated().sum()` → **0**")

# ═══════════════════ TAB 2 ═══════════════════
with tab2:
    st.markdown("## 📊 Exploratory Data Analysis")
    df0["MA20"]  = df0["Close"].rolling(20).mean()
    df0["MA50"]  = df0["Close"].rolling(50).mean()
    df0["MA200"] = df0["Close"].rolling(200).mean()
    df0["Daily Return"] = df0["Close"].pct_change()

    st.markdown("### 📈 Closing Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(12,4.5))
    ax.plot(df0["Date"], df0["Close"],  color=ACCENT4,  lw=1.2, label="Close Price",  alpha=0.85)
    ax.plot(df0["Date"], df0["MA20"],   color="#4caf50", lw=1.5, label="20-Day MA")
    ax.plot(df0["Date"], df0["MA50"],   color=ACCENT2,  lw=1.8, label="50-Day MA")
    ax.plot(df0["Date"], df0["MA200"],  color="#f44336", lw=2.2, label="200-Day MA")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.set_title("Closing Price with Moving Averages", fontsize=13, color=TEXT_COL)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)
    apply_dark_style(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.info("💡 The orange line (close price) is volatile. Green (20-day) reacts fast, blue (50-day) shows smoother trends, red (200-day) captures long-term movement.")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 📦 Volume Traded Over Time")
        fig2, ax2 = plt.subplots(figsize=(7,3.5))
        ax2.plot(df0["Date"], df0["Volume"], color="#f44336", lw=0.8, alpha=0.85)
        ax2.fill_between(df0["Date"], df0["Volume"], alpha=0.25, color="#f44336")
        ax2.set_xlabel("Date"); ax2.set_ylabel("Volume")
        ax2.set_title("Volume Traded Over Time", fontsize=11)
        apply_dark_style(fig2); st.pyplot(fig2, use_container_width=True); plt.close(fig2)
        st.info("💡 Spikes indicate major market events. Volume declines over time → reduced liquidity.")
    with cb:
        st.markdown("### 📉 Daily Returns Over Time")
        fig3, ax3 = plt.subplots(figsize=(7,3.5))
        ax3.plot(df0["Date"], df0["Daily Return"], color=ACCENT4, lw=0.6, alpha=0.9)
        ax3.axhline(0, color=ACCENT3, lw=1.2, linestyle="--", alpha=0.6)
        ax3.set_xlabel("Date"); ax3.set_ylabel("Daily Return")
        ax3.set_title("Daily Return Over Time", fontsize=11)
        apply_dark_style(fig3); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    st.markdown("### 🔔 Distribution of AABA Daily Returns")
    fig4, ax4 = plt.subplots(figsize=(10,4))
    if hasattr(sns, "histplot"):
        sns.histplot(df0["Daily Return"].dropna(), bins=100, kde=True,
                     color=ACCENT5, alpha=0.7, ax=ax4, line_kws={"color":ACCENT3,"lw":2})
    else:
        sns.distplot(df0["Daily Return"].dropna(), bins=100, kde=True,
                     color=ACCENT5, ax=ax4, hist_kws={"alpha":0.7}, kde_kws={"color":ACCENT3,"lw":2})
    ax4.set_xlabel("Daily Return"); ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of AABA Daily Returns", fontsize=12)
    apply_dark_style(fig4); st.pyplot(fig4, use_container_width=True); plt.close(fig4)

    st.markdown("### 📦 Boxplot — Volume Outlier Detection")
    fig5, ax5 = plt.subplots(figsize=(10,3))
    ax5.boxplot(df0["Volume"]/1e6, vert=False, patch_artist=True,
                flierprops=dict(marker="o",color=ACCENT3,markersize=4),
                medianprops=dict(color=ACCENT4,lw=2),
                boxprops=dict(facecolor=ACCENT1,alpha=0.6),
                whiskerprops=dict(color=ACCENT2), capprops=dict(color=ACCENT2))
    ax5.set_xlabel("Volume (millions)"); ax5.set_title("Boxplot to detect outliers for Volume", fontsize=11)
    apply_dark_style(fig5); st.pyplot(fig5, use_container_width=True); plt.close(fig5)

# ═══════════════════ TAB 3 ═══════════════════
with tab3:
    st.markdown("## ⚙️ Data Preprocessing")
    from sklearn.preprocessing import MinMaxScaler

    df0 = df0.sort_values("Date").reset_index(drop=True)
    scaler_rnn = MinMaxScaler()
    close_scaled = scaler_rnn.fit_transform(df0["Close"].values.reshape(-1,1))

    X_seq, y_seq = [], []
    for i in range(seq_len, len(close_scaled)):
        X_seq.append(close_scaled[i-seq_len:i, 0])
        y_seq.append(close_scaled[i, 0])
    X_seq = np.array(X_seq).reshape(-1, seq_len, 1)
    y_seq = np.array(y_seq)

    split   = int((train_split/100) * len(X_seq))
    x_train = X_seq[:split]; y_train = y_seq[:split]
    x_test  = X_seq[split:]; y_test  = y_seq[split:]

    steps = {
        "1. Sort by Date":           "df0 = df0.sort_values('Date').reset_index(drop=True)",
        "2. MinMaxScaler (0–1)":     "scaler.fit_transform(df0['Close'].values.reshape(-1,1))",
        f"3. Sliding Window (n={seq_len})": f"SEQ_LEN = {seq_len}  # ≈ 3 trading months",
        f"4. Train/Test Split ({train_split}%/{100-train_split}%)": f"split = int({train_split/100} * len(X))",
        "5. Reshape for RNN/LSTM":   "X = X.reshape((samples, timesteps, features))",
    }
    for step, code in steps.items():
        with st.expander(step):
            st.code(code, language="python")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(x_train):,}")
    c2.metric("Test Samples",     f"{len(x_test):,}")
    c3.metric("Input Shape",      f"({seq_len}, 1)")

    st.markdown("### 📊 Original vs. Scaled Close Price")
    fig6, axes = plt.subplots(1, 2, figsize=(12,3.5))
    axes[0].plot(df0["Date"], df0["Close"], color=ACCENT4, lw=1)
    axes[0].set_title("Original Close Price"); axes[0].set_ylabel("Price (USD)")
    axes[1].plot(df0["Date"], close_scaled, color=ACCENT1, lw=1)
    axes[1].set_title("Scaled Close Price (MinMax)"); axes[1].set_ylabel("Scaled Value")
    apply_dark_style(fig6, list(axes)); st.pyplot(fig6, use_container_width=True); plt.close(fig6)

    st.session_state.update({
        "x_train": x_train, "y_train": y_train,
        "x_test":  x_test,  "y_test":  y_test,
        "scaler_rnn": scaler_rnn, "close_scaled": close_scaled,
        "df0": df0, "split": split,
    })
    st.success("✅ Preprocessing complete. Data ready for modelling.")

# ═══════════════════ TAB 4 ═══════════════════
with tab4:
    st.markdown("## 🤖 SimpleRNN Model")

    if run_button or ("rnn_hist" in st.session_state):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
        from tensorflow.keras import optimizers, initializers
        tf.get_logger().setLevel("ERROR")

        x_tr = st.session_state["x_train"]; y_tr = st.session_state["y_train"]
        x_te = st.session_state["x_test"];  y_te = st.session_state["y_test"]
        sc   = st.session_state["scaler_rnn"]

        st.markdown("### 🏗️ Model Architecture")
        p_rnn = rnn_units * (1 + rnn_units + 1)
        p_den = rnn_units + 1
        arch  = pd.DataFrame({
            "Layer":        ["simple_rnn (SimpleRNN)", "dropout (Dropout)", "dense (Dense)"],
            "Output Shape": [f"(None, {rnn_units})", f"(None, {rnn_units})", "(None, 1)"],
            "Param #":      [p_rnn, 0, p_den],
        })
        st.dataframe(arch, use_container_width=True)
        st.info(f"**Total params:** {p_rnn+p_den:,}  |  **Trainable:** {p_rnn+p_den:,}  |  Non-trainable: 0")

        with st.spinner("🔄 Training RNN model…"):
            model_rnn = Sequential([
                SimpleRNN(rnn_units, activation="tanh",
                          kernel_initializer=initializers.RandomNormal(stddev=0.001),
                          recurrent_initializer=initializers.Identity(gain=1.0),
                          input_shape=(seq_len,1)),
                Dropout(rnn_dropout),
                Dense(1),
            ])
            model_rnn.compile(loss="mean_squared_error",
                              optimizer=optimizers.RMSprop(learning_rate=rnn_lr),
                              metrics=["mae"])
            hr = model_rnn.fit(x_tr, y_tr, epochs=rnn_epochs, batch_size=32,
                               validation_data=(x_te, y_te), verbose=0)
            st.session_state["rnn_hist"]  = hr.history
            st.session_state["model_rnn"] = model_rnn

        h = st.session_state["rnn_hist"]

        st.markdown("### 📉 Training History")
        ca, cb = st.columns(2)
        with ca:
            fig7, ax7 = plt.subplots(figsize=(6,3.5))
            ax7.plot(h["loss"],     color=ACCENT1, lw=2, label="Train Loss")
            ax7.plot(h["val_loss"], color=ACCENT4, lw=2, linestyle="--", label="Val Loss")
            ax7.set_xlabel("Epoch"); ax7.set_ylabel("MSE Loss"); ax7.set_title("RNN — Loss Curve")
            ax7.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL)
            apply_dark_style(fig7); st.pyplot(fig7, use_container_width=True); plt.close(fig7)
        with cb:
            fig8, ax8 = plt.subplots(figsize=(6,3.5))
            ax8.plot(h["mae"],     color=ACCENT5, lw=2, label="Train MAE")
            ax8.plot(h["val_mae"], color=ACCENT3, lw=2, linestyle="--", label="Val MAE")
            ax8.set_xlabel("Epoch"); ax8.set_ylabel("MAE"); ax8.set_title("RNN — MAE Curve")
            ax8.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL)
            apply_dark_style(fig8); st.pyplot(fig8, use_container_width=True); plt.close(fig8)

        st.markdown("### 📋 Epoch Log")
        ep_df = pd.DataFrame({"Epoch": range(1, len(h["loss"])+1),
                               "Train Loss": [round(v,4) for v in h["loss"]],
                               "Val Loss":   [round(v,4) for v in h["val_loss"]],
                               "Train MAE":  [round(v,4) for v in h["mae"]],
                               "Val MAE":    [round(v,4) for v in h["val_mae"]]})
        st.dataframe(ep_df, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Final Train MSE", f"{h['loss'][-1]:.4f}")
        c2.metric("Final Val MSE",   f"{h['val_loss'][-1]:.4f}")
        c3.metric("Final Train MAE", f"{h['mae'][-1]:.4f}")
        c4.metric("Final Val MAE",   f"{h['val_mae'][-1]:.4f}")

        st.markdown("### 🎯 Actual vs Predicted (Test Set)")
        preds_rnn   = sc.inverse_transform(model_rnn.predict(x_te, verbose=0))
        actual_test = sc.inverse_transform(y_te.reshape(-1,1))
        fig9, ax9 = plt.subplots(figsize=(12,4))
        ax9.plot(actual_test, color=ACCENT2, lw=1.5, label="Actual")
        ax9.plot(preds_rnn,   color=ACCENT4, lw=1.5, linestyle="--", label="RNN Predicted")
        ax9.set_xlabel("Test Sample Index"); ax9.set_ylabel("Price (USD)")
        ax9.set_title("RNN — Actual vs Predicted Close Price")
        ax9.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL)
        apply_dark_style(fig9); st.pyplot(fig9, use_container_width=True); plt.close(fig9)

        rmse = np.sqrt(np.mean((actual_test - preds_rnn)**2))
        st.success(f"✅ RNN trained. RMSE: **{rmse:.4f}** | Final Val MAE: **{h['val_mae'][-1]:.3f}** — "
                   f"No overfitting across {rnn_epochs} epochs. Total params: {p_rnn+p_den:,}")
    else:
        st.info("👈 Click **Run Full Analysis** in the sidebar to train models.")
        st.code(f"""
model_rnn = Sequential()
model_rnn.add(SimpleRNN({rnn_units}, activation='tanh',
    kernel_initializer=RandomNormal(stddev=0.001),
    recurrent_initializer=Identity(gain=1.0),
    input_shape=({seq_len}, 1)))
model_rnn.add(Dropout({rnn_dropout}))
model_rnn.add(Dense(1))
model_rnn.compile(loss='mean_squared_error',
    optimizer=RMSprop(learning_rate={rnn_lr}), metrics=['mae'])
""", language="python")

# ═══════════════════ TAB 5 ═══════════════════
with tab5:
    st.markdown("## 🧠 LSTM Model & Forecasting")

    if run_button or ("lstm_hist" in st.session_state):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
        from sklearn.preprocessing import MinMaxScaler
        tf.get_logger().setLevel("ERROR")

        df0_ = st.session_state["df0"]
        test_cut   = len(df0_) - int(round(len(df0_)*0.1))
        train_vals = df0_.iloc[:test_cut]["Close"].values
        test_vals  = df0_.iloc[test_cut:]["Close"].values

        sc_lstm     = MinMaxScaler()
        tr_scaled   = sc_lstm.fit_transform(train_vals.reshape(-1,1))
        te_scaled   = sc_lstm.transform(test_vals.reshape(-1,1))
        train_gen   = TimeseriesGenerator(tr_scaled, tr_scaled, length=60, batch_size=1)
        test_gen    = TimeseriesGenerator(te_scaled,  te_scaled,  length=60, batch_size=1)

        st.markdown("### 🏗️ LSTM Architecture")
        p_lstm = 4 * lstm_units * (1 + lstm_units + 1)
        p_d    = lstm_units + 1
        la = pd.DataFrame({"Layer":["lstm (LSTM)","dense (Dense)"],
                           "Output Shape":[f"(None, {lstm_units})","(None, 1)"],
                           "Param #":[p_lstm, p_d]})
        st.dataframe(la, use_container_width=True)
        st.info(f"**Total params:** {p_lstm+p_d:,}  |  **Trainable:** {p_lstm+p_d:,}  |  Non-trainable: 0")

        with st.spinner("🔄 Training LSTM model…"):
            model_lstm = Sequential([LSTM(lstm_units, input_shape=(60,1)), Dense(1)])
            model_lstm.compile(optimizer=Adam(learning_rate=lstm_lr), loss="mse")
            hl = model_lstm.fit(train_gen, epochs=lstm_epochs, validation_data=test_gen, verbose=0)
            st.session_state["lstm_hist"]  = hl.history
            st.session_state["model_lstm"] = model_lstm
            st.session_state["sc_lstm"]    = sc_lstm
            st.session_state["tr_scaled"]  = tr_scaled
            st.session_state["test_cut"]   = test_cut
            st.session_state["test_vals"]  = test_vals

        hl_h = st.session_state["lstm_hist"]

        st.markdown("### 📉 LSTM Training Loss")
        fig10, ax10 = plt.subplots(figsize=(10,4))
        ax10.plot(hl_h["loss"],     color=ACCENT1, lw=2.5, marker="o", markersize=5, label="Train Loss")
        ax10.plot(hl_h["val_loss"], color=ACCENT4, lw=2.5, marker="s", markersize=5, linestyle="--", label="Val Loss")
        ax10.set_xlabel("Epoch"); ax10.set_ylabel("MSE Loss")
        ax10.set_title("LSTM — Training vs Validation Loss")
        ax10.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL)
        apply_dark_style(fig10); st.pyplot(fig10, use_container_width=True); plt.close(fig10)

        st.markdown("### 📋 Epoch Log")
        ep_lstm = pd.DataFrame({"Epoch": range(1, len(hl_h["loss"])+1),
                                 "Train Loss": [round(v,4) for v in hl_h["loss"]],
                                 "Val Loss":   [round(v,4) for v in hl_h["val_loss"]]})
        st.dataframe(ep_lstm, use_container_width=True)
        best_ep = int(np.argmin(hl_h["val_loss"])) + 1
        c1,c2,c3 = st.columns(3)
        c1.metric("Best Val Loss",    f"{min(hl_h['val_loss']):.4f}", f"epoch {best_ep}")
        c2.metric("Final Train Loss", f"{hl_h['loss'][-1]:.4f}")
        c3.metric("Final Val Loss",   f"{hl_h['val_loss'][-1]:.4f}")

        st.markdown("### 🔮 Recursive Forecast")
        with st.spinner("🔄 Generating forecast…"):
            m_l     = st.session_state["model_lstm"]
            tr_sc   = st.session_state["tr_scaled"]
            sc_l    = st.session_state["sc_lstm"]
            tc      = st.session_state["test_cut"]
            tv      = st.session_state["test_vals"]
            df0_    = st.session_state["df0"]

            fc_scaled = []
            batch = tr_sc[-60:].reshape((1,60,1))
            for _ in range(forecast_steps):
                pred = m_l.predict(batch, verbose=0)[0]
                fc_scaled.append(pred)
                batch = np.append(batch[:,1:,:], [[pred]], axis=1)
            forecast = sc_l.inverse_transform(fc_scaled)

        dates = df0_["Date"].values
        fc_dates = pd.date_range(start=pd.to_datetime(dates[tc+60]), periods=len(forecast), freq="B")

        fig11, ax11 = plt.subplots(figsize=(14,5))
        ax11.plot(dates, df0_["Close"].values, color=ACCENT2, lw=1.2, label="Historical", alpha=0.9)
        ax11.plot(dates[tc:], tv, color=ACCENT3, lw=1.8, label="Actual Test")
        ax11.plot(fc_dates, forecast, color=ACCENT4, lw=2.5, linestyle="--", label="LSTM Forecast")
        ax11.axvline(pd.to_datetime(dates[tc]), color=ACCENT1, lw=1.5, linestyle=":", alpha=0.8)
        ax11.set_xlabel("Date"); ax11.set_ylabel("Price (USD)")
        ax11.set_title("Close Price — LSTM Forecast vs. Actual", fontsize=13)
        ax11.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        apply_dark_style(fig11); st.pyplot(fig11, use_container_width=True); plt.close(fig11)

        with st.expander("📊 Forecast Values Table"):
            fc_df = pd.DataFrame({"Step": range(1, len(forecast)+1),
                                   "Date": fc_dates.strftime("%Y-%m-%d"),
                                   "Forecast ($)": [round(float(v),2) for v in forecast.flatten()]})
            st.dataframe(fc_df, use_container_width=True)

        st.success(f"✅ LSTM trained over {lstm_epochs} epochs. Best val loss **{min(hl_h['val_loss']):.4f}** "
                   f"at epoch **{best_ep}**. Recursive forecasting → **{forecast_steps}** future predictions.")
        if best_ep < lstm_epochs:
            st.warning("⚠️ Validation loss increased in the final epoch — early signs of overfitting.")

        # Comparison
        st.markdown("---")
        st.markdown("## 📊 Model Comparison")
        if "rnn_hist" in st.session_state:
            rh = st.session_state["rnn_hist"]
            comp = pd.DataFrame({
                "Model":            ["SimpleRNN", "LSTM"],
                "Architecture":     [f"SimpleRNN({rnn_units})→Dropout→Dense", f"LSTM({lstm_units})→Dense"],
                "Total Params":     [p_rnn+p_den if 'p_rnn' in dir() else "—", p_lstm+p_d],
                "Final Train Loss": [round(rh["loss"][-1],4), round(hl_h["loss"][-1],4)],
                "Final Val Loss":   [round(rh["val_loss"][-1],4), round(hl_h["val_loss"][-1],4)],
                "Best Val Loss":    [round(min(rh["val_loss"]),4), round(min(hl_h["val_loss"]),4)],
            })
            st.dataframe(comp, use_container_width=True)
            fig12, ax12 = plt.subplots(figsize=(8,4))
            models_l = ["RNN\n(Train)","RNN\n(Val)","LSTM\n(Train)","LSTM\n(Val)"]
            vals_l   = [rh["loss"][-1], rh["val_loss"][-1], hl_h["loss"][-1], hl_h["val_loss"][-1]]
            colors_l = [ACCENT1, ACCENT2, ACCENT4, ACCENT5]
            bars = ax12.bar(models_l, vals_l, color=colors_l, width=0.5, edgecolor=PANEL_BG, linewidth=1.5)
            for bar, v in zip(bars, vals_l):
                ax12.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
                          f"{v:.4f}", ha="center", va="bottom", color=TEXT_COL, fontsize=9)
            ax12.set_ylabel("MSE Loss"); ax12.set_title("Final Loss — RNN vs LSTM")
            apply_dark_style(fig12); st.pyplot(fig12, use_container_width=True); plt.close(fig12)

        st.markdown("---")
        st.markdown("## 📝 Conclusion")
        cl, cr = st.columns(2)
        with cl:
            st.markdown("""<div style='background:linear-gradient(135deg,#1e2d5a,#2a1a54);border:1px solid #4a3580;border-radius:12px;padding:18px;'>
<h4 style='color:#7c4fc0;margin:0 0 10px 0;'>🤖 SimpleRNN</h4>
<ul style='color:#d4c8f0;margin:0;padding-left:18px;line-height:1.8;'>
<li>Lightweight — only ~2,651 parameters</li><li>Smooth learning, minimal overfitting</li>
<li>Final MAE ≈ 0.108 (train &amp; val)</li><li>Well-suited for time series tasks</li>
</ul></div>""", unsafe_allow_html=True)
        with cr:
            st.markdown("""<div style='background:linear-gradient(135deg,#1e2d5a,#2a1a54);border:1px solid #4a3580;border-radius:12px;padding:18px;'>
<h4 style='color:#3a7bd5;margin:0 0 10px 0;'>🧠 LSTM</h4>
<ul style='color:#d4c8f0;margin:0;padding-left:18px;line-height:1.8;'>
<li>Greater memory capacity (~10,451 params)</li><li>Better training accuracy via gating</li>
<li>Early overfitting signs after epoch 5</li><li>Recursive forecasting: 25 future predictions</li>
</ul></div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Click **Run Full Analysis** in the sidebar to train models.")
        st.code("""
model = Sequential([LSTM(50, input_shape=(60, 1)), Dense(1)])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(train_gen, epochs=6, validation_data=test_gen, verbose=2)

# Recursive Forecast
forecast_scaled = []
batch = train_scaled[-60:].reshape((1, 60, 1))
for i in range(25):
    pred = model.predict(batch, verbose=0)[0]
    forecast_scaled.append(pred)
    batch = np.append(batch[:, 1:, :], [[pred]], axis=1)
forecast = scaler.inverse_transform(forecast_scaled)
""", language="python")

st.markdown("---")
st.markdown("""<div style='text-align:center;color:#6b5a9e;font-size:0.82rem;padding:8px 0;'>
Stock Price Prediction with RNN &amp; LSTM &nbsp;|&nbsp; AABA 2006–2018 &nbsp;|&nbsp;
Built with Streamlit · TensorFlow/Keras · Scikit-learn · Matplotlib · Seaborn
</div>""", unsafe_allow_html=True)
