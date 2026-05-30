# 📈 Stock Price Prediction with RNN & LSTM

Aplikasi Streamlit interaktif untuk memprediksi harga saham menggunakan arsitektur **Recurrent Neural Network (SimpleRNN)** dan **Long Short-Term Memory (LSTM)** berbasis TensorFlow/Keras.

Dataset yang digunakan adalah data historis harga saham **AABA (Yahoo Finance / Altaba)** periode **2006–2018** dengan 3.019 entri harian.

---

## 🖥️ Tampilan Aplikasi

```
Tab 1 — 📋 Data Overview       → eksplorasi awal dataset
Tab 2 — 📊 EDA & Visualisation → grafik harga, volume, return
Tab 3 — ⚙️  Preprocessing       → scaling, windowing, splitting
Tab 4 — 🤖 RNN Model           → training, evaluasi, prediksi
Tab 5 — 🧠 LSTM & Forecast     → training, forecast rekursif, komparasi
```

Tema visual: **biru tua + ungu muda** — seluruh chart Matplotlib mengikuti dark theme yang konsisten.

---

## 🗂️ Struktur Proyek

```
stock-prediction-rnn-lstm/
├── app.py               # Aplikasi utama Streamlit
├── requirements.txt     # Daftar dependensi Python
└── README.md            # Dokumentasi proyek
```

---

## ⚙️ Instalasi & Menjalankan

### 1. Clone / unduh proyek

```bash
git clone https://github.com/username/stock-prediction-rnn-lstm.git
cd stock-prediction-rnn-lstm
```

### 2. (Opsional) Buat virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📦 Dependensi

| Library | Versi | Kegunaan |
|---|---|---|
| `streamlit` | 1.57.0 | Framework aplikasi web interaktif |
| `tensorflow` | 2.21.0 | Backend deep learning (RNN & LSTM) |
| `keras` | 3.14.1 | High-level API neural network |
| `scikit-learn` | 1.8.0 | MinMaxScaler untuk normalisasi data |
| `pandas` | 3.0.2 | Manipulasi dan analisis data tabular |
| `numpy` | 2.4.4 | Komputasi numerik & array multidimensi |
| `matplotlib` | 3.10.8 | Visualisasi data (line chart, boxplot) |
| `seaborn` | 0.13.2 | Visualisasi statistik (histplot, KDE) |

---

## 🧪 Alur Kerja (Pipeline)

```
Raw Data (AABA OHLCV)
        │
        ▼
┌──────────────────┐
│  EDA             │  Moving Averages (MA20/50/200), Volume,
│  & Visualisasi   │  Daily Returns, Distribusi, Outlier
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │  MinMaxScaler → Sliding Window (n=60)
│                  │  → Train/Test Split (80/20) → Reshape 3D
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────┐
│  RNN  │ │ LSTM │
│ Model │ │Model │
└───┬───┘ └──┬───┘
    │         │
    ▼         ▼
Evaluasi  Recursive
MAE/MSE   Forecast
          (25 steps)
```

---

## 🏗️ Arsitektur Model

### SimpleRNN
```
Layer               Output Shape    Param #
────────────────────────────────────────────
SimpleRNN (tanh)    (None, 50)      2,600
Dropout (0.2)       (None, 50)          0
Dense               (None,  1)         51
────────────────────────────────────────────
Total params: 2,651 (10.36 KB)
```

- **Optimizer:** RMSprop (lr = 0.0005)
- **Loss:** Mean Squared Error
- **Metric:** Mean Absolute Error (MAE)
- **Epochs:** 15 | **Batch size:** 32

### LSTM
```
Layer               Output Shape    Param #
────────────────────────────────────────────
LSTM                (None, 50)     10,400
Dense               (None,  1)         51
────────────────────────────────────────────
Total params: 10,451 (40.82 KB)
```

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Mean Squared Error
- **Epochs:** 6 | **Batch size:** 1 (TimeseriesGenerator)

---

## 📊 Hasil Eksperimen

| Metrik | SimpleRNN | LSTM |
|---|---|---|
| Total Parameter | 2,651 | 10,451 |
| Final Train Loss (MSE) | ~0.021 | ~0.016 |
| Final Val Loss (MSE) | ~0.022 | ~0.029 |
| Final Train MAE | ~0.106 | — |
| Final Val MAE | ~0.108 | — |
| Overfitting | ✅ Tidak | ⚠️ Mulai epoch 5 |
| Recursive Forecast | — | 25 langkah ke depan |

---

## 🎛️ Konfigurasi via Sidebar

Semua hyperparameter dapat diubah langsung dari sidebar tanpa menyentuh kode:

| Parameter | Default | Range |
|---|---|---|
| Sequence Length (window) | 60 | 30 – 120 |
| Train Split | 80% | 70% – 90% |
| RNN Hidden Units | 50 | 20 – 100 |
| RNN Epochs | 15 | 5 – 30 |
| RNN Learning Rate | 0.0005 | 0.0001 – 0.005 |
| RNN Dropout | 0.2 | 0.0 – 0.5 |
| LSTM Hidden Units | 50 | 20 – 100 |
| LSTM Epochs | 6 | 3 – 15 |
| LSTM Learning Rate | 0.001 | 0.0001 – 0.005 |
| Forecast Steps | 25 | 10 – 50 |

---

## 📝 Kesimpulan

- **SimpleRNN** terbukti ringan dan efektif: hanya 2.651 parameter, MAE akhir ≈ 0.108, tidak ada tanda overfitting sepanjang 15 epoch.
- **LSTM** memiliki kapasitas memori lebih besar sehingga akurasi training lebih tinggi, namun mulai menunjukkan tanda overfitting setelah epoch ke-5.
- **Recursive forecasting** dengan LSTM berhasil menghasilkan 25 prediksi harga masa depan menggunakan pendekatan autoregresif.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan eksplorasi teknik deep learning pada data time series keuangan.
