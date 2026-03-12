import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import joblib
import json
import os

# --- 1. FIX NOT_EQUAL: MENGGUNAKAN LAMBDA UNTUK MENERIMA ARGUMEN STANDAR ---
def not_equal_layer(x, **kwargs):
    # Kita ambil cuma inputnya, abaikan 'name' atau argumen lain dari Keras
    if isinstance(x, list):
        return tf.math.not_equal(x[0], x[1])
    return tf.math.not_equal(x, 0) # Atau sesuaikan dengan logika model lo

# --- 2. ASSETS LOADING ---
BASE_PATH = "src/" 

@st.cache_resource
def load_assets():
    try:
        with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
            meta = json.load(f)
        
        model_path = f"{BASE_PATH}feature_extractor.h5"
        
        # Registrasi 'NotEqual' sebagai fungsi yang fleksibel
        # Keras 3 sering melabeli operasi '!=' sebagai 'NotEqual'
        custom_objects = {
            'NotEqual': not_equal_layer,
            'tf': tf # Terkadang dibutuhkan jika ada ekspresi tf langsung
        }
        
        with custom_object_scope(custom_objects):
            # compile=False tetap wajib biar gak error di Optimizer
            fe_model = load_model(model_path, compile=False, safe_mode=False)
            
        vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
        mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
        
        return meta, fe_model, vol_model, mape_model
    except Exception as e:
        st.error(f"❌ Keras 3 beneran nolak file .h5 ini: {e}")
        st.markdown("""
        ### 🛠️ Solusi Terakhir (Wajib):
        Keras 3 di Streamlit Cloud (Python 3.12) punya standar keamanan & serialisasi baru yang sangat ketat terhadap file `.h5` lama.
        
        **Lakukan ini di Laptop/Notebook lokal lo:**
        1. `model = load_model('feature_extractor.h5', compile=False)`
        2. `model.save('src/feature_extractor.keras')`
        3. Push file `.keras` baru itu ke GitHub.
        4. Ganti line loading di code ini jadi: `load_model('src/feature_extractor.keras')`
        """)
        st.stop()

# --- 3. FORECAST ENGINE (SAMA SEPERTI SEBELUMNYA) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').copy()
    
    # Feature Engineering Dasar
    hist['day_sin'] = np.sin(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['day_cos'] = np.cos(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['lag_1'] = np.log1p(hist['Net_Sales'].shift(1))
    hist['lag_7'] = hist['Net_Sales'].shift(7)
    hist['lag_28'] = hist['Net_Sales'].shift(28)
    hist['rolling_mean_7'] = hist['Net_Sales'].rolling(window=7).mean()
    
    all_req_cols = meta['features'] + meta['kat_cols']
    for col in all_req_cols:
        if col not in hist.columns:
            hist[col] = (1 if kat.lower() in col.lower() else 0) if "Kategori" in col else 0
    hist = hist.fillna(0)

    # Padding
    if len(hist) < time_steps:
        pad = pd.DataFrame(0, index=range(time_steps - len(hist)), columns=hist.columns)
        hist = pd.concat([pad, hist], ignore_index=True)
    
    hist_input = hist.tail(time_steps)
    expected_f = fe_model.input_shape[-1]
    curr_win = hist_input[all_req_cols].values.tolist()
    
    last_date = pd.to_datetime(hist_input['Waktu Pesanan Dibuat'].iloc[-1])
    kat_onehot = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]

    preds = []
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]], dtype='float32')
        X_in = X_in[:, :, :expected_f] 

        lat = fe_model.predict(X_in, verbose=0)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [np.sin(2*nxt_d.day*np.pi/31), np.cos(2*nxt_d.day*np.pi/31), np.log1p(val),
                   curr_win[-7][2] if len(curr_win)>=7 else 0,
                   curr_win[-28][2] if len(curr_win)>=28 else 0,
                   np.mean([r[2] for r in curr_win[-7:]])]
        curr_win.append((new_row + kat_onehot)[:expected_f])
        
    return preds, int(np.ceil(np.sum(preds) * recipe['mult'])), last_date, hist.tail(30)

# --- 4. UI (MENU PREDICTION) ---
def run_prediction():
    meta, fe_model, vol_model, mape_model = load_assets()
    
    # Load Data CSV
    files = {'Kitchen': 'forecast_kitchen_data.csv', 'Home': 'forecast_home_data.csv',
             'Tools': 'forecast_tools_data.csv', 'Bathroom': 'forecast_bathroom_data.csv',
             'Storage': 'forecast_storage_data.csv', 'Other': 'forecast_other_data.csv'}
    all_dfs = []
    for kat, fname in files.items():
        p = f"{BASE_PATH}{fname}"
        if os.path.exists(p):
            df = pd.read_csv(p); df['Kategori'] = kat
            if 'Jumlah Terjual Bersih' in df.columns: df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
            all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])

    # Header Box
    st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #3b82f6; margin-bottom:20px;">
            <h3 style="color:#3b82f6; margin:0;">🔮 30-Day Demand Forecasting</h3>
            <p style="color:white; margin:0;">Estimasi kebutuhan stok berdasarkan tren historis.</p>
        </div>
    """, unsafe_allow_html=True)

    # PILIH KATEGORI DI MAIN PAGE
    selected_kat = st.selectbox("Pilih Kategori Produk:", list(meta['final_recipes'].keys()))

    daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    # Visualisasi
    st.markdown(f"## Prediction Zone: <span style='color:#f97316'>{selected_kat}</span>", unsafe_allow_html=True)
    f_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], name='Historis', line=dict(color='#3b82f6', width=3)))
    fig.add_trace(go.Scatter(x=f_dates, y=daily_preds, name='Forecast', line=dict(color='#f97316', width=3, dash='dash')))
    fig.add_vrect(x0=f_dates[0], x1=f_dates[-1], fillcolor="#f97316", opacity=0.1, layer="below", line_width=0)
    
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", text=f"📦 Total Estimasi Stok: <b>{total_stok} Unit</b>",
                       showarrow=False, font=dict(color="white"), bgcolor="#f97316", borderpad=10)

    fig.update_layout(template="plotly_white", xaxis_title="Tanggal", yaxis_title="Unit Terjual", height=500)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
