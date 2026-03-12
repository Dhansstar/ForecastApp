import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os

# --- 1. SETUP PATH & CONFIG ---
st.set_page_config(page_title="DemandSense Production", layout="wide")

# Mengambil path absolut dari folder 'src'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gabungkan ke folder model yang ada di dalam 'src'
MODEL_DIR = os.path.join(BASE_DIR, 'models_demandsense_v1')

@st.cache_resource
def load_assets():
    """Load model dan metadata dengan path absolut."""
    meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    
    # Cek apakah file metadata ada
    if not os.path.exists(meta_path):
        st.error(f"File tidak ditemukan di: {meta_path}")
        st.stop()
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load Model (Coba .keras, fallback ke .h5)
    fe_path = os.path.join(MODEL_DIR, 'feature_extractor.keras')
    if not os.path.exists(fe_path):
        fe_path = os.path.join(MODEL_DIR, 'feature_extractor.h5')
    
    fe_model = load_model(fe_path)
    vol_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_vol_model.joblib'))
    mape_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_mape_model.joblib'))
    
    return meta, fe_model, vol_model, mape_model

@st.cache_data
def load_and_sync_data():
    """Membaca CSV per kategori agar hasilnya sinkron dengan notebook."""
    # Mapping file sesuai yang kamu upload
    files = {
        'Kitchen': 'forecast_kitchen_data.csv',
        'Home': 'forecast_home_data.csv',
        'Tools': 'forecast_tools_data.csv',
        'Bathroom': 'forecast_bathroom_data.csv',
        'Storage': 'forecast_storage_data.csv',
        'Other': 'forecast_other_data.csv'
    }
    
    all_dfs = []
    for kat_name, file_name in files.items():
        # Membaca file dari folder 'src'
        path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Kategori'] = kat_name
            # Sinkronisasi nama kolom target ke Net_Sales
            if 'Jumlah Terjual Bersih' in df.columns:
                df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
            all_dfs.append(df)
        else:
            st.warning(f"File {file_name} tidak ditemukan di folder src!")
            
    if not all_dfs:
        st.error("Tidak ada data CSV yang berhasil dimuat!")
        st.stop()
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])
    return full_df

# Load Engine & Data
meta, fe_model, vol_model, mape_model = load_assets()
full_df = load_and_sync_data()

# --- 2. RECURSIVE FORECASTING ENGINE ---
def run_recursive_forecast(kat):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    # Ambil data historis terakhir untuk kategori ini
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').tail(time_steps)
    curr_win = hist[meta['features']].values.tolist()
    kat_onehot = hist[meta['kat_cols']].iloc[0].tolist()
    last_date = hist['Waktu Pesanan Dibuat'].max()
    
    preds = []
    # Loop 30 hari kedepan secara rekursif
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        lat = fe_model.predict(X_in, verbose=0)
        
        # Hybrid Inference (Volume + MAPE weights)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Post-Processing sesuai Notebook
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            # Rounding logic khusus agar angka total stok sama dengan notebook
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Feed back ke window (Recursive)
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(val), # lag_1
            curr_win[-7][2] if len(curr_win)>=7 else 0, # lag_7
            curr_win[-28][2] if len(curr_win)>=28 else 0, # lag_28
            np.mean([r[2] for r in curr_win[-7:]]) # rolling mean
        ]
        new_row.extend(kat_onehot)
        curr_win.append(new_row)
        
    # Smoothing jika diaktifkan di metadata
    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)
        
    total_recom = int(np.ceil(np.sum(preds) * recipe['mult']))
    return preds, total_recom, last_date

# --- 3. UI DASHBOARD ---
st.title("🚀 DemandSense: Production Forecasting")
st.markdown("---")

# Sidebar untuk memilih kategori
st.sidebar.header("Control Panel")
kategori_options = list(meta['final_recipes'].keys())
selected_kat = st.sidebar.selectbox("Pilih Kategori Produk", kategori_options)

# Eksekusi Forecast
daily_preds, total_stok, last_dt = run_recursive_forecast(selected_kat)
forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Rekomendasi Stok", f"{total_stok} Unit")
c2.metric("Multiplier Kategori", f"x{meta['final_recipes'][selected_kat]['mult']}")
c3.metric("Data Terakhir", last_dt.strftime('%d %b %Y'))

# Visualisasi Plotly
fig = go.Figure()
hist_view = full_df[full_df['Kategori'] == selected_kat].tail(30)

fig.add_trace(go.Scatter(x=hist_view['Waktu Pesanan Dibuat'], y=hist_view['Net_Sales'], 
                         name="Historical", line=dict(color='#1f77b4', width=2)))
fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, 
                         name="AI Forecast", line=dict(color='#ff7f0e', width=3, dash='dash')))

fig.update_layout(title=f"Trend Demand: {selected_kat}", template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Detail Data
st.subheader("📋 Rincian Prediksi Harian")
df_res = pd.DataFrame({'Tanggal': forecast_dates, 'Prediksi Unit': daily_preds})
st.dataframe(df_res.set_index('Tanggal'), use_container_width=True)

# Export
csv = df_res.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast Report (CSV)",
    data=csv,
    file_name=f"DemandSense_{selected_kat}.csv",
    mime='text/csv'
)
