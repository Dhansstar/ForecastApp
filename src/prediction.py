import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="DemandSense AI", layout="wide", initial_sidebar_state="expanded")

# Path handling yang aman buat Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_demandsense_v1')

@st.cache_resource
def load_engine():
    """Load semua model dan metadata sekali saja."""
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        meta = json.load(f)
    
    fe_model = load_model(os.path.join(MODEL_DIR, 'feature_extractor.keras'))
    vol_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_vol_model.joblib'))
    mape_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_mape_model.joblib'))
    
    return meta, fe_model, vol_model, mape_model

@st.cache_data
def load_dataset():
    """Load data transaksi terakhir."""
    # Pastikan file ini ada di folder src/ atau root sesuai path di bawah
    data_path = os.path.join(BASE_DIR, 'full_dataset_final.csv') 
    df = pd.read_csv(data_path)
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
    return df

# Initialize Assets
try:
    meta, fe_model, vol_model, mape_model = load_engine()
    full_df = load_dataset()
except Exception as e:
    st.error(f"Error Loading Assets: {e}")
    st.stop()

# --- 2. FORECASTING LOGIC ---
def generate_forecast(kategori_name):
    recipe = meta['final_recipes'][kategori_name]
    time_steps = meta['time_steps']
    features = meta['features']
    kat_cols = meta['kat_cols']

    # Persiapan Window Terakhir
    curr_df = full_df[full_df['Kategori'] == kategori_name].sort_values('Waktu Pesanan Dibuat').tail(time_steps)
    curr_win = curr_df[features].values.tolist()
    kat_onehot = curr_df[kat_cols].iloc[0].tolist()
    last_date = curr_df['Waktu Pesanan Dibuat'].max()

    preds = []
    
    # Recursive Loop 30 Hari
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        latent = fe_model.predict(X_in, verbose=0)
        
        p_v = vol_model.predict(latent)[0]
        p_m = mape_model.predict(latent)[0]
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Threshold & Rounding
        val = val if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kategori_name in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)

        # Update Window untuk t+1
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

    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)

    total_recom = int(np.ceil(np.sum(preds) * recipe['mult']))
    return preds, total_recom, last_date

# --- 3. UI RENDER ---
st.sidebar.header("🕹️ Control Panel")
selected_kat = st.sidebar.selectbox("Pilih Kategori Produk", list(meta['final_recipes'].keys()))

st.title("🚀 DemandSense: AI Inventory Forecasting")
st.markdown(f"Prediksi kebutuhan stok kategori **{selected_kat}** untuk 30 hari ke depan.")

# Run Forecast
daily_preds, total_stok, last_dt = generate_forecast(selected_kat)
forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Total Rekomendasi Stok", f"{total_stok} Unit")
m2.metric("Multiplier Kategori", f"x{meta['final_recipes'][selected_kat]['mult']}")
m3.metric("Last Data Point", last_dt.strftime('%d %b %Y'))

# Visualisasi
st.subheader("📈 Tren Penjualan & Proyeksi")
hist_data = full_df[full_df['Kategori'] == selected_kat].tail(30)

fig = go.Figure()
# Historis
fig.add_trace(go.Scatter(x=hist_data['Waktu Pesanan Dibuat'], y=hist_data['Net_Sales'], 
                         name="Historical Sales", line=dict(color='#1f77b4', width=3)))
# Forecast
fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, 
                         name="AI Forecast", line=dict(color='#ff7f0e', width=3, dash='dash')))

fig.update_layout(hovermode="x unified", template="plotly_white", height=500,
                  margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# Data Detail & Export
st.divider()
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Detail Prediksi Harian")
    df_result = pd.DataFrame({'Tanggal': forecast_dates, 'Estimasi_Unit': daily_preds})
    st.dataframe(df_result.style.highlight_max(axis=0), use_container_width=True, height=400)

with col_right:
    st.subheader("📥 Export Report")
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name=f'Forecast_{selected_kat}_{last_dt.date()}.csv',
        mime='text/csv',
    )
    st.info("File ini bisa langsung di-import ke sistem manajemen gudang (WMS).")

st.sidebar.markdown("---")
st.sidebar.caption("DemandSense v1.0.2 | Jakarta, Indonesia")
