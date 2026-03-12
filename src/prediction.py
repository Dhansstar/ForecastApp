import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# --- 1. CONFIG & LOAD DATA ---
st.set_page_config(page_title="DemandSense Forecast", layout="wide")

@st.cache_resource
def load_assets():
    # Load Metadata
    with open('models_demandsense_v1/model_metadata.json', 'r') as f:
        meta = json.load(f)
    
    # Load Models
    fe_model = load_model('models_demandsense_v1/feature_extractor.keras')
    vol_model = joblib.load('models_demandsense_v1/xgb_vol_model.joblib')
    mape_model = joblib.load('models_demandsense_v1/xgb_mape_model.joblib')
    
    return meta, fe_model, vol_model, mape_model

# Load assets
meta, fe_model, vol_model, mape_model = load_assets()

# Simulasikan data (Ganti dengan pd.read_csv data aslimu)
@st.cache_data
def get_data():
    # Di sini lo load dataset 'full_df' lo yang terakhir
    df = pd.read_csv('your_processed_data.csv') 
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
    return df

try:
    full_df = get_data()
except:
    st.error("Gagal load data. Pastikan 'your_processed_data.csv' tersedia!")
    st.stop()

# --- 2. SIDEBAR & UI ---
st.sidebar.title("📦 DemandSense")
st.sidebar.info("AI-Driven Inventory Forecasting Engine")

kategori_list = list(meta['final_recipes'].keys())
selected_kat = st.sidebar.selectbox("Pilih Kategori Produk:", kategori_list)

st.title(f"Forecast Penjualan: {selected_kat}")
st.markdown(f"Prediksi kebutuhan stok untuk 30 hari ke depan berdasarkan data historis.")

# --- 3. RECURSIVE FORECASTING ENGINE ---
def run_forecast(kat):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    # Ambil window terakhir
    curr_df = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').tail(time_steps)
    curr_win = curr_df[meta['features']].values.tolist()
    kat_onehot = curr_df[meta['kat_cols']].iloc[0].tolist()
    last_date = curr_df['Waktu Pesanan Dibuat'].max()
    
    daily_preds = []
    
    # Loop 30 Hari
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        lat = fe_model.predict(X_in, verbose=0)
        
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        d_p = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Apply Threshold
        d_p = d_p if d_p >= recipe['thresh'] else 0
        
        # Smoothing/Rounding
        if recipe['smooth']:
            daily_preds.append(d_p)
        else:
            val = np.ceil(d_p) if kat in ['Kitchen', 'Home'] else np.round(d_p)
            daily_preds.append(val)
        
        # Update Window
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(d_p),
            curr_win[-7][2] if len(curr_win)>=7 else 0,
            curr_win[-28][2] if len(curr_win)>=28 else 0,
            np.mean([r[2] for r in curr_win[-7:]])
        ]
        new_row.extend(kat_onehot)
        curr_win.append(new_row)
    
    # Final Smoothing if needed
    if recipe['smooth']:
        daily_preds = pd.Series(daily_preds).rolling(window=5, min_periods=1, center=True).mean().values
        daily_preds = np.floor(daily_preds + 0.3)
        
    total_recom = int(np.ceil(np.sum(daily_preds) * recipe['mult']))
    
    return daily_preds, total_recom, last_date

# Jalankan Engine
daily_preds, total_recom, last_date = run_forecast(selected_kat)

# --- 4. VISUALISASI ---
col1, col2 = st.columns([3, 1])

with col1:
    # Logic Plotly yang tadi udah kita buat
    hist_data = full_df[full_df['Kategori'] == selected_kat].tail(30)
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data['Waktu Pesanan Dibuat'], y=hist_data['Net_Sales'], name="Historis", line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, name="Forecast", line=dict(color='#ff7f0e', dash='dash')))
    
    fig.update_layout(template="plotly_white", hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Total Rekomendasi Stok", f"{total_recom} Unit")
    st.success(f"Gunakan multiplier x{meta['final_recipes'][selected_kat]['mult']} untuk kalibrasi.")
    
    # Tabel prediksi harian
    st.write("📋 **Detail Harian**")
    df_detail = pd.DataFrame({'Tanggal': forecast_dates, 'Prediksi': daily_preds})
    st.dataframe(df_detail, height=300)

st.divider()
st.caption("Developed by Risyadhana Syaifuddin - IT Professional & Data Science Specialist")