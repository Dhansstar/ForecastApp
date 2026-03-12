import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os

# --- 1. ASSETS LOADING (Format Baru .keras) ---
BASE_PATH = "src/" 

@st.cache_resource
def load_assets():
    try:
        with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
            meta = json.load(f)
        
        # Pake file format baru .keras yang udah lo save ulang
        model_path = f"{BASE_PATH}feature_extractor.keras"
        
        # Keras 3 sangat lancar baca format .keras
        fe_model = load_model(model_path, compile=False)
        vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
        mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
        
        return meta, fe_model, vol_model, mape_model
    except Exception as e:
        st.error(f"Gagal Load Model: {e}")
        st.stop()

# --- 2. UI (DROPDOWN DI MAIN PAGE) ---
def run_prediction():
    meta, fe_model, vol_model, mape_model = load_assets()
    
    # Load Data CSV (Kitchen, Home, dll)
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

    # Branding Header
    st.markdown("""
        <div style="background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #3b82f6; margin-bottom:20px;">
            <h3 style="color:#3b82f6; margin:0;">🔮 30-Day Demand Forecasting</h3>
            <p style="color:white; margin:0;">Pilih kategori produk di bawah ini untuk memulai prediksi.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- DROP-DOWN DI HALAMAN UTAMA ---
    st.write("### Pilih Kategori Produk")
    selected_kat = st.selectbox("", list(meta['final_recipes'].keys()), label_visibility="collapsed")

    # Panggil fungsi forecast (pake logic run_recursive_forecast yang lama)
    daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    # Plotting (Identik Request Lo)
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

# (Pastikan fungsi run_recursive_forecast lo ada di sini juga)
