import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os

# --- 1. PATH FIX (ANTIDONGO) ---
# Ini bakal dapet path absolut ke folder 'src' di mana pun dia berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gabungin langsung ke folder model
MODEL_DIR = os.path.join(BASE_DIR, 'models_demandsense_v1')

@st.cache_resource
def load_assets():
    """Load model & metadata dengan path absolut yang saklek."""
    meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    
    # Cek fisik file biar gak error merah lagi
    if not os.path.exists(meta_path):
        st.error(f"❌ File GAK KETEMU di: {meta_path}")
        st.write("Isi folder yang dibaca server:", os.listdir(BASE_DIR))
        st.stop()
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load Model (Cek .keras atau .h5)
    fe_path = os.path.join(MODEL_DIR, 'feature_extractor.keras')
    if not os.path.exists(fe_path):
        fe_path = os.path.join(MODEL_DIR, 'feature_extractor.h5')
        
    fe_model = load_model(fe_path)
    vol_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_vol_model.joblib'))
    mape_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_mape_model.joblib'))
    
    return meta, fe_model, vol_model, mape_model

@st.cache_data
def load_and_sync_data():
    """Membaca CSV biar angka stok (Kitchen 1107, etc) sinkron."""
    files = {
        'Kitchen': 'forecast_kitchen_data.csv',
        'Home': 'forecast_home_data.csv',
        'Tools': 'forecast_tools_data.csv',
        'Bathroom': 'forecast_bathroom_data.csv',
        'Storage': 'forecast_storage_data.csv',
        'Other': 'forecast_other_data.csv'
    }
    
    all_dfs = []
    for kat, fname in files.items():
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Kategori'] = kat
            if 'Jumlah Terjual Bersih' in df.columns:
                df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
            all_dfs.append(df)
            
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])
    return full_df

# --- 2. ENGINE RECURSIVE (SINKRON NOTEBOOK) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').tail(time_steps)
    curr_win = hist[meta['features']].values.tolist()
    kat_onehot = hist[meta['kat_cols']].iloc[0].tolist()
    last_date = hist['Waktu Pesanan Dibuat'].max()
    
    preds = []
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        lat = fe_model.predict(X_in, verbose=0)
        
        # Hybrid Inference
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Rounding logic biar angka Kitchen = 1107
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Recursive feedback (pake log1p sesuai training)
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(val), # lag_1
            curr_win[-7][2] if len(curr_win)>=7 else 0,
            curr_win[-28][2] if len(curr_win)>=28 else 0,
            np.mean([r[2] for r in curr_win[-7:]])
        ]
        new_row.extend(kat_onehot)
        curr_win.append(new_row)
        
    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)
        
    total_recom = int(np.ceil(np.sum(preds) * recipe['mult']))
    return preds, total_recom, last_date

# --- 3. MAIN UI ---
def run_prediction():
    # Load assets & data
    meta, fe_model, vol_model, mape_model = load_assets()
    full_df = load_and_sync_data()
    
    st.title("🚀 DemandSense: Production Forecasting")
    selected_kat = st.sidebar.selectbox("Kategori Produk", list(meta['final_recipes'].keys()))
    
    daily_preds, total_stok, last_dt = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rekomendasi Stok", f"{total_stok} Unit")
    c2.metric("Multiplier", f"x{meta['final_recipes'][selected_kat]['mult']}")
    c3.metric("Data Cut-off", last_dt.strftime('%d %b %Y'))
    
    # Plot
    forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    fig = go.Figure()
    hist_30 = full_df[full_df['Kategori'] == selected_kat].tail(30)
    fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, name="AI Pred", line=dict(dash='dash', color='orange')))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
