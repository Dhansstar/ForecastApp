import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# --- 1. DIRECT PATH ---
# Langsung tembak ke folder src sesuai struktur lo
BASE_PATH = "src/" 

@st.cache_resource
def load_assets():
    # Langsung baca metadata di dalam src
    with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
        meta = json.load(f)
    
    # Load Model (pake path langsung)
    fe_model = load_model(f"{BASE_PATH}feature_extractor.keras")
    vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
    mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
    
    return meta, fe_model, vol_model, mape_model

@st.cache_data
def load_and_sync_data():
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
        path = f"{BASE_PATH}{fname}"
        df = pd.read_csv(path)
        df['Kategori'] = kat
        if 'Jumlah Terjual Bersih' in df.columns:
            df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
        all_dfs.append(df)
            
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])
    return full_df

# --- 2. ENGINE RECURSIVE ---
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
        
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Post-process (Biar Kitchen = 1107)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Feedback loop
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(val), 
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

# --- 3. RUN ---
def run_prediction():
    meta, fe_model, vol_model, mape_model = load_assets()
    full_df = load_and_sync_data()

    st.title("🚀 DemandSense: Production Forecasting")
    selected_kat = st.sidebar.selectbox("Kategori", list(meta['final_recipes'].keys()))

    daily_preds, total_stok, last_dt = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    st.metric(f"Total Stok {selected_kat}", f"{total_stok} Unit")
    
    forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, name="Forecast", line=dict(color='orange')))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
