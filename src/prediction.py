import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# --- 1. CONFIG PATH ---
BASE_PATH = "src/" 

@st.cache_resource
def load_assets():
    with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
        meta = json.load(f)
    fe_model = load_model(f"{BASE_PATH}feature_extractor.keras")
    vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
    mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
    return meta, fe_model, vol_model, mape_model

@st.cache_data
def load_and_sync_data():
    files = {
        'Kitchen': 'forecast_kitchen_data.csv', 'Home': 'forecast_home_data.csv',
        'Tools': 'forecast_tools_data.csv', 'Bathroom': 'forecast_bathroom_data.csv',
        'Storage': 'forecast_storage_data.csv', 'Other': 'forecast_other_data.csv'
    }
    all_dfs = []
    for kat, fname in files.items():
        try:
            df = pd.read_csv(f"{BASE_PATH}{fname}")
            df['Kategori'] = kat
            if 'Jumlah Terjual Bersih' in df.columns:
                df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
            all_dfs.append(df)
        except: continue
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])
    return full_df

# --- 2. ENGINE RECURSIVE (STRICT DIMENSION) ---
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
    
    # SINKRONISASI KOLOM SAKLEK
    all_req_cols = meta['features'] + meta['kat_cols']
    for col in all_req_cols:
        if col not in hist.columns:
            if "Kategori" in col:
                # Cek case-insensitive biar aman
                hist[col] = 1 if kat.lower() in col.lower() else 0
            else:
                hist[col] = 0
                
    hist = hist.fillna(0)
    hist_input = hist.tail(time_steps)
    
    # Ambil nilai fitur
    curr_win = hist_input[all_req_cols].values.tolist()
    kat_onehot_vals = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]
    last_date = hist_input['Waktu Pesanan Dibuat'].max()
    
    # CEK DIMENSI SEBELUM PREDICT (Pencegah ValueError)
    expected_dim = fe_model.input_shape[-1]
    
    preds = []
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        
        # Jika dimensi gak cocok, potong atau tambahin 0 secara paksa
        if X_in.shape[-1] != expected_dim:
            if X_in.shape[-1] > expected_dim:
                X_in = X_in[:, :, :expected_dim] # Potong kalau kelebihan
            else:
                # Tambah kolom 0 kalau kurang (Harusnya gak terjadi kalau meta bener)
                padding = np.zeros((1, time_steps, expected_dim - X_in.shape[-1]))
                X_in = np.concatenate([X_in, padding], axis=-1)

        lat = fe_model.predict(X_in, verbose=0)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Feedback loop t+1
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31), np.log1p(val),
                   curr_win[-7][2] if len(curr_win)>=7 else 0,
                   curr_win[-28][2] if len(curr_win)>=28 else 0,
                   np.mean([r[2] for r in curr_win[-7:]])]
        
        # Pastikan panjang row feedback sama dengan expected_dim
        feedback_row = new_row + kat_onehot_vals
        if len(feedback_row) > expected_dim:
            feedback_row = feedback_row[:expected_dim]
        elif len(feedback_row) < expected_dim:
            feedback_row += [0] * (expected_dim - len(feedback_row))
            
        curr_win.append(feedback_row)
        
    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)
        
    return preds, int(np.ceil(np.sum(preds) * recipe['mult'])), last_date, hist.tail(30)

# --- 3. UI & PLOT ---
def run_prediction():
    meta, fe_model, vol_model, mape_model = load_assets()
    full_df = load_and_sync_data()

    st.title(f"DemandSense: {st.sidebar.selectbox('Pilih Kategori', list(meta['final_recipes'].keys()), key='kat')}")
    selected_kat = st.session_state.kat

    daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    # Layout mirip gambar lo
    forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], 
                             mode='lines+markers', name='Historis', line=dict(color='#1f77b4', width=3)))
    
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, 
                             mode='lines+markers', name='Forecast', line=dict(color='#ff7f0e', width=3, dash='dash')))

    fig.add_vrect(x0=forecast_dates[0], x1=forecast_dates[-1], fillcolor="orange", opacity=0.1, line_width=0)

    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=f"Total Estimasi Stok: {total_stok} Unit",
                       showarrow=False, font=dict(color="white", size=14), bgcolor="#ff7f0e", borderpad=10)

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
