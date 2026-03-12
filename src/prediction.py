import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# --- 1. DIRECT PATH ---
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
        df = pd.read_csv(f"{BASE_PATH}{fname}")
        df['Kategori'] = kat
        if 'Jumlah Terjual Bersih' in df.columns:
            df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
        all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])
    return full_df

# --- 2. ENGINE RECURSIVE (FIX VALUEERROR) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    # Filter kategori & urutkan
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').copy()
    
    # Fitur Engineering (Pake Logika Notebook lo)
    hist['day_sin'] = np.sin(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['day_cos'] = np.cos(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['lag_1'] = np.log1p(hist['Net_Sales'].shift(1))
    hist['lag_7'] = hist['Net_Sales'].shift(7)
    hist['lag_28'] = hist['Net_Sales'].shift(28)
    hist['rolling_mean_7'] = hist['Net_Sales'].rolling(window=7).mean()
    
    # One-Hot Encoding sesuai daftar meta['kat_cols']
    for col in meta['kat_cols']:
        hist[col] = 1 if f"Kategori_{kat}" == col or kat in col else 0

    hist = hist.fillna(0)
    hist_input = hist.tail(time_steps)
    
    # URUTAN KOLOM HARUS SAMA PERSIS DENGAN TRAINING
    # Kita gabung fitur utama + fitur kategori
    all_feature_cols = meta['features'] + meta['kat_cols']
    
    # Ambil nilai window terakhir
    curr_win = hist_input[all_feature_cols].values.tolist()
    kat_onehot = [1 if f"Kategori_{kat}" == col or kat in col else 0 for col in meta['kat_cols']]
    last_date = hist_input['Waktu Pesanan Dibuat'].max()
    
    preds = []
    for i in range(1, 31):
        # Input ke model (Shape: [1, time_steps, n_features])
        X_in = np.array([curr_win[-time_steps:]])
        
        try:
            lat = fe_model.predict(X_in, verbose=0)
            p_v = vol_model.predict(lat)[0]
            p_m = mape_model.predict(lat)[0]
        except Exception as e:
            st.error(f"Shape Error! Model minta {fe_model.input_shape}, tapi lo kasih {X_in.shape}")
            st.stop()
            
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Feed back (Update window untuk t+1)
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(val), # lag_1
            curr_win[-7][2] if len(curr_win)>=7 else 0, # lag_7
            curr_win[-28][2] if len(curr_win)>=28 else 0, # lag_28
            np.mean([r[2] for r in curr_win[-7:]]) # rolling
        ]
        new_row.extend(kat_onehot)
        curr_win.append(new_row)
        
    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)
        
    return preds, int(np.ceil(np.sum(preds) * recipe['mult'])), last_date

# --- 3. UI ---
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
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_preds, name="Forecast", line=dict(color='#FFA500', width=3)))
    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
