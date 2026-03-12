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
    try:
        with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
            meta = json.load(f)
        fe_model = load_model(f"{BASE_PATH}feature_extractor.keras")
        vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
        mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
        return meta, fe_model, vol_model, mape_model
    except Exception as e:
        st.error(f"Gagal Load Asset: {e}")
        st.stop()

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
        except:
            continue
    return pd.concat(all_dfs, ignore_index=True)

# --- 2. ENGINE RECURSIVE (STRICT MODE) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    # Ambil data kategori & buat fitur dasar
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').copy()
    hist['Waktu Pesanan Dibuat'] = pd.to_datetime(hist['Waktu Pesanan Dibuat'])
    
    # Buat Fitur (Sesuai Logika Training)
    hist['day_sin'] = np.sin(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['day_cos'] = np.cos(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['lag_1'] = np.log1p(hist['Net_Sales'].shift(1))
    hist['lag_7'] = hist['Net_Sales'].shift(7)
    hist['lag_28'] = hist['Net_Sales'].shift(28)
    hist['rolling_mean_7'] = hist['Net_Sales'].rolling(window=7).mean()
    
    # Gabungin daftar kolom yang diminta Model
    all_req_cols = meta['features'] + meta['kat_cols']
    
    # --- PROSES SINKRONISASI KOLOM (PENTING!) ---
    for col in all_req_cols:
        if col not in hist.columns:
            # Jika kolom kategori, set 1 jika nama kategori ada di dalam nama kolom
            if "Kategori" in col:
                hist[col] = 1 if kat.lower() in col.lower() else 0
            else:
                hist[col] = 0
                
    hist = hist.fillna(0)
    hist_input = hist.tail(time_steps)
    
    # Ambil data dengan URUTAN SAKLEK sesuai meta
    try:
        curr_win = hist_input[all_req_cols].values.tolist()
    except KeyError as e:
        st.error(f"Kolom ini ilang dari DataFrame: {e}")
        st.write("Kolom yang ada:", list(hist_input.columns))
        st.stop()

    # Siapkan One-Hot untuk feedback loop
    kat_onehot_vals = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]
    last_date = hist_input['Waktu Pesanan Dibuat'].max()
    
    preds = []
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]])
        
        # Predict
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
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31),
            np.log1p(val),
            curr_win[-7][2] if len(curr_win)>=7 else 0,
            curr_win[-28][2] if len(curr_win)>=28 else 0,
            np.mean([r[2] for r in curr_win[-7:]])
        ]
        new_row.extend(kat_onehot_vals)
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
    
    fig = go.Figure()
    f_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    fig.add_trace(go.Scatter(x=f_dates, y=daily_preds, name="Forecast", line=dict(color='orange', width=3)))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
