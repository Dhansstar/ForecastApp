import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# --- 1. CONFIG & ASSETS ---
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

# --- 2. ENGINE RECURSIVE (FIX SHAPE MISMATCH) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    
    # Filter kategori
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').copy()
    
    # Feature Engineering Dasar
    hist['day_sin'] = np.sin(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['day_cos'] = np.cos(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['lag_1'] = np.log1p(hist['Net_Sales'].shift(1))
    hist['lag_7'] = hist['Net_Sales'].shift(7)
    hist['lag_28'] = hist['Net_Sales'].shift(28)
    hist['rolling_mean_7'] = hist['Net_Sales'].rolling(window=7).mean()
    
    # Sinkronisasi Kolom Meta
    all_req_cols = meta['features'] + meta['kat_cols']
    for col in all_req_cols:
        if col not in hist.columns:
            hist[col] = (1 if kat.lower() in col.lower() else 0) if "Kategori" in col else 0

    hist = hist.fillna(0)

    # --- THE ULTIMATE FIX FOR VALUEERROR ---
    # Jika data lo kurang dari time_steps model (misal cuma ada 20 baris padahal model minta 30)
    if len(hist) < time_steps:
        needed = time_steps - len(hist)
        # Bikin padding baris nol agar shape-nya PAS
        padding_df = pd.DataFrame(0, index=range(needed), columns=hist.columns)
        hist = pd.concat([padding_df, hist], ignore_index=True)
    
    hist_input = hist.tail(time_steps)
    curr_win = hist_input[all_req_cols].values.tolist()
    
    # Pastikan jumlah kolom pas dengan input_shape model
    expected_f = fe_model.input_shape[-1]
    last_date = pd.to_datetime(hist_input['Waktu Pesanan Dibuat'].iloc[-1]) if not pd.isnull(hist_input['Waktu Pesanan Dibuat'].iloc[-1]) else pd.Timestamp.now()
    kat_onehot_vals = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]

    preds = []
    for i in range(1, 31):
        # Slice window agar kolomnya presisi
        X_in = np.array([curr_win[-time_steps:]])
        X_in = X_in[:, :, :expected_f] 

        # Prediksi Latent Feature
        lat = fe_model.predict(X_in, verbose=0)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        
        # Gabung prediksi volume & mape
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        # Update Window untuk recursive t+1
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [np.sin(2*np.pi*nxt_d.day/31), np.cos(2*np.pi*nxt_d.day/31), np.log1p(val),
                   curr_win[-7][2] if len(curr_win)>=7 else 0,
                   curr_win[-28][2] if len(curr_win)>=28 else 0,
                   np.mean([r[2] for r in curr_win[-7:]])]
        
        feedback = (new_row + kat_onehot_vals)[:expected_f]
        curr_win.append(feedback)
        
    if recipe['smooth']:
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        preds = np.floor(preds + 0.3)
        
    return preds, int(np.ceil(np.sum(preds) * recipe['mult'])), last_date, hist.tail(30)

# --- 3. UI (MENU PREDICTION - BUKAN SIDEBAR) ---
def run_prediction():
    meta, fe_model, vol_model, mape_model = load_assets()
    full_df = load_and_sync_data()

    # Model Performance Box (Identik Gambar)
    st.markdown("""
        <div style="background-color:#26324a; padding:20px; border-radius:10px; margin-bottom:20px">
            <h4 style="color:#4a90e2; margin:0">Model Performance: <span style="color:white">Total Volume Accuracy: 95.98%</span></h4>
        </div>
    """, unsafe_allow_html=True)

    # Pilih Kategori di Halaman Utama
    st.write("### Pilih Kategori")
    selected_kat = st.selectbox("", list(meta['final_recipes'].keys()), label_visibility="collapsed")

    daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
        selected_kat, meta, fe_model, vol_model, mape_model, full_df
    )
    
    # --- VISUALISASI IDENTIK GAMBAR ---
    st.write(f"## DemandSense: **{selected_kat}**")
    
    forecast_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
    fig = go.Figure()
    
    # Historis (Line Solid)
    fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], 
                             name='Historis (30 Hari)', line=dict(color='#1f77b4', width=3), mode='lines+markers'))
    # Forecast (Dashed Line)
    fig.add_trace(go.Scatter(x=f_dates, y=daily_preds, 
                             name='Forecast (30 Hari)', line=dict(color='#ff7f0e', width=3, dash='dash'), mode='lines+markers'))
    
    # Prediction Zone Arsir
    fig.add_vrect(x0=f_dates[0], x1=f_dates[-1], fillcolor="orange", opacity=0.1, layer="below", line_width=0,
                  annotation_text="PREDICTION ZONE", annotation_position="top left")
    
    # Total Stok Annotation
    fig.add_annotation(x=0.05, y=0.9, xref="paper", yref="paper", text=f"Total Estimasi Stok: {total_stok} Unit",
                       showarrow=False, font=dict(color="white", size=14), bgcolor="#ff7f0e", borderpad=10)

    fig.update_layout(template="plotly_white", xaxis_title="Waktu", yaxis_title="Jumlah Unit Terjual",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_prediction()
