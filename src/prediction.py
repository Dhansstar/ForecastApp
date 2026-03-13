import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os
import time

# --- 1. ASSETS LOADING ---
BASE_PATH = os.path.dirname(__file__)

@st.cache_resource
def load_assets():
    try:
        with open(os.path.join(BASE_PATH, "model_metadata.json"), 'r') as f:
            meta = json.load(f)
        fe_model = load_model(os.path.join(BASE_PATH, "feature_extractor.keras"), compile=False)
        vol_model = joblib.load(os.path.join(BASE_PATH, "xgb_vol_model.joblib"))
        mape_model = joblib.load(os.path.join(BASE_PATH, "xgb_mape_model.joblib"))
        return meta, fe_model, vol_model, mape_model
    except Exception as e:
        st.error(f"Gagal Load Assets: {e}")
        st.stop()

# --- 2. ENGINE RECURSIVE (OPTIMIZED FOR BUSINESS SENSE) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    expected_f = fe_model.input_shape[-1]
    
    # Prep data historis terakhir sesuai EDA (fokus pada tren)
    hist = full_df[full_df['Kategori'] == kat].sort_values('Waktu Pesanan Dibuat').copy()
    hist['day_sin'] = np.sin(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['day_cos'] = np.cos(2 * np.pi * hist['Waktu Pesanan Dibuat'].dt.day / 31)
    hist['lag_1'] = np.log1p(hist['Net_Sales'].shift(1))
    hist['lag_7'] = hist['Net_Sales'].shift(7)
    hist['lag_28'] = hist['Net_Sales'].shift(28)
    hist['rolling_mean_7'] = hist['Net_Sales'].rolling(window=7).mean()
    
    all_req_cols = meta['features'] + meta['kat_cols']
    for col in all_req_cols:
        if col not in hist.columns:
            hist[col] = (1 if kat.lower() in col.lower() else 0) if "Kategori" in col else 0
    
    hist = hist.fillna(0)
    hist_input = hist.tail(time_steps)
    
    # Window initialization
    curr_win = [row[:expected_f] for row in hist_input[all_req_cols].values.tolist()]
    last_date = pd.to_datetime(hist_input['Waktu Pesanan Dibuat'].iloc[-1])
    kat_onehot_vals = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]

    daily_preds = []
    
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]], dtype='float32')
        lat = fe_model.predict(X_in, verbose=0)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        
        # Hybrid Calculation - Menjaga sensitivitas terhadap fluktuasi EDA
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        
        # Booster Trend: Menjaga momentum agar tidak dropping di minggu ke-3 & 4
        if i > 7:
            val = val * 1.05 
            
        val = max(0, val) if val >= recipe['thresh'] else 0
        
        # Pembulatan cerdas sesuai dominasi kategori di EDA
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        
        daily_preds.append(val)
        
        # Recursive feedback loop
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*np.pi*nxt_d.day/31), 
            np.cos(2*np.pi*nxt_d.day/31), 
            np.log1p(val), 
            curr_win[-7][2] if len(curr_win)>=7 else 0,
            curr_win[-28][2] if len(curr_win)>=28 else 0,
            np.mean([r[2] for r in curr_win[-7:]]) 
        ]
        curr_win.append((new_row + kat_onehot_vals)[:expected_f])
        
    total_stok = int(np.ceil(np.sum(daily_preds) * recipe['mult']))
    return daily_preds, total_stok, last_date, hist.tail(30)

# --- 3. UI RENDERING ---
def run():
    # Load CSS External & Force Reset Kotak Hantu
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Reset total margin/padding hantu */
    .element-container { margin-bottom: 0px !important; }
    div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
    
    /* Fokus border hanya pada chart dan input wrapper */
    .stPlotlyChart, .input-wrapper {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        margin-top: 15px !important;
    }
    
    label[data-testid="stWidgetLabel"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div id="text-split"><h2 class="animate-header">🚀 DEMANDSENSE</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><strong>Market Demand & Sales Analysis:</strong> Mentransformasi insight EDA menjadi prediksi stok presisi.</div>', unsafe_allow_html=True)

    meta, fe_model, vol_model, mape_model = load_assets()
    
    # CSV Data Handling
    categories_files = {'Kitchen': 'forecast_kitchen_data.csv', 'Home': 'forecast_home_data.csv', 
                        'Tools': 'forecast_tools_data.csv', 'Bathroom': 'forecast_bathroom_data.csv',
                        'Storage': 'forecast_storage_data.csv', 'Other': 'forecast_other_data.csv'}
    all_dfs = []
    for kat, fname in categories_files.items():
        p = os.path.join(BASE_PATH, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Kategori'] = kat
            if 'Jumlah Terjual Bersih' in df.columns:
                df = df.rename(columns={'Jumlah Terjual Bersih': 'Net_Sales'})
            all_dfs.append(df)
    
    if not all_dfs:
        st.error("Data sumber (.csv) tidak ditemukan!")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])

    # --- UI INPUT SECTION ---
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-weight: 600; margin-bottom: 10px;">Pilih Kategori Prioritas EDA</p>', unsafe_allow_html=True)
    
    selected_kat = st.selectbox("pilih", list(meta['final_recipes'].keys()), label_visibility="collapsed")
    
    st.markdown('<div style="display: flex; justify-content: center; margin: 15px 0;"><div class="square" style="width: 32px; height: 32px; background: linear-gradient(45deg, #3b82f6, #ec4899); border-radius: 8px;"></div></div>', unsafe_allow_html=True)
    
    run_btn = st.button("Generate Demand Forecast 🔮", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- EXECUTION ---
    if run_btn:
        with st.status(f"AI sedang memproses data {selected_kat}...", expanded=False):
            daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
                selected_kat, meta, fe_model, vol_model, mape_model, full_df
            )
        
        # Forecast Visualization
        f_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], name='Historis (EDA)', line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=f_dates, y=daily_preds, name='AI Forecast', line=dict(color='#f97316', width=3, dash='dash')))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=400,
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig, use_container_width=True)

        # Business Summary Result
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid #3b82f6;">
            <h4 style="margin:0;">Hasil Prediksi Kebutuhan Stok</h4>
            <p style="font-size: 24px; color: #3b82f6; font-weight: bold; margin: 10px 0;">{total_stok} Unit</p>
            <p style="font-size: 14px; color: #94a3b8;">Berdasarkan tren 30 hari ke depan untuk kategori <strong>{selected_kat}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    # Animation Script
    st.markdown("<script>anime({targets: '.input-wrapper, .glass-card, .square', translateY: [15, 0], opacity: [0, 1], delay: anime.stagger(100), easing: 'easeOutExpo', duration: 800});</script>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()