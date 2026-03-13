import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os
import time
import matplotlib.pyplot as plt

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

# --- 2. ENGINE RECURSIVE (AI LOGIC) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    expected_f = fe_model.input_shape[-1] 
    
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
    if len(hist) < time_steps:
        pad = pd.DataFrame(0, index=range(time_steps - len(hist)), columns=hist.columns)
        hist = pd.concat([pad, hist], ignore_index=True)
    
    hist_input = hist.tail(time_steps)
    curr_win = [row[:expected_f] for row in hist_input[all_req_cols].values.tolist()]
    
    last_date = pd.to_datetime(hist_input['Waktu Pesanan Dibuat'].iloc[-1])
    kat_onehot_vals = [1 if kat.lower() in c.lower() else 0 for c in meta['kat_cols']]

    preds = []
    for i in range(1, 31):
        X_in = np.array([curr_win[-time_steps:]], dtype='float32')
        lat = fe_model.predict(X_in, verbose=0)
        p_v = vol_model.predict(lat)[0]
        p_m = mape_model.predict(lat)[0]
        
        val = (recipe['w_vol'] * p_v) + (recipe['w_mape'] * p_m)
        val = max(0, val) if val >= recipe['thresh'] else 0
        if not recipe['smooth']:
            val = np.ceil(val) if kat in ['Kitchen', 'Home'] else np.round(val)
        preds.append(val)
        
        nxt_d = last_date + pd.Timedelta(days=i)
        new_row = [
            np.sin(2*nxt_d.day*np.pi/31), np.cos(2*nxt_d.day*np.pi/31), np.log1p(val),
            curr_win[-7][2] if len(curr_win)>=7 else 0,
            curr_win[-28][2] if len(curr_win)>=28 else 0,
            np.mean([r[2] for r in curr_win[-7:]])
        ]
        combined = (new_row + kat_onehot_vals)[:expected_f]
        curr_win.append(combined)
        
    return preds, int(np.ceil(np.sum(preds) * recipe['mult'])), last_date, hist.tail(30)

# --- 3. UI RENDERING ---
def run():
    # Injeksi CSS (Tanpa margin aneh)
    st.markdown("""
    <style>
    .input-container {
        padding: 25px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }
    div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2) !important;
    }
    /* Ngilangin gap kosong yang dihasilkan Markdown */
    .stMarkdown div p { margin-bottom: 0px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div id="text-split"><h2 class="animate-header">🔮 AI DEMAND FORECASTING</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card animate-card">
        Prediksi stok 30 hari ke depan menggunakan <strong>Hybrid LSTM-XGBoost</strong>.
    </div>
    """, unsafe_allow_html=True)

    meta, fe_model, vol_model, mape_model = load_assets()
    
    # Load Data
    categories_files = {
        'Kitchen': 'forecast_kitchen_data.csv', 'Home': 'forecast_home_data.csv',
        'Tools': 'forecast_tools_data.csv', 'Bathroom': 'forecast_bathroom_data.csv',
        'Storage': 'forecast_storage_data.csv', 'Other': 'forecast_other_data.csv'
    }
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
        st.warning("Data CSV tidak ditemukan.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])

    # --- WRAPPER INPUT (Biar Rapi & Kotak Kosong Ilang) ---
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        st.write("#### Pilih Kategori Produk")
        selected_kat = st.selectbox("", list(meta['final_recipes'].keys()), label_visibility="collapsed")
        
        # Animasi Square (Dikecilin dikit biar proporsional)
        st.markdown('<div style="display: flex; justify-content: center; margin: 20px 0;"><div class="square" style="width: 30px; height: 30px; background: linear-gradient(45deg, #3b82f6, #ec4899); border-radius: 8px;"></div></div>', unsafe_allow_html=True)

        if st.button("Run Hybrid Prediction 🚀", use_container_width=True, type="primary"):
            with st.status(f"AI sedang menganalisis kategori {selected_kat}...", expanded=True) as status:
                daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
                    selected_kat, meta, fe_model, vol_model, mape_model, full_df
                )
                time.sleep(1)
                status.update(label="Prediction Complete!", state="complete", expanded=False)
                
                # --- PLOTLY ---
                f_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=30)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_30['Waktu Pesanan Dibuat'], y=hist_30['Net_Sales'], name='Historis', line=dict(color='#3b82f6', width=3)))
                fig.add_trace(go.Scatter(x=f_dates, y=daily_preds, name='Forecast', line=dict(color='#f97316', width=3, dash='dash')))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"), height=400, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- TABLE ---
                st.markdown('<h3>📦 Ringkasan Kebutuhan Stok</h3>', unsafe_allow_html=True)
                summary_df = pd.DataFrame([{'Kategori': selected_kat, 'Total Stok (30H)': f"{total_stok} unit"}])
                
                fig_tbl, ax = plt.subplots(figsize=(10, 2))
                fig_tbl.patch.set_alpha(0)
                ax.axis('off')
                tbl = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
                for (row, col), cell in tbl.get_celld().items():
                    cell.set_edgecolor((1, 1, 1, 0.2))
                    if row == 0:
                        cell.set_facecolor('#1e293b')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor((1, 1, 1, 0.05))
                        cell.set_text_props(color='white')
                
                st.pyplot(fig_tbl)
                st.success(f"Prediksi selesai. Stok aman: {total_stok} unit.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- SCRIPT ANIMASI ---
    st.markdown("""
    <script>
        anime({
            targets: '.input-container, .animate-card, .square',
            translateY: [15, 0],
            opacity: [0, 1],
            delay: anime.stagger(150),
            easing: 'easeOutExpo',
            duration: 1000
        });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()