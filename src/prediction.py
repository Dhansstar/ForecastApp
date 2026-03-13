import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os
import matplotlib.pyplot as plt

# --- 1. ASSETS LOADING ---
BASE_PATH = "src/" 

@st.cache_resource
def load_assets():
    try:
        with open(f"{BASE_PATH}model_metadata.json", 'r') as f:
            meta = json.load(f)
        
        model_path = f"{BASE_PATH}feature_extractor.keras"
        fe_model = load_model(model_path, compile=False)
        vol_model = joblib.load(f"{BASE_PATH}xgb_vol_model.joblib")
        mape_model = joblib.load(f"{BASE_PATH}xgb_mape_model.joblib")
        
        return meta, fe_model, vol_model, mape_model
    except Exception as e:
        st.error(f"Gagal Load Assets: {e}")
        st.stop()

# --- 2. ENGINE RECURSIVE (FIXED JAGGED ARRAY) ---
def run_recursive_forecast(kat, meta, fe_model, vol_model, mape_model, full_df):
    recipe = meta['final_recipes'][kat]
    time_steps = meta['time_steps']
    expected_f = fe_model.input_shape[-1] 
    
    # Prep Data
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
    # Header Spesifik sesuai style.css lo
    st.markdown('<div id="text-split"><h2 class="text-xl">🔮 AI DEMAND FORECASTING</h2></div>', unsafe_allow_html=True)
    
    meta, fe_model, vol_model, mape_model = load_assets()
    
    # Load Data (Pastikan CSV ada di src/)
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
    
    if not all_dfs:
        st.warning("Data CSV tidak ditemukan di folder src/.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Waktu Pesanan Dibuat'] = pd.to_datetime(full_df['Waktu Pesanan Dibuat'])

    st.write("### Pilih Kategori Produk")
    selected_kat = st.selectbox("", list(meta['final_recipes'].keys()), label_visibility="collapsed")

    if st.button("Run Prediction"):
        with st.spinner(f"AI sedang menganalisis kategori {selected_kat}..."):
            daily_preds, total_stok, last_dt, hist_30 = run_recursive_forecast(
                selected_kat, meta, fe_model, vol_model, mape_model, full_df
            )
            
            # 1. CHART VISUALIZATION
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

            # 2. GRADIENT SUMMARY TABLE (Tampilan Pro)
            st.markdown("### 📦 Ringkasan Kebutuhan Stok")
            summary_df = pd.DataFrame([{
                'Kategori': selected_kat,
                'Rata-rata/Hari': f"{np.mean(daily_preds):.2f} unit",
                'Puncak Permintaan': f"{int(np.max(daily_preds))} unit",
                'Total Stok (30 Hari)': f"{total_stok} unit"
            }])

            fig_tbl, ax = plt.subplots(figsize=(10, 2))
            ax.axis('off')
            tbl = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            
            tbl.auto_set_font_size(False); tbl.set_fontsize(11)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor('#cbd5e1')
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#1e293b')
                else:
                    cell.set_facecolor('#f1f5f9') # Light background for readability
            
            st.pyplot(fig_tbl)
            st.success(f"Prediksi selesai. Total stok yang disarankan: {total_stok} unit.")

if __name__ == "__main__":
    run()