import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io

# --- PALETTE & THEME ---
COLORS = ['#3b82f6', '#f97316', '#10b981', '#a855f7', '#ec4899', '#64748b', '#06b6d4']

def apply_plotly_style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color="#94a3b8")
    )
    return fig

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data_from_DE.csv")
    if not os.path.exists(data_path):
        st.error("Data 'data_from_DE.csv' tidak ditemukan di folder src/!")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    # Data Cleaning
    df['Weekend'] = df['Weekend'].replace({0:'No', 1:'Yes'})
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
    
    cols = ['Total Diskon', 'Ongkos Kirim Dibayar oleh Pembeli', 
            'Estimasi Potongan Biaya Pengiriman', 'Perkiraan Ongkos Kirim']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            
    if 'Provinsi' in df.columns:
        df['Provinsi'] = df['Provinsi'].str.replace(r'\(.*\)', '', regex=True).str.strip()
    return df

def run():
    # --- HEADER ANIMASI ---
    st.markdown('<div id="text-split"><h1 class="animate-header">📊 MARKET DEMAND ANALYSIS</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 25px;">
    Analisis ini bertujuan untuk memahami pola permintaan penjualan melalui eksplorasi tren, distribusi produk, dan perilaku konsumen pada dataset e-commerce.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Processing deep-dive analysis..."):
        df = load_data()

    if df.empty: return

    # ==============================
    # 1. PRODUCT DISTRIBUTION
    # ==============================
    st.markdown('<h2 class="animate-header">🔍 Product Distribution</h2>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 2])
    product_count = df['Kategori Produk'].value_counts().reset_index()
    product_count.columns = ['Kategori Produk', 'Total']
    
    with col_a:
        st.write("### Top Categories")
        st.dataframe(product_count, use_container_width=True)
    
    with col_b:
        fig_pie = px.pie(product_count, values='Total', names='Kategori Produk', 
                         color_discrete_sequence=COLORS, hole=0.4)
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(apply_plotly_style(fig_pie), use_container_width=True)

    st.info("**Kitchen & Dining (36.5%)** dan **Home Organization (33.1%)** mendominasi pasar.")

    # ==============================
    # 2. SALES TRENDS
    # ==============================
    st.markdown('<h2 class="animate-header">📈 Monthly Sales Trend & Growth</h2>', unsafe_allow_html=True)
    
    df['year_month'] = df['Waktu Pesanan Dibuat'].dt.to_period('M').astype(str)
    monthly_total = df.groupby('year_month')['Jumlah Terjual Bersih'].sum().reset_index()
    monthly_total['MoM_growth_pct'] = monthly_total['Jumlah Terjual Bersih'].pct_change() * 100

    tab1, tab2 = st.tabs(["Sales Volume", "Growth Rate %"])
    
    with tab1:
        fig_line = px.line(monthly_total, x='year_month', y='Jumlah Terjual Bersih', markers=True)
        fig_line.update_traces(line=dict(width=4, color='#3b82f6'))
        st.plotly_chart(apply_plotly_style(fig_line), use_container_width=True)
    
    with tab2:
        fig_mom = px.bar(monthly_total, x='year_month', y='MoM_growth_pct', 
                         color='MoM_growth_pct', color_continuous_scale='RdYlGn')
        st.plotly_chart(apply_plotly_style(fig_mom), use_container_width=True)

    # ==============================
    # 3. REVENUE & LOGISTICS
    # ==============================
    st.markdown('<h2 class="animate-header">💰 Financial & Operational Insights</h2>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("#### Weekday vs Weekend")
        wk_sales = df.groupby(['Weekend','Kategori Produk'])['Jumlah Terjual Bersih'].sum().reset_index()
        fig_wk = px.bar(wk_sales, x='Kategori Produk', y='Jumlah Terjual Bersih', color='Weekend', barmode='group')
        st.plotly_chart(apply_plotly_style(fig_wk), use_container_width=True)
        
    with c2:
        st.write("#### Top 10 Provinces by Sales")
        prov_sales = df.groupby('Provinsi')['Jumlah'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prov = px.bar(prov_sales, x='Jumlah', y='Provinsi', orientation='h', color='Jumlah')
        st.plotly_chart(apply_plotly_style(fig_prov), use_container_width=True)

    # ==============================
    # FINAL EXECUTIVE SUMMARY
    # ==============================
    st.markdown('<h2 class="animate-header">🏛️ Executive Summary</h2>', unsafe_allow_html=True)
    
    with st.expander("BACA ANALISIS LENGKAP", expanded=True):
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.subheader("Key Findings")
            st.markdown("""
            - **Dominasi Kategori:** Produk rumah tangga (Kitchen & Home) menguasai >70% volume.
            - **Pola Pembelian:** Demand bersifat *event-driven* dengan fluktuasi tajam di bulan tertentu.
            - **Metode Pembayaran:** COD masih menjadi pilihan utama konsumen.
            """)
        with col_sum2:
            st.subheader("Logistics Note")
            st.markdown("""
            - **Biaya Kirim:** Provinsi luar Jawa (Indonesia Timur) memiliki beban ongkir 3-4x lipat lebih tinggi.
            - **Waktu Transaksi:** Aktivitas belanja lebih tinggi pada *weekdays* dibanding *weekend*.
            """)

    # --- BUSINESS RECOMMENDATIONS ---
    st.markdown('<h2 class="animate-header">💡 Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    recs = [
        ("🚀 Fokus Core Product", "Kitchen & Dining sebagai penyumbang revenue terbesar harus mendapatkan prioritas stok dan promosi."),
        ("📅 Campaign Management", "Manfaatkan pola event-driven dengan menyiapkan campaign khusus pada bulan-bulan puncak."),
        ("💳 Digital Adoption", "Berikan insentif (cashback/free ongkir) untuk mendorong peralihan dari COD ke ShopeePay."),
        ("📦 Warehouse Strategy", "Pertimbangkan gudang regional di luar Jawa untuk menekan biaya logistik yang tinggi.")
    ]
    
    for r_title, r_desc in recs:
        st.success(f"**{r_title}**: {r_desc}")

    # Trigger Anime.js Square
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 20px;"><div class="square" style="width: 40px; height: 40px; background: #3b82f6; border-radius: 8px;"></div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run()