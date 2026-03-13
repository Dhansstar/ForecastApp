import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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
        st.error("Data 'data_from_DE.csv' tidak ditemukan!")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    # Data Cleaning & Type Casting
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
    # --- HEADER DENGAN ANIMASI ---
    st.markdown('<div id="text-split"><h2 class="animate-header">📊 MARKET DEMAND ANALYSIS</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 25px;">
    Eksplorasi mendalam terhadap dataset e-commerce untuk memahami tren penjualan, distribusi produk, perilaku konsumen, dan efisiensi logistik.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Mengambil data lengkap..."):
        df = load_data()

    if df.empty: return

    # ==============================
    # 1. PRODUCT & VOLUME DISTRIBUTION
    # ==============================
    st.markdown('<h2 class="animate-header">🔍 Product Distribution Analysis</h2>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 2])
    product_count = df['Kategori Produk'].value_counts().reset_index()
    product_count.columns = ['Kategori Produk', 'Total Order']
    
    with col_a:
        st.write("### Volume by Category")
        st.dataframe(product_count, use_container_width=True)
    
    with col_b:
        fig_pie = px.pie(product_count, values='Total Order', names='Kategori Produk', 
                         color_discrete_sequence=COLORS, hole=0.4)
        fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#1e293b', width=2)))
        st.plotly_chart(apply_plotly_style(fig_pie), use_container_width=True)

    # ==============================
    # 2. TIME SERIES & VARIABILITY (DENGAN STD DEV)
    # ==============================
    st.markdown('<h2 class="animate-header">📈 Sales Variability & Monthly Trends</h2>', unsafe_allow_html=True)
    df['year_month'] = df['Waktu Pesanan Dibuat'].dt.to_period('M').astype(str)
    
    # Area Chart untuk komposisi bulanan
    monthly_cat = df.groupby(['year_month','Kategori Produk'])['Jumlah Terjual Bersih'].sum().reset_index()
    fig_area = px.area(monthly_cat, x='year_month', y='Jumlah Terjual Bersih', 
                       color='Kategori Produk', color_discrete_sequence=COLORS)
    st.plotly_chart(apply_plotly_style(fig_area), use_container_width=True)

    # Tabel Standar Deviasi (Sesuai aslinya)
    st.write("#### Analisis Variabilitas (Standard Deviation)")
    std_data = monthly_cat.pivot(index='year_month', columns='Kategori Produk', values='Jumlah Terjual Bersih').std().sort_values(ascending=False).reset_index()
    std_data.columns = ['Kategori Produk', 'Std Deviation']
    st.table(std_data)

    # ==============================
    # 3. GROWTH METRICS (MoM & ROLLING)
    # ==============================
    st.markdown('<h2 class="animate-header">🚀 Growth & Momentum</h2>', unsafe_allow_html=True)
    monthly_total = df.groupby('year_month')['Jumlah Terjual Bersih'].sum().reset_index()
    monthly_total['MoM_growth'] = monthly_total['Jumlah Terjual Bersih'].pct_change() * 100
    monthly_total['Rolling_Avg'] = monthly_total['Jumlah Terjual Bersih'].rolling(3).mean()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("#### Month-over-Month Growth (%)")
        fig_mom = px.bar(monthly_total, x='year_month', y='MoM_growth', 
                         color='MoM_growth', color_continuous_scale='RdYlGn')
        st.plotly_chart(apply_plotly_style(fig_mom), use_container_width=True)
    
    with col_m2:
        st.write("#### 3-Month Rolling Average")
        fig_roll = px.line(monthly_total, x='year_month', y='Rolling_Avg', markers=True)
        fig_roll.update_traces(line_color='#a855f7', line_width=4)
        st.plotly_chart(apply_plotly_style(fig_roll), use_container_width=True)

    # ==============================
    # 4. FINANCIAL ANALYSIS (REVENUE BREAKDOWN)
    # ==============================
    st.markdown('<h2 class="animate-header">💰 Financial Breakdown</h2>', unsafe_allow_html=True)
    df['Gross_Price'] = df['Total Pembayaran'] + df['Total Diskon']
    
    fin_sum = df.groupby('Kategori Produk').agg(
        Gross_Revenue=('Gross_Price','sum'),
        Net_Revenue=('Total Pembayaran','sum'),
        Total_Discount=('Total Diskon','sum')
    ).sort_values('Gross_Revenue', ascending=False).reset_index()

    st.dataframe(fin_sum.style.format(precision=0), use_container_width=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=fin_sum['Kategori Produk'], y=fin_sum['Gross_Revenue'], name='Gross', marker_color='#64748b'))
    fig_rev.add_trace(go.Bar(x=fin_sum['Kategori Produk'], y=fin_sum['Net_Revenue'], name='Net (Received)', marker_color='#3b82f6'))
    fig_rev.update_layout(barmode='group')
    st.plotly_chart(apply_plotly_style(fig_rev), use_container_width=True)

    # ==============================
    # 5. LOGISTICS & REGIONAL INSIGHTS
    # ==============================
    st.markdown('<h2 class="animate-header">🚚 Logistics & Region Distribution</h2>', unsafe_allow_html=True)
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.write("#### Top 10 Provinces by Volume")
        prov_data = df.groupby('Provinsi')['Jumlah'].sum().nlargest(10).reset_index()
        fig_prov = px.bar(prov_data, x='Jumlah', y='Provinsi', orientation='h', color='Jumlah', color_continuous_scale='Viridis')
        st.plotly_chart(apply_plotly_style(fig_prov), use_container_width=True)
    
    with col_l2:
        st.write("#### Average Shipping Cost by Category")
        ship_data = df.groupby('Kategori Produk')['Ongkos Kirim Dibayar oleh Pembeli'].mean().sort_values().reset_index()
        fig_ship = px.bar(ship_data, x='Ongkos Kirim Dibayar oleh Pembeli', y='Kategori Produk', orientation='h', marker_color='#f97316')
        st.plotly_chart(apply_plotly_style(fig_ship), use_container_width=True)

    # ==============================
    # 6. CONSUMER BEHAVIOR (PAYMENT & WEEKEND)
    # ==============================
    st.markdown('<h2 class="animate-header">💳 Consumer Behavior</h2>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Payment Method Usage")
        pay_count = df['Metode Pembayaran'].value_counts().reset_index()
        fig_pay = px.bar(pay_count, x='count', y='Metode Pembayaran', orientation='h', color='count')
        st.plotly_chart(apply_plotly_style(fig_pay), use_container_width=True)
    
    with c2:
        st.write("#### Weekend vs Weekday Sales")
        wk_data = df.groupby('Weekend')['Jumlah Terjual Bersih'].sum().reset_index()
        fig_wk = px.pie(wk_data, values='Jumlah Terjual Bersih', names='Weekend', color_discrete_sequence=['#3b82f6','#ec4899'])
        st.plotly_chart(apply_plotly_style(fig_wk), use_container_width=True)

    # ==============================
    # SUMMARY & RECOMMENDATIONS
    # ==============================
    st.divider()
    st.markdown('<h2 class="animate-header">🏛️ Strategic Insights</h2>', unsafe_allow_html=True)
    
    with st.expander("LIHAT KESIMPULAN ANALISIS", expanded=True):
        st.info("""
        - **Dominasi Produk:** Kategori Kitchen & Home Living menyumbang porsi terbesar volume transaksi.
        - **Variabilitas:** Fluktuasi bulanan tertinggi terjadi pada kategori Kitchen, menandakan sensitivitas tinggi terhadap promo.
        - **Logistik:** Biaya pengiriman rata-rata berbanding lurus dengan jarak provinsi; wilayah luar Jawa memerlukan optimasi subsidi ongkir.
        - **Metode Bayar:** COD masih sangat dominan, menunjukkan profil user yang mengutamakan keamanan transaksi fisik.
        """)

    # --- TRIGGER SQUARE UNTUK ANIME.JS ---
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 30px;"><div class="square" style="width: 50px; height: 50px; background: #3b82f6; border-radius: 12px; box-shadow: 0 10px 20px rgba(59,130,246,0.3);"></div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run()