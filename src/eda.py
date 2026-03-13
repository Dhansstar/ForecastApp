import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        st.error("Data 'data_from_DE.csv' tidak ditemukan!")
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
    # --- HEADER ANIMASI (STRUKTUR IDENTIK DENGAN PREDICTION.PY) ---
    st.markdown('<div id="text-split"><h2 class="animate-header">📊 MARKET DEMAND ANALYSIS</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 25px;">
    Analisis ini bertujuan untuk memahami pola permintaan penjualan pada e-commerce sales dataset melalui eksplorasi tren penjualan, distribusi produk, perilaku pembelian konsumen, serta faktor operasional yang mempengaruhi transaksi.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading full dataset..."):
        df = load_data()

    if df.empty: return

    # --- INTERNAL EDA STATS (Processing) ---
    buffer = io.StringIO()
    df.info(buf=buffer)
    dataframe_info = buffer.getvalue()
    duplicate_count = df.duplicated().sum()
    missing_values = df.isnull().sum()

    # ==============================
    # 1. PRODUCT DISTRIBUTION
    # ==============================
    st.markdown('<h2 class="animate-header">🔍 Inspecting Product Distribution</h2>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 2])
    product_count = df['Kategori Produk'].value_counts().reset_index()
    product_count.columns = ['Kategori Produk', 'Total']
    
    with col_a:
        st.write("### Total Items by Category")
        st.dataframe(product_count, use_container_width=True)
    
    with col_b:
        fig_pie = px.pie(product_count, values='Total', names='Kategori Produk', 
                         color_discrete_sequence=COLORS, hole=0.4)
        fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#1e293b', width=2)))
        st.plotly_chart(apply_plotly_style(fig_pie), use_container_width=True)

    st.markdown("""
    - **Kitchen & Dining (36.5%)** dan **Home Organization & Living (33.1%)** mendominasi penjualan (>70%).
    - Demand sangat terkonsentrasi pada produk kebutuhan rumah tangga.
    """)

    # ==============================
    # 2. MONTHLY SALES DISTRIBUTION
    # ==============================
    st.markdown('<h2 class="animate-header">1. Monthly Sales Distribution</h2>', unsafe_allow_html=True)
    df['year_month'] = df['Waktu Pesanan Dibuat'].dt.to_period('M').astype(str)
    
    monthly_category = df.groupby(['year_month','Kategori Produk'])['Jumlah Terjual Bersih'].sum().reset_index()
    
    fig_area = px.area(monthly_category, x='year_month', y='Jumlah Terjual Bersih', 
                       color='Kategori Produk', color_discrete_sequence=COLORS)
    st.plotly_chart(apply_plotly_style(fig_area), use_container_width=True)

    st.write("#### Category Variability (Standard Deviation)")
    st.write(monthly_category.pivot(index='year_month', columns='Kategori Produk', values='Jumlah Terjual Bersih').std().sort_values(ascending=False))
    
    st.info("- **Kitchen & Dining memiliki variabilitas permintaan tertinggi**.")

    # ==============================
    # 3. MONTHLY SALES TREND & GROWTH
    # ==============================
    st.markdown('<h2 class="animate-header">2. Monthly Sales Trend and Growth</h2>', unsafe_allow_html=True)
    monthly_total = df.groupby('year_month')['Jumlah Terjual Bersih'].sum().reset_index()
    monthly_total['MoM_growth_pct'] = monthly_total['Jumlah Terjual Bersih'].pct_change() * 100
    monthly_total['Rolling_3M'] = monthly_total['Jumlah Terjual Bersih'].rolling(3).mean()

    # Trend Line
    fig_line = px.line(monthly_total, x='year_month', y='Jumlah Terjual Bersih', markers=True)
    fig_line.update_traces(line=dict(width=3, color='#3b82f6'))
    st.plotly_chart(apply_plotly_style(fig_line), use_container_width=True)

    # MoM Growth & Rolling
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("#### Month-over-Month (MoM) Growth")
        fig_mom = px.bar(monthly_total, x='year_month', y='MoM_growth_pct', 
                         color='MoM_growth_pct', color_continuous_scale='RdYlGn')
        st.plotly_chart(apply_plotly_style(fig_mom), use_container_width=True)
    with col_g2:
        st.write("#### Rolling 3-Month Average")
        fig_roll = px.bar(monthly_total, x='year_month', y='Rolling_3M')
        fig_roll.update_traces(marker_color='#a855f7')
        st.plotly_chart(apply_plotly_style(fig_roll), use_container_width=True)

    st.markdown("- Lonjakan penjualan kemungkinan besar dipicu oleh **event-driven demand**.")

    # ==============================
    # 4. REVENUE ANALYSIS
    # ==============================
    st.markdown('<h2 class="animate-header">3. Gross and Net Revenue</h2>', unsafe_allow_html=True)
    df['Total Harga'] = df[['Total Pembayaran','Total Diskon','Ongkos Kirim Dibayar oleh Pembeli']].sum(axis=1)
    
    rev_sum = df.groupby('Kategori Produk').agg(
        Gross_Revenue=('Total Harga','sum'),
        Net_Revenue=('Total Pembayaran','sum')
    ).sort_values('Gross_Revenue', ascending=False).reset_index()
    rev_sum['Cost_Revenue'] = rev_sum['Gross_Revenue'] - rev_sum['Net_Revenue']

    st.dataframe(rev_sum, use_container_width=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=rev_sum['Kategori Produk'], y=rev_sum['Gross_Revenue'], name='Gross', marker_color='#64748b'))
    fig_rev.add_trace(go.Bar(x=rev_sum['Kategori Produk'], y=rev_sum['Net_Revenue'], name='Net', marker_color='#3b82f6'))
    fig_rev.add_trace(go.Bar(x=rev_sum['Kategori Produk'], y=rev_sum['Cost_Revenue'], name='Cost', marker_color='#ec4899'))
    fig_rev.update_layout(barmode='group')
    st.plotly_chart(apply_plotly_style(fig_rev), use_container_width=True)

    # ==============================
    # 5. WEEKDAY VS WEEKEND & PROVINCES
    # ==============================
    st.markdown('<h2 class="animate-header">4. Operations & Logistics Analysis</h2>', unsafe_allow_html=True)
    
    col_op1, col_op2 = st.columns(2)
    with col_op1:
        st.write("#### Weekday vs Weekend Sales")
        wk_sales = df.groupby(['Weekend','Kategori Produk'])['Jumlah Terjual Bersih'].sum().reset_index()
        fig_wk = px.bar(wk_sales, x='Kategori Produk', y='Jumlah Terjual Bersih', color='Weekend', barmode='group')
        st.plotly_chart(apply_plotly_style(fig_wk), use_container_width=True)
    
    with col_op2:
        st.write("#### Sales by Top 10 Provinces")
        prov_sales = df.groupby('Provinsi')['Jumlah'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prov = px.bar(prov_sales, x='Jumlah', y='Provinsi', orientation='h', color='Jumlah', color_continuous_scale='Blues')
        st.plotly_chart(apply_plotly_style(fig_prov), use_container_width=True)

    # Shipping Cost
    st.write("#### Average Shipping Cost by Province (Top 15)")
    ship_prov = df.groupby('Provinsi')['Ongkos Kirim Dibayar oleh Pembeli'].mean().sort_values(ascending=False).head(15).reset_index()
    fig_ship = px.bar(ship_prov, x='Ongkos Kirim Dibayar oleh Pembeli', y='Provinsi', orientation='h')
    fig_ship.update_traces(marker_color='#f97316')
    st.plotly_chart(apply_plotly_style(fig_ship), use_container_width=True)

    # ==============================
    # 6. PAYMENT METHODS
    # ==============================
    st.markdown('<h2 class="animate-header">5. Payment Methods Analysis</h2>', unsafe_allow_html=True)
    pay_sum = df.groupby('Metode Pembayaran').agg(
        Qty=('Jumlah','sum'), Rev=('Total Pembayaran','sum')
    ).sort_values('Qty', ascending=False).head(5).reset_index()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig1 = px.bar(pay_sum, x='Qty', y='Metode Pembayaran', orientation='h', title="By Quantity")
        st.plotly_chart(apply_plotly_style(fig1), use_container_width=True)
    with col_p2:
        fig2 = px.bar(pay_sum, x='Rev', y='Metode Pembayaran', orientation='h', title="By Revenue")
        st.plotly_chart(apply_plotly_style(fig2), use_container_width=True)

    # ==============================
    # FINAL EXECUTIVE SUMMARY
    # ==============================
    st.divider()
    st.markdown('<h2 class="animate-header">🏛️ Executive Summary</h2>', unsafe_allow_html=True)
    
    st.write("""
    Analisis ini bertujuan untuk memahami pola permintaan penjualan pada e-commerce sales dataset melalui eksplorasi tren penjualan, 
    distribusi produk, perilaku pembelian konsumen, serta faktor operasional yang mempengaruhi transaksi.
    """)
    
    st.subheader("Key Findings")
    st.markdown("""
    1. **Penjualan Terkonsentrasi:** Kitchen & Dining mendominasi volume transaksi.
    2. **Event-Driven:** Permintaan bersifat fluktuatif mengikuti momentum promosi.
    3. **Dominasi COD:** Metode tunai masih menjadi preferensi utama pelanggan.
    4. **Disparitas Logistik:** Ongkir luar Jawa sangat tinggi, mempengaruhi margin keuntungan.
    5. **Peak Time:** Aktivitas belanja lebih tinggi pada hari kerja (*weekdays*).
    """)

    # --- RECOMMENDATIONS ---
    sections = [
        ("1. Demand Overview", "Secara umum, penjualan didominasi Kitchen & Dining dan Home Organization yang menyumbang lebih dari 70% total transaksi."),
        ("2. Demand Pattern", "Tren bulanan menunjukkan fluktuasi tajam yang mengindikasikan sensitivitas pembeli terhadap periode promosi."),
        ("3. Demand Drivers", "Sistem COD dan faktor biaya kirim menjadi pendorong utama keputusan pembelian di berbagai wilayah."),
        ("4. Operational Factors", "Return rate yang rendah menunjukkan kualitas produk yang baik, namun pembatalan sering terjadi di fase checkout.")
    ]
    for title, content in sections:
        st.header(title)
        st.write(content)

    st.markdown('<h2 class="animate-header">💡 Business Recommendations</h2>', unsafe_allow_html=True)
    recs = [
        ("🚀 Fokus Core Product", "Kitchen & Dining sebagai core product line harus mendapatkan prioritas stok dan strategi bundling."),
        ("📅 Momentum Promosi", "Selaraskan manajemen inventory dengan event-driven demand agar tidak terjadi stockout."),
        ("💳 Digital Adoption", "Berikan promo khusus untuk pembayaran digital guna mengurangi resiko gagal bayar pada sistem COD."),
        ("📦 Warehouse Strategy", "Optimalisasi pengiriman luar Jawa untuk mengurangi beban ongkir pembeli."),
        ("📉 Quality Control", "Pertahankan return rate rendah dengan perbaikan deskripsi produk berkelanjutan."),
        ("🛒 Checkout Flow", "Sederhanakan proses konfirmasi pesanan untuk mengurangi angka pembatalan.")
    ]
    for r_title, r_desc in recs:
        st.success(f"**{r_title}**: {r_desc}")

    # Trigger Anime.js Square
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 20px;"><div class="square" style="width: 40px; height: 40px; background: #3b82f6; border-radius: 8px;"></div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run()