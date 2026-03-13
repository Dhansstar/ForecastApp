import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io

# --- CONFIG & THEME ---
# Warna aksen sesuai style.css lo biar sinkron
COLOR_PALETTE = ['#3b82f6', '#f97316', '#10b981', '#a855f7', '#ec4899', '#64748b']

# Function buat nerapin layout dark theme ke Plotly
def update_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', # Transparan biar nempel background GIF
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="'Inter', sans-serif"), # Font putih soft
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color="#94a3b8"),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color="#f8fafc"))
    )
    return fig

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    # Pastikan file CSV ini ada di folder yang sama
    data_path = os.path.join(base_dir, "data_from_DE.csv")
    
    if not os.path.exists(data_path):
        st.error(f"❌ File data tidak ditemukan di: {data_path}")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    df['Weekend'] = df['Weekend'].replace({0:'No', 1:'Yes'})
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])

    cols = ['Total Diskon', 'Ongkos Kirim Dibayar oleh Pembeli',
            'Estimasi Potongan Biaya Pengiriman', 'Perkiraan Ongkos Kirim']
    df[cols] = df[cols].astype('int64')
    df['Provinsi'] = df['Provinsi'].str.replace(r'\(.*\)', '', regex=True).str.strip()
    return df

# --- MAIN EDA FUNCTION ---
def run():
    # Header Spesifik sesuai style.css lo
    st.markdown('<div id="text-split"><h2 class="text-xl">📊 MARKET DEMAND & SALES ANALYSIS</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255,255,255,0.02); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 20px;">
    Eksplorasi tren penjualan, distribusi produk, dan perilaku konsumen untuk memahami dinamika permintaan sebelum proses forecasting.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Analyzing dataset..."):
        df = load_data()
    
    if df.empty: return # Stop jika data gagal load

    # Prep data untuk visualisasi
    df['year_month'] = df['Waktu Pesanan Dibuat'].dt.to_period('M').astype(str)

    # ==========================================
    # CONTAINER 1: OVERVIEW METRICS
    # ==========================================
    st.markdown("### 📈 Quick Overview")
    total_sales = df['Jumlah Terjual Bersih'].sum()
    avg_disc = df['Total Diskon'].mean()
    unique_cats = df['Kategori Produk'].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Unit Terjual", f"{total_sales:,} Unit")
    c2.metric("Rata-rata Diskon", f"Rp {avg_disc:,.0f}")
    c3.metric("Jumlah Kategori", f"{unique_cats}")
    
    st.divider()

    # ==========================================
    # CONTAINER 2: PRODUCT DISTRIBUTION (PIE)
    # ==========================================
    st.markdown("### 🍰 Product Category Proportion")
    product_count = df['Kategori Produk'].value_counts().reset_index()
    product_count.columns = ['Kategori Produk', 'Total']

    fig_pie = px.pie(product_count, values='Total', names='Kategori Produk', 
                     color_discrete_sequence=COLOR_PALETTE,
                     hole=0.5) # Bikin donut chart biar lebih modern
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', 
                          marker=dict(line=dict(color='rgba(15, 23, 42, 0.7)', width=2)))
    fig_pie = update_plotly_theme(fig_pie)
    fig_pie.update_layout(showlegend=False, title_text="Proporsi Penjualan per Kategori", title_x=0.5)
    
    st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("Lihat Insight Kategori"):
        st.markdown("""
        - **Kitchen & Dining** dan **Home Organization & Living** mendominasi penjualan (>70%).
        - Permintaan utama berasal dari produk kebutuhan rumah tangga dasar.
        """)

    # ==========================================
    # CONTAINER 3: SALES TREND & GROWTH (STACKED AREA)
    # ==========================================
    st.markdown('<div id="text-split"><h2 class="text-xl">1. Monthly Sales Trend & Growth</h2></div>', unsafe_allow_html=True)

    # Stacked Area Chart (Plotly Express)
    monthly_category = df.groupby(['year_month','Kategori Produk'])['Jumlah Terjual Bersih'].sum().reset_index()
    
    fig_area = px.area(monthly_category, x='year_month', y='Jumlah Terjual Bersih', color='Kategori Produk',
                       color_discrete_sequence=COLOR_PALETTE,
                       title="Monthly Sales Distribution by Category")
    
    fig_area = update_plotly_theme(fig_area)
    fig_area.update_layout(xaxis_title="Bulan", yaxis_title="Total Sold")
    st.plotly_chart(fig_area, use_container_width=True)

    # MoM Growth & Rolling Mean (Pake Subplots go.Figure biar pro)
    monthly_total = df.groupby('year_month')['Jumlah Terjual Bersih'].sum().reset_index()
    monthly_total['MoM_growth_pct'] = monthly_total['Jumlah Terjual Bersih'].pct_change()*100
    monthly_total['Rolling_3M'] = monthly_total['Jumlah Terjual Bersih'].rolling(3).mean()

    # Create subplot: 2 baris, 1 kolom
    fig_combo = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1,
                              subplot_titles=("MoM Growth (%)", "Rolling 3-Month Average"))

    # Trace 1: MoM Growth (Bar)
    fig_combo.add_trace(
        go.Bar(x=monthly_total['year_month'], y=monthly_total['MoM_growth_pct'], 
               name='MoM Growth', marker_color='#f97316', opacity=0.8),
        row=1, col=1
    )
    # Garis 0%
    fig_combo.add_shape(type="line", x0=monthly_total['year_month'].iloc[0], x1=monthly_total['year_month'].iloc[-1], 
                        y0=0, y1=0, line=dict(color="white", width=1, dash="dash"), row=1, col=1)

    # Trace 2: Rolling Average (Area)
    fig_combo.add_trace(
        go.Scatter(x=monthly_total['year_month'], y=monthly_total['Rolling_3M'], 
                   name='Rolling 3M Avg', fill='tozeroy', line=dict(color='#3b82f6', width=3)),
        row=2, col=1
    )

    fig_combo = update_plotly_theme(fig_combo)
    fig_combo.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_combo, use_container_width=True)

    # ==========================================
    # CONTAINER 4: REVENUE ANALYSIS (GROUPED BAR)
    # ==========================================
    st.markdown('<div id="text-split"><h2 class="text-xl">2. Revenue Analysis</h2></div>', unsafe_allow_html=True)

    df['Gross_Revenue'] = df[['Total Pembayaran','Total Diskon','Ongkos Kirim Dibayar oleh Pembeli']].sum(axis=1)
    
    revenue_summary = df.groupby('Kategori Produk').agg(
        Gross=('Gross_Revenue','sum'),
        Net=('Total Pembayaran','sum')
    ).reset_index()
    revenue_summary['Cost'] = revenue_summary['Gross'] - revenue_summary['Net']
    
    # Melt dataframe biar gampang dibikin Grouped Bar di Plotly
    rev_melt = revenue_summary.melt(id_vars='Kategori Produk', var_name='Type', value_name='Value')

    fig_rev = px.bar(rev_melt, x='Kategori Produk', y='Value', color='Type', barmode='group',
                     color_discrete_map={'Gross': '#64748b', 'Net': '#3b82f6', 'Cost': '#ec4899'},
                     title="Gross vs Net vs Cost Revenue")
    
    fig_rev.update_traces(texttemplate='%{y:.2s}', textposition='outside', textfont=dict(color='white'))
    fig_rev = update_plotly_theme(fig_rev)
    fig_rev.update_layout(yaxis_title="Revenue (Rp)", xaxis_tickangle=45)
    st.plotly_chart(fig_rev, use_container_width=True)

    # ==========================================
    # CONTAINER 5: GEOGRAPHIC & OPERATIONAL (HORIZONTAL BAR)
    # ==========================================
    st.markdown('<div id="text-split"><h2 class="text-xl">3. Geographic & Operational Drivers</h2></div>', unsafe_allow_html=True)

    col_geo, col_pay = st.columns(2)

    with col_geo:
        st.write("#### Top 10 Provinces by Volume")
        province_sales = df.groupby('Provinsi')['Jumlah'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prov = px.bar(province_sales, x='Jumlah', y='Provinsi', orientation='h',
                          color='Jumlah', color_continuous_scale='Blues')
        fig_prov = update_plotly_theme(fig_prov)
        fig_prov.update_layout(showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prov, use_container_width=True)

    with col_pay:
        st.write("#### Payment Methods Contribution")
        payment_summary = df.groupby('Metode Pembayaran')['Total Pembayaran'].sum().sort_values(ascending=False).reset_index()
        fig_pay = px.bar(payment_summary, x='Total Pembayaran', y='Metode Pembayaran', orientation='h',
                         color='Metode Pembayaran', color_discrete_sequence=COLOR_PALETTE)
        fig_pay = update_plotly_theme(fig_pay)
        fig_pay.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), xaxis_title="Total Revenue")
        st.plotly_chart(fig_pay, use_container_width=True)

    # ==========================================
    # FINAL EXECUTIVE SUMMARY (CLEAN LOOK)
    # ==========================================
    st.divider()
    st.markdown('<div id="text-split"><h2 class="text-xl">Executive Summary</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.2);">
    <h4 style="color: #3b82f6; margin-top:0;">Key Findings</h4>
    <ol style="color: #f8fafc; font-size: 0.95rem;">
        <li><b>Konsentrasi Produk:</b> Mayoritas transaksi (70%) terkonsentrasi di kategori Kitchen & Home Organization.</li>
        <li><b>Pola Volatil:</b> Permintaan sangat fluktuatif, dipicu oleh event promosi (Event-Driven Demand), bukan pertumbuhan organik yang stabil.</li>
        <li><b>COD Dominan:</b> Metode pembayaran COD masih menjadi pendorong utama volume transaksi dan revenue.</li>
        <li><b>Hambatan Logistik:</b> Wilayah Indonesia Timur (Papua, Maluku) menghadapi kendala ongkir tinggi yang menghambat distribusi.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()