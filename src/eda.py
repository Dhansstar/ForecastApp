import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import io


@st.cache_data
def load_data():

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data_from_DE.csv")

    df = pd.read_csv(data_path)

    df['Weekend'] = df['Weekend'].replace({0:'No',1:'Yes'})
    df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])

    cols = [
        'Total Diskon',
        'Ongkos Kirim Dibayar oleh Pembeli',
        'Estimasi Potongan Biaya Pengiriman',
        'Perkiraan Ongkos Kirim'
    ]

    df[cols] = df[cols].astype('int64')

    df['Provinsi'] = df['Provinsi'].str.replace(r'\(.*\)', '', regex=True).str.strip()

    return df


def run_eda():

    st.title("📊 Market Demand and Sales Analysis")

    st.markdown("""
Analisis ini bertujuan untuk memahami pola permintaan penjualan pada e-commerce sales dataset melalui eksplorasi tren penjualan, distribusi produk, perilaku pembelian konsumen, serta faktor operasional yang mempengaruhi transaksi.

Insight yang dihasilkan dari analisis ini digunakan sebagai dasar untuk memahami dinamika permintaan sebelum dilakukan proses demand forecasting.

""")

    # ==============================
    # LOAD DATA
    # ==============================

    with st.spinner("Loading dataset..."):
        df = load_data()

    # ==============================
    # EDA PROCESS (TIDAK DITAMPILKAN)
    # ==============================

    # check unique weekend values
    weekend_unique = df['Weekend'].unique()

    # dataframe info
    buffer = io.StringIO()
    df.info(buf=buffer)
    dataframe_info = buffer.getvalue()

    # duplicate check
    duplicate_count = df.duplicated().sum()

    # missing value check
    missing_values = df.isnull().sum()

    # semua variabel ini tetap ada untuk analisis
    _ = weekend_unique
    _ = dataframe_info
    _ = duplicate_count
    _ = missing_values

    # ==============================
    # PRODUCT DISTRIBUTION
    # ==============================

    st.header("Inspecting product distribution")

    product_count = df['Kategori Produk'].value_counts()
    st.write(product_count)

    colors = cm.Set2(range(len(product_count)))

    fig, ax = plt.subplots()
    ax.pie(product_count,
    labels=product_count.index,
    autopct='%1.1f%%',
    colors=colors)

    ax.set_title("Proporsi Kategori Produk")
    st.pyplot(fig)

    st.markdown("""

- **Kitchen & Dining (36.5%)** dan **Home Organization & Living (33.1%)** mendominasi penjualan dan menyumbang hampir **70% dari total transaksi**.
- Hal ini menunjukkan bahwa permintaan terutama berasal dari **produk kebutuhan rumah tangga dan peralatan dapur**.
- **Tools & Accessories (~16%)** menjadi kategori pendukung dengan kontribusi menengah.
- **Food Storage & Packaging (~8%)**, **Other (4%)**, dan **Bathroom & Cleaning (2.4%)** memiliki kontribusi yang relatif kecil terhadap total penjualan.
- Secara keseluruhan, demand sangat terkonsentrasi pada **beberapa kategori utama**, terutama produk rumah tangga.
""")

    # ==============================
    # CATEGORICAL COLUMN
    # ==============================

    cat_cols = list(df.select_dtypes(include='object').columns)

    listItem = []

    for col in cat_cols:
        listItem.append([col, df[col].nunique(), df[col].unique()])

    cat_df = pd.DataFrame(
        columns=['Column Name','Unique Count','Unique Values'],
        data=listItem
    )

    cat_cols_filtered = df.select_dtypes('object') \
        .drop(columns=['Kategori Produk','Provinsi','Alasan Pembatalan']) \
        .columns


    # ==============================
    # NUMERIC COLUMN
    # ==============================

    numeric_summary = df.drop(columns='Waktu Pesanan Dibuat').describe().round(2)

    num_cols = df.select_dtypes(np.number).columns
    
    st.header("Exploratory Data Analysis")
    # ==============================
    # MONTHLY SALES DISTRIBUTION
    # ==============================

    st.header("1. Monthly Sales Distribution")


    df['year_month'] = df['Waktu Pesanan Dibuat'].dt.to_period('M')

    monthly_category = (
    df.groupby(['year_month','Kategori Produk'])['Jumlah Terjual Bersih']
    .sum()
    .reset_index()
    )

    monthly_category['year_month'] = monthly_category['year_month'].astype(str)

    pivot_df = monthly_category.pivot(
    index='year_month',
    columns='Kategori Produk',
    values='Jumlah Terjual Bersih'
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12,8))

    pivot_df.plot(kind='area',stacked=True,alpha=0.85,ax=ax)

    plt.title('Monthly Sales Distribution of Each Product')
    plt.xlabel('Month')
    plt.ylabel('Total Sold')
    plt.xticks(rotation=45)
    plt.legend(title='Product Category',bbox_to_anchor=(1.05,1))
    st.pyplot(fig)

    monthly_std = pivot_df.std().sort_values(ascending=False)
    st.write(monthly_std)
    st.markdown("""

- Penjualan bulanan secara konsisten didominasi oleh **Kitchen & Dining** dan **Home Organization & Living**.
- **Tools & Accessories** menunjukkan kontribusi yang stabil meskipun volumenya lebih kecil.
- Tiga kategori lainnya memiliki kontribusi yang lebih kecil tetapi tetap mengikuti pola fluktuasi penjualan secara keseluruhan.
- **Kitchen & Dining memiliki variabilitas permintaan tertinggi**, menunjukkan perubahan demand yang cukup besar antar bulan.
- Hal ini mengindikasikan bahwa perubahan total penjualan bulanan sebagian besar dipengaruhi oleh kategori **Kitchen & Dining**.
""")

    # ==============================
    # MONTHLY SALES TREND
    # ==============================

    st.header("2. Monthly Sales Trend and Growth")

    monthly_total = (
    df.groupby('year_month')['Jumlah Terjual Bersih']
    .sum()
    .reset_index()
    )

    monthly_total['year_month'] = monthly_total['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(12,5))

    sns.lineplot(data=monthly_total,
    x='year_month',
    y='Jumlah Terjual Bersih',
    marker='o',
    linewidth=2)

    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""

- Tren penjualan bulanan menunjukkan **fluktuasi yang cukup tajam** dengan beberapa lonjakan penjualan di periode tertentu.
- Pola ini mengindikasikan bahwa permintaan **dipengaruhi oleh faktor eksternal** seperti promo marketplace atau event musiman.
- Analisis **Month-over-Month (MoM) Growth** menunjukkan bahwa pertumbuhan penjualan bersifat **sangat volatil**.
- Rolling **3-month average smoothing** menunjukkan bahwa permintaan cenderung **naik turun tanpa tren pertumbuhan jangka panjang yang kuat**.
- Lonjakan penjualan kemungkinan besar dipicu oleh **event-driven demand**, bukan pertumbuhan organik yang konsisten.
""")
    monthly_total['MoM_growth_pct'] = monthly_total['Jumlah Terjual Bersih'].pct_change()*100

    fig, ax = plt.subplots(figsize=(12,4))

    sns.barplot(data=monthly_total,
    x='year_month',
    y='MoM_growth_pct')

    plt.axhline(0,color='black')

    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""

                
- Analisis **Month-over-Month (MoM) Growth** menunjukkan bahwa pertumbuhan penjualan bersifat **sangat volatil**.
- Beberapa bulan mengalami **lonjakan penjualan yang tinggi**, namun sering diikuti oleh **penurunan tajam pada bulan berikutnya**.
- Pola ini menunjukkan bahwa demand **tidak stabil** dan cenderung dipengaruhi oleh **momentum tertentu**.
- Lonjakan penjualan kemungkinan besar dipicu oleh **event-driven demand**, seperti promo marketplace, diskon musiman, atau event belanja besar.
""")

    monthly_total['Rolling_3M'] = monthly_total['Jumlah Terjual Bersih'].rolling(3).mean()

    fig, ax = plt.subplots(figsize=(12,4))

    sns.barplot(data=monthly_total,
    x='year_month',
    y='Rolling_3M')

    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""

- Untuk melihat pola permintaan yang lebih stabil, dilakukan **smoothing menggunakan rolling 3-month average**.
- Teknik ini membantu **mengurangi fluktuasi jangka pendek** sehingga pola permintaan dasar lebih mudah diamati.
- Setelah smoothing, terlihat bahwa permintaan cenderung **bergerak dalam pola naik-turun yang relatif konsisten**.
- Secara keseluruhan, **tidak terlihat tren pertumbuhan jangka panjang yang kuat** pada penjualan.
- Lonjakan penjualan pada bulan tertentu kemungkinan besar disebabkan oleh **promosi atau event musiman**, bukan oleh peningkatan demand yang berkelanjutan.
""")

    # ==============================
    # REVENUE
    # ==============================

    st.header("3. Gross and Net Revenue")
    
    df['Total Harga'] = df[
    ['Total Pembayaran','Total Diskon','Ongkos Kirim Dibayar oleh Pembeli']
    ].sum(axis=1)

    revenue_summary = (
    df.groupby('Kategori Produk')
    .agg(
    Gross_Revenue=('Total Harga','sum'),
    Net_Revenue=('Total Pembayaran','sum')
    )
    .sort_values('Gross_Revenue',ascending=False)
    .reset_index()
    )

    revenue_summary['Cost_Revenue'] = revenue_summary['Gross_Revenue'] - revenue_summary['Net_Revenue']

    st.write(revenue_summary)

    fig, ax = plt.subplots(figsize=(12,6))

    x = np.arange(len(revenue_summary['Kategori Produk']))
    width = 0.25

    bars1 = ax.bar(x - width, revenue_summary['Gross_Revenue'], width, label='Gross Revenue')
    bars2 = ax.bar(x, revenue_summary['Net_Revenue'], width, label='Net Revenue')
    bars3 = ax.bar(x + width, revenue_summary['Cost_Revenue'], width, label='Cost Revenue')

    ax.set_xticks(x)
    ax.set_xticklabels(revenue_summary['Kategori Produk'], rotation=45)

    ax.set_title("Gross vs Net vs Cost Revenue by Product Category")
    ax.set_ylabel("Revenue")
    ax.legend()

    # add value label
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height/1e6:.0f}M',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()

    st.pyplot(fig)
    st.markdown("""

- **Kitchen & Dining** menjadi kontributor pendapatan terbesar dengan gross revenue sekitar **500 juta** dan net revenue sekitar **423 juta**.
- Selisih revenue menunjukkan adanya **cost yang cukup besar**, terutama dari diskon dan biaya logistik.
- **Home Organization & Living** memiliki volume transaksi tinggi namun net revenue lebih rendah karena cost yang lebih besar.
- **Tools & Accessories** menghasilkan net revenue yang relatif lebih efisien dengan cost yang lebih kecil.
- Pola ini menunjukkan bahwa kategori dengan volume tinggi biasanya membutuhkan **promo dan subsidi ongkir untuk mendorong penjualan**.
""")

    # ==============================
    # WEEKDAY VS WEEKEND
    # ==============================

    st.header("4. Weekday vs Weekend Sales")

    weekend_sales = (
    df.groupby(['Weekend','Kategori Produk'])['Jumlah Terjual Bersih']
    .sum()
    .reset_index()
    )

    pivot_weekend = weekend_sales.pivot(
    index='Kategori Produk',
    columns='Weekend',
    values='Jumlah Terjual Bersih'
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10,6))

    pivot_weekend.plot(kind='bar',ax=ax)

    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown("""

- Penjualan pada **weekday secara konsisten lebih tinggi dibandingkan weekend**.
- Hal ini menunjukkan bahwa sebagian besar transaksi terjadi saat **hari kerja**.
- Perilaku ini kemungkinan dipengaruhi oleh aktivitas konsumen yang membuka marketplace saat menjalani rutinitas harian.
""")

    # ==============================
    # PROVINCE SALES
    # ==============================

    st.header("5. Sales by Provinces")

    province_sales = (
    df.groupby('Provinsi')['Jumlah']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,6))

    sns.barplot(x=province_sales.values,y=province_sales.index)

    st.pyplot(fig)
    st.markdown("""

- Permintaan produk paling tinggi terkonsentrasi di **Pulau Jawa**, terutama **Jawa Barat, DKI Jakarta, dan Banten**.
- Wilayah ini memiliki kepadatan penduduk tinggi serta akses logistik yang lebih baik.
- Provinsi seperti **Jawa Timur, Jawa Tengah, dan Sumatera Selatan** juga memberikan kontribusi penjualan yang cukup signifikan.
- Provinsi di luar Jawa menunjukkan volume penjualan lebih kecil tetapi tetap memiliki potensi pasar.
""")

    shipping_by_province = (
    df.groupby('Provinsi')['Ongkos Kirim Dibayar oleh Pembeli']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    )

    top_shipping = shipping_by_province.head(15)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(data=top_shipping,
    x='Ongkos Kirim Dibayar oleh Pembeli',
    y='Provinsi')

    st.pyplot(fig)
    st.markdown("""

- Rata-rata ongkir tertinggi berasal dari wilayah **Indonesia Timur** seperti Papua dan Maluku.
- Biaya logistik yang tinggi menjadi salah satu hambatan utama distribusi penjualan di wilayah tersebut.
- Hal ini menunjukkan bahwa **akses logistik dan jarak pengiriman mempengaruhi permintaan produk**.
- Strategi seperti **subsidi ongkir atau promo khusus** dapat membantu meningkatkan penjualan di wilayah dengan ongkir tinggi.
""")

    # ==============================
    # PAYMENT METHODS
    # ==============================

    st.header("6. Payment Methods to Drive Sales and Revenue")

    payment_summary = (
    df.groupby('Metode Pembayaran')
    .agg(
    Total_Quantity=('Jumlah','sum'),
    Total_Revenue=('Total Pembayaran','sum')
    )
    .sort_values('Total_Quantity',ascending=False)
    .head(5)
    .reset_index()
    )

    fig, ax = plt.subplots(1,2,figsize=(14,5))

    sns.barplot(data=payment_summary,
    x='Total_Quantity',
    y='Metode Pembayaran',
    ax=ax[0])

    sns.barplot(data=payment_summary,
    x='Total_Revenue',
    y='Metode Pembayaran',
    ax=ax[1])

    st.pyplot(fig)
    st.markdown("""

- **COD (Cash on Delivery)** merupakan metode pembayaran yang paling sering digunakan.
- COD juga menjadi **kontributor revenue terbesar** karena tingginya volume transaksi.
- **Online Payment dan ShopeePay** memiliki kontribusi transaksi yang lebih kecil namun tetap stabil.
- **SPayLater** memiliki jumlah transaksi lebih kecil tetapi menghasilkan **nilai pembelian yang relatif lebih tinggi**.
- Hal ini menunjukkan bahwa metode cicilan dapat meningkatkan **average order value**.
""")

    
    
    st.header("Executive Summary")

    st.write("""
    Analisis ini bertujuan untuk memahami pola permintaan, faktor pendorong penjualan, serta beberapa aspek operasional yang mempengaruhi performa penjualan dalam dataset. Dengan mengeksplorasi distribusi produk, tren penjualan bulanan, perilaku pembelian konsumen, hingga faktor logistik dan operasional, analisis ini memberikan gambaran menyeluruh mengenai dinamika penjualan yang terjadi.
    """)

    st.subheader("Key Findings")

    st.write("""
    1. Penjualan terkonsentrasi pada beberapa kategori utama, terutama Kitchen & Dining dan Home Organization & Living yang menyumbang mayoritas transaksi.
    2. Permintaan bersifat fluktuatif dan cenderung dipengaruhi oleh momentum tertentu, seperti event promosi atau periode belanja musiman.
    3. Metode pembayaran COD masih mendominasi transaksi, meskipun metode pembayaran digital mulai menunjukkan kontribusi pendapatan yang signifikan.
    4. Biaya logistik menjadi faktor penting dalam distribusi penjualan, dengan wilayah di luar Pulau Jawa memiliki ongkir yang jauh lebih tinggi.
    5. Aktivitas pembelian lebih banyak terjadi pada hari kerja, menunjukkan pola belanja yang berkaitan dengan rutinitas harian konsumen.
    6. Tingkat return produk relatif rendah, yang menunjukkan bahwa sebagian besar produk sesuai dengan ekspektasi pembeli.
    7. Pembatalan pesanan lebih banyak dipicu oleh keputusan pembeli, bukan oleh masalah produk atau operasional.
    """)

    st.header("1. Demand Overview")

    st.write("""
    Secara umum, penjualan dalam dataset didominasi oleh beberapa kategori produk utama. Kitchen & Dining dan Home Organization & Living menyumbang hampir 70% dari total transaksi, menjadikannya kontributor utama terhadap volume penjualan. Kategori Tools & Accessories juga memberikan kontribusi yang cukup signifikan, sementara kategori lainnya memiliki porsi penjualan yang lebih kecil.

    Dari sisi geografis, sebagian besar penjualan terkonsentrasi di provinsi-provinsi dengan populasi tinggi dan akses logistik yang baik, khususnya di wilayah Pulau Jawa seperti Jawa Barat, DKI Jakarta, dan Banten. Hal ini menunjukkan bahwa permintaan produk sangat dipengaruhi oleh kepadatan populasi serta kemudahan distribusi logistik.
    """)

    st.header("2. Demand Pattern")

    st.write("""
    Analisis tren penjualan bulanan menunjukkan bahwa permintaan cenderung fluktuatif dari waktu ke waktu. Beberapa periode menunjukkan lonjakan penjualan yang cukup tinggi, namun sering diikuti dengan penurunan tajam pada bulan berikutnya.

    Hal ini terlihat jelas pada analisis Month-over-Month growth, yang menunjukkan adanya spike pertumbuhan yang ekstrem di beberapa bulan. Setelah dilakukan smoothing menggunakan rolling 3-month average, pola permintaan terlihat lebih stabil dan menunjukkan adanya indikasi seasonality dalam penjualan.

    Meskipun terdapat lonjakan penjualan pada bulan tertentu, secara keseluruhan tidak terlihat tren pertumbuhan jangka panjang yang konsisten. Hal ini mengindikasikan bahwa permintaan kemungkinan dipengaruhi oleh event promosi atau momentum belanja tertentu.
    """)

    st.header("3. Demand Drivers")

    st.write("""
    Beberapa faktor yang berpotensi mempengaruhi permintaan dapat diidentifikasi dari perilaku transaksi dalam dataset.

    Dari sisi metode pembayaran, COD (Bayar di Tempat) masih menjadi metode pembayaran yang paling dominan baik dari sisi jumlah transaksi maupun kontribusi revenue. Namun metode pembayaran digital seperti Online Payment dan ShopeePay juga memberikan kontribusi pendapatan yang cukup besar.

    Selain itu, metode cicilan seperti SPayLater menunjukkan indikasi menghasilkan nilai transaksi yang relatif lebih tinggi meskipun jumlah penggunaannya lebih sedikit.

    Faktor logistik juga berperan dalam distribusi penjualan antar wilayah. Analisis ongkos kirim menunjukkan bahwa wilayah di luar Pulau Jawa, seperti Papua dan Maluku, memiliki biaya pengiriman yang jauh lebih tinggi dibanding wilayah lainnya. Kondisi ini berpotensi menjadi hambatan bagi aktivitas e-commerce di daerah tersebut.

    Dari sisi waktu transaksi, penjualan cenderung lebih tinggi pada hari kerja dibanding akhir pekan. Hal ini menunjukkan bahwa aktivitas pembelian kemungkinan besar dilakukan selama rutinitas harian, misalnya ketika konsumen membuka marketplace di sela aktivitas kerja.
    """)

    st.header("4. Operational Factors")

    st.write("""
    Selain faktor permintaan, beberapa aspek operasional juga dianalisis untuk melihat potensi hambatan dalam proses transaksi.

    Tingkat pengembalian produk (return rate) secara keseluruhan relatif rendah di hampir semua kategori produk, yang menunjukkan bahwa sebagian besar produk diterima dengan baik oleh pembeli. Namun kategori Food Storage & Packaging memiliki return rate yang sedikit lebih tinggi dibanding kategori lainnya sehingga dapat menjadi area yang perlu dievaluasi lebih lanjut.

    Sementara itu, order cancellation rate menunjukkan pola yang relatif konsisten di seluruh kategori produk dengan kisaran sekitar 10–18%. Analisis alasan pembatalan menunjukkan bahwa sebagian besar pembatalan terjadi karena faktor dari sisi pembeli, seperti perubahan keputusan atau modifikasi pesanan, sementara faktor operasional seperti kegagalan pengiriman hanya menyumbang sebagian kecil dari total pembatalan.
    """)

    st.header("Business Recommendations")

    st.write("""
    Berdasarkan keseluruhan hasil analisis, terdapat beberapa strategi bisnis yang dapat dipertimbangkan untuk meningkatkan penjualan sekaligus memperbaiki efisiensi operasional.
    """)

    st.subheader("1. Fokus pada Kategori Produk dengan Permintaan Tinggi")

    st.write("""
    Kategori Kitchen & Dining serta Home Organization & Living merupakan kontributor utama terhadap volume penjualan dan revenue. Oleh karena itu, kedua kategori ini dapat dijadikan sebagai core product line dalam strategi bisnis.

    Beberapa langkah yang dapat dilakukan antara lain meningkatkan ketersediaan stok pada kategori tersebut, memperluas variasi produk yang masih relevan dengan kebutuhan rumah tangga, serta memanfaatkan kategori ini sebagai produk utama dalam kampanye promosi.

    Dengan memaksimalkan kategori yang sudah memiliki permintaan tinggi, perusahaan dapat meningkatkan penjualan tanpa harus membangun permintaan dari nol.
    """)

    st.subheader("2. Memanfaatkan Momentum Promosi untuk Mengoptimalkan Lonjakan Penjualan")

    st.write("""
    Analisis tren bulanan menunjukkan bahwa penjualan sering mengalami lonjakan pada periode tertentu dan kemudian turun kembali pada bulan berikutnya. Pola ini mengindikasikan bahwa penjualan kemungkinan dipengaruhi oleh event-driven demand.

    Strategi yang dapat diterapkan adalah menyelaraskan aktivitas pemasaran dengan momentum tersebut, misalnya dengan memberikan promo tambahan pada periode kampanye besar, meningkatkan eksposur produk saat event belanja berlangsung, serta memastikan ketersediaan stok sebelum periode promosi dimulai.
    """)

    st.subheader("3. Mendorong Adopsi Metode Pembayaran Digital")

    st.write("""
    Walaupun COD masih mendominasi transaksi, metode pembayaran digital seperti Online Payment dan ShopeePay sudah menunjukkan kontribusi revenue yang cukup besar.

    Perusahaan dapat mendorong penggunaan metode pembayaran digital melalui program cashback, diskon tambahan, atau promosi eksklusif untuk metode pembayaran tertentu.
    """)

    st.subheader("4. Mengoptimalkan Strategi Logistik untuk Wilayah di Luar Pulau Jawa")

    st.write("""
    Analisis ongkos kirim menunjukkan bahwa wilayah di luar Pulau Jawa memiliki biaya logistik yang jauh lebih tinggi dibanding wilayah lain.

    Salah satu strategi yang dapat dipertimbangkan adalah memberikan subsidi ongkir atau promosi khusus untuk wilayah dengan biaya pengiriman tinggi.
    """)

    st.subheader("5. Mengurangi Return Rate pada Kategori dengan Risiko Lebih Tinggi")

    st.write("""
    Kategori Food Storage & Packaging menunjukkan return rate yang lebih tinggi dibanding kategori lainnya.

    Perbaikan dapat dilakukan melalui peningkatan deskripsi produk, foto produk yang lebih jelas, serta informasi spesifikasi yang lebih detail.
    """)

    st.subheader("6. Mengurangi Pembatalan Pesanan Melalui Perbaikan Proses Checkout")

    st.write("""
    Sebagian besar pembatalan pesanan berasal dari keputusan pembeli, seperti perubahan pesanan atau perubahan alamat pengiriman.

    Perusahaan dapat menerapkan fitur seperti konfirmasi pesanan sebelum checkout selesai, batas waktu perubahan alamat, atau pengingat pembayaran otomatis.
    """)