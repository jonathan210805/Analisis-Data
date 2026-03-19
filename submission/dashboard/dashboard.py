import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

#Load Data 
def load_data():
    main_df = pd.read_csv("main_data.csv")

    season_map  = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
    weather_map = {1:'Clear', 2:'Cloudy', 3:'Light Rain/Snow', 4:'Heavy Rain/Snow'}
    weekday_map = {0:'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'}

    main_df['dteday']        = pd.to_datetime(main_df['dteday'])
    main_df['season_label']  = main_df['season'].map(season_map)
    main_df['weather_label'] = main_df['weathersit'].map(weather_map)
    main_df['weekday_label'] = main_df['weekday'].map(weekday_map)
    main_df['temp_actual']      = main_df['temp'] * 41
    main_df['hum_actual']       = main_df['hum'] * 100
    main_df['windspeed_actual'] = main_df['windspeed'] * 67

    # Pisahkan berdasarkan kolom 'hr'
    day_df  = main_df[main_df['hr'].isna()].copy()
    hour_df = main_df[main_df['hr'].notna()].copy()
    hour_df['hr'] = hour_df['hr'].astype(int)

    # Clustering jam
    def cluster_hour(hour):
        if hour in [0,1,2,3,4,5]:        return 'Low (Dini Hari)'
        elif hour in [6,7,8,9]:           return 'Peak (Rush Pagi)'
        elif hour in [10,11,12,13,14,15]: return 'Medium (Siang)'
        elif hour in [16,17,18,19]:       return 'Peak (Rush Sore)'
        else:                             return 'Low (Malam)'
    hour_df['hour_cluster'] = hour_df['hr'].apply(cluster_hour)

    # Binning suhu
    day_df['temp_bin'] = pd.cut(
        day_df['temp_actual'],
        bins=[0, 10, 20, 30, 41],
        labels=['Dingin (0-10°C)','Sejuk (10-20°C)',
                'Hangat (20-30°C)','Panas (30-41°C)']
    )
    return day_df, hour_df

day_df, hour_df = load_data()

# Sidebar
st.sidebar.header("Filter Data")

# Filter tanggal
date_range = st.sidebar.date_input(
    "Rentang Waktu",
    [day_df['dteday'].min(), day_df['dteday'].max()]
)

# Filter musim
seasons    = sorted(day_df['season_label'].unique())
sel_season = st.sidebar.multiselect("Musim", options=seasons, default=seasons)

# Terapkan filter
filtered_day  = day_df[
    (day_df['dteday'] >= pd.to_datetime(date_range[0])) &
    (day_df['dteday'] <= pd.to_datetime(date_range[1])) &
    (day_df['season_label'].isin(sel_season))
]
filtered_hour = hour_df[
    (hour_df['dteday'] >= pd.to_datetime(date_range[0])) &
    (hour_df['dteday'] <= pd.to_datetime(date_range[1])) &
    (hour_df['season_label'].isin(sel_season))
]

st.sidebar.markdown('---')
st.sidebar.info(f"**{len(filtered_day):,}** hari ditampilkan")

# Header
st.title("Bike Sharing Analytics Dashboard")
st.markdown("**Dataset**:Bike Sharing Dataset | **Periode:** 2011–2012")
st.markdown("---")

# Metriks inti 
col1, col2, col3 = st.columns(3)
col1.metric("Total Peminjaman",      f"{filtered_day['cnt'].sum():,}")
col2.metric("Rata-rata/Hari",        f"{int(filtered_day['cnt'].mean())}")
col3.metric("Peminjaman Tertinggi",  f"{filtered_day['cnt'].max():,}")
st.markdown("---")

# Filter Tab berdasarkan pertanyaan bisnis
tab1, tab2, tab3 = st.tabs([
    " Pengaruh Cuaca",
    "Pola Jam & Hari",
    "Analisis Lanjutan"
])

# TAB 1:PENGARUH CUACA
with tab1:
    st.subheader("Pengaruh Kondisi Cuaca terhadap Peminjaman")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=filtered_day, x='weather_label', y='cnt',
                    estimator='mean', ax=ax, palette='viridis')
        ax.set_title("Rata-rata Peminjaman per Kondisi Cuaca", fontweight='bold')
        ax.set_xlabel("Kondisi Cuaca")
        ax.set_ylabel("Rata-rata Peminjaman")
        ax.tick_params(axis='x', rotation=15)
        st.pyplot(fig); plt.close()

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=filtered_day, x='season_label', y='cnt',
                    estimator='mean', ax=ax2, palette='Set2')
        ax2.set_title("Rata-rata Peminjaman per Musim", fontweight='bold')
        ax2.set_xlabel("Musim")
        ax2.set_ylabel("Rata-rata Peminjaman")
        st.pyplot(fig2); plt.close()

    # Scatter plot suhu vs peminjaman
    st.markdown("#### Hubungan Suhu dengan Jumlah Peminjaman")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.scatter(filtered_day['temp_actual'], filtered_day['cnt'],
                alpha=0.4, color='#E64A19', s=20)
    z = np.polyfit(filtered_day['temp_actual'], filtered_day['cnt'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(filtered_day['temp_actual'].min(),
                         filtered_day['temp_actual'].max(), 100)
    ax3.plot(x_line, p(x_line), 'b-', lw=2, label='Trend')
    ax3.set_xlabel("Suhu (°C)"); ax3.set_ylabel("Jumlah Peminjaman")
    ax3.set_title("Suhu vs Jumlah Peminjaman", fontweight='bold')
    ax3.legend()
    st.pyplot(fig3); plt.close()

# TAB 2:POLA JAM & HARI
with tab2:
    st.subheader("Pola Peminjaman Berdasarkan Jam dan Hari")

    col_c, col_d = st.columns(2)

    with col_c:
        hour_avg = filtered_hour.groupby('hr')['cnt'].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.plot(hour_avg['hr'], hour_avg['cnt'],'o-', color='#1565C0', lw=2.5, ms=5)
        ax4.fill_between(hour_avg['hr'], hour_avg['cnt'], alpha=0.2, color='#1565C0')
        ax4.set_xticks(range(0, 24))
        ax4.set_xlabel("Jam"); ax4.set_ylabel("Rata-rata Peminjaman")
        ax4.set_title("Avg Peminjaman per Jam", fontweight='bold')
        ax4.axvspan(7,  9,  alpha=0.15, color='red')
        ax4.axvspan(17, 19, alpha=0.15, color='orange')
        st.pyplot(fig4); plt.close()

    with col_d:
        # Perbandingan hari kerja vs akhir pekan
        hourly_workday = filtered_hour.groupby(['hr','workingday'])['cnt'].mean().reset_index()
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        for wday, color, label in [(1,'#E64A19','Hari Kerja'),(0,'#4CAF50','Akhir Pekan')]:
            sub = hourly_workday[hourly_workday['workingday'] == wday]
            ax5.plot(sub['hr'], sub['cnt'], 'o-', color=color, lw=2, ms=4, label=label)
        ax5.set_xticks(range(0, 24))
        ax5.set_xlabel("Jam"); ax5.set_ylabel("Rata-rata Peminjaman")
        ax5.set_title("Hari Kerja vs Akhir Pekan", fontweight='bold')
        ax5.legend()
        st.pyplot(fig5); plt.close()

    # Heatmap jam vs hari
    st.markdown("#### Heatmap Peminjaman: Jam vs Hari")
    day_order    = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heatmap_data = filtered_hour.groupby(['weekday_label','hr'])['cnt'].mean().unstack()
    heatmap_data = heatmap_data.reindex(day_order)
    fig6, ax6 = plt.subplots(figsize=(14, 5))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax6,
                cbar_kws={'label':'Avg Peminjaman'})
    ax6.set_xlabel("Jam"); ax6.set_ylabel("Hari")
    ax6.set_title("Heatmap Peminjaman per Jam dan Hari", fontweight='bold')
    st.pyplot(fig6); plt.close()

# TAB 3: ANALISIS LANJUTAN
with tab3:
    st.subheader(" Analisis Lanjutan  Clustering & Binning")

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("#### Clustering Jam  Tingkat Kesibukan")
        cluster_avg = filtered_hour.groupby('hour_cluster')['cnt'].mean().reset_index()
        cluster_avg.columns = ['cluster','avg_rentals']
        cluster_order  = ['Peak (Rush Pagi)','Peak (Rush Sore)','Medium (Siang)','Low (Malam)','Low (Dini Hari)']
        cluster_colors = ['#F44336','#FF9800','#2196F3','#9E9E9E','#607D8B']
        cluster_plot   = cluster_avg.set_index('cluster').reindex(cluster_order).reset_index()

        fig7, ax7 = plt.subplots(figsize=(7, 4))
        bars7 = ax7.bar(cluster_plot['cluster'], cluster_plot['avg_rentals'],
                        color=cluster_colors, edgecolor='white', lw=1.5)
        ax7.set_xlabel("Cluster"); ax7.set_ylabel("Rata-rata Peminjaman")
        ax7.set_title("Clustering Jam Berdasarkan Kesibukan", fontweight='bold')
        ax7.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars7, cluster_plot['avg_rentals']):
            ax7.text(bar.get_x()+bar.get_width()/2, val+2,f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
        st.pyplot(fig7); plt.close()

    with col_f:
        st.markdown("#### Binning Suhu  Pengaruh Rentang Suhu")
        filtered_day2 = filtered_day.copy()
        filtered_day2['temp_bin'] = pd.cut(
            filtered_day2['temp_actual'],
            bins=[0, 10, 20, 30, 41],
            labels=['Dingin (0-10°C)','Sejuk (10-20°C)',
                    'Hangat (20-30°C)','Panas (30-41°C)']
        )
        temp_bin_avg = filtered_day2.groupby('temp_bin', observed=True)['cnt'].mean().reset_index()

        fig8, ax8 = plt.subplots(figsize=(7, 4))
        temp_colors = ['#64B5F6','#4CAF50','#FF9800','#F44336']
        bars8 = ax8.bar(temp_bin_avg['temp_bin'].astype(str), temp_bin_avg['cnt'],
                        color=temp_colors, edgecolor='white', lw=1.5)
        ax8.set_xlabel("Kelompok Suhu"); ax8.set_ylabel("Rata-rata Peminjaman")
        ax8.set_title("Binning Suhu vs Peminjaman", fontweight='bold')
        ax8.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars8, temp_bin_avg['cnt']):
            ax8.text(bar.get_x()+bar.get_width()/2, val+2,f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
        st.pyplot(fig8); plt.close()
