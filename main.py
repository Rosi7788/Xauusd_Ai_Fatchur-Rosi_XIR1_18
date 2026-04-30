"""
=============================================================
 ANALISIS DATA EMAS (Gold Futures - GC=F)
 Adaptasi dari Modul: Optimasi Strategi Pemasaran → Analisis Pasar Emas
=============================================================

Data: Gold_Data.csv (Yahoo Finance, 2015–2026)
Kolom: Date (index), Close, High, Low, Open, Volume

Karena data ini bukan data e-commerce (tidak ada CustomerID, Ad_Budget,
Discount, dll), setiap tugas dari modul diadaptasi ke konteks pasar emas:
  - "Monthly Sales Trend"  → Tren Harga Emas Bulanan
  - "Correlation Heatmap"  → Korelasi antar OHLCV
  - "Underperformer"       → Periode Volume Rendah tapi Harga Tinggi
  - "RFM Analysis"         → Profil Tahunan (Return, Frequency Kenaikan, Max Harga)
  - "Category Efficiency"  → Efisiensi Pergerakan per Tahun (Range vs Volume)
  - "Hypothesis Test"      → Apakah Volume Tinggi → Return Harian Lebih Besar?
  - "Linear Regression"    → Prediksi Harga Close dari Open

=============================================================
"""

# ── 0. Import & Load Data ───────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e2b96b',
    'axes.labelcolor': '#f5d78e',
    'xtick.color': '#c9a84c',
    'ytick.color': '#c9a84c',
    'text.color': '#f5f5f5',
    'grid.color': '#2a2a4a',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'lines.linewidth': 1.5,
    'font.size': 10,
})
GOLD   = '#f5d78e'
DARK   = '#e2b96b'
RED    = '#e05c5c'
GREEN  = '#5ce08a'
BLUE   = '#5c9ee0'

# Load Data
df = pd.read_csv('data/Gold_Data.csv', header=[0, 1], index_col=0)
df.index = pd.to_datetime(df.index)
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df = df.sort_index()

# Feature Engineering
df['Year']        = df.index.year
df['Month']       = df.index.to_period('M').astype(str)
df['Daily_Return']= df['Close'].pct_change() * 100   # % return harian
df['Range']       = df['High'] - df['Low']           # Volatilitas harian

print("=" * 55)
print("  DATA EMAS BERHASIL DIMUAT")
print("=" * 55)
print(f"  Jumlah baris  : {len(df):,}")
print(f"  Periode       : {df.index.min().date()} → {df.index.max().date()}")
print(f"  Kolom         : {list(df.columns)}")
print(f"  Close min/max : ${df['Close'].min():.2f} / ${df['Close'].max():.2f}")
print()


# ── 1. TREN HARGA EMAS BULANAN (Line Chart) ─────────────────
print("── [1] Tren Harga Emas Bulanan ──────────────────────")

monthly_avg = df.groupby('Month')['Close'].mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(monthly_avg)), monthly_avg.values, color=GOLD, lw=1.8)
ax.fill_between(range(len(monthly_avg)), monthly_avg.values,
                monthly_avg.values.min(), alpha=0.15, color=GOLD)

# Tick setiap tahun
yearly_ticks = [i for i, m in enumerate(monthly_avg.index) if m.endswith('-01')]
ax.set_xticks(yearly_ticks)
ax.set_xticklabels([m[:4] for m in monthly_avg.index[yearly_ticks]], rotation=45)

ax.set_title('Tren Rata-Rata Harga Emas Bulanan (USD/troy oz)', color=GOLD, fontsize=13)
ax.set_xlabel('Tahun')
ax.set_ylabel('Harga Close (USD)')
ax.grid(True)
plt.tight_layout()
plt.savefig('1_tren_bulanan.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Disimpan: 1_tren_bulanan.png\n")


# ── 2. KORELASI ANTAR VARIABEL (Heatmap) ────────────────────
print("── [2] Korelasi Antar Variabel OHLCV ────────────────")

cols_corr = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily_Return', 'Range']
correlation = df[cols_corr].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
sns.heatmap(
    correlation, annot=True, fmt='.2f', cmap='YlOrBr',
    linewidths=0.5, linecolor='#1a1a2e',
    ax=ax, vmin=-1, vmax=1
)
ax.set_title('Peta Korelasi Variabel Emas', color=GOLD, fontsize=13)
plt.tight_layout()
plt.savefig('2_korelasi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Disimpan: 2_korelasi_heatmap.png\n")

print("  Top korelasi dengan Close:")
corr_close = correlation['Close'].drop('Close').sort_values(ascending=False)
for col, val in corr_close.items():
    print(f"    {col:<15}: {val:+.4f}")
print()


# ── 3. SCATTER: VOLUME RENDAH vs HARGA TINGGI ("Underperformer") ──
print("── [3] Identifikasi: Harga Tinggi – Volume Rendah ───")

# Filter: Close di atas rata-rata & Volume di bawah median
avg_close  = df['Close'].mean()
med_volume = df['Volume'].median()

df_scatter   = df[df['Volume'] > 0].copy()  # hilangkan volume 0
underperform = df_scatter[(df_scatter['Close'] > avg_close) &
                          (df_scatter['Volume'] < med_volume)]
normal       = df_scatter[~df_scatter.index.isin(underperform.index)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(normal['Volume'],      normal['Close'],
           alpha=0.3, s=8, color=BLUE, label='Normal')
ax.scatter(underperform['Volume'], underperform['Close'],
           alpha=0.6, s=12, color=RED,
           label=f'Harga Tinggi–Volume Rendah (n={len(underperform):,})')

ax.axvline(med_volume, color=DARK, ls='--', lw=1, label=f'Median Volume ({med_volume:.0f})')
ax.axhline(avg_close,  color=GREEN, ls='--', lw=1, label=f'Rata-rata Close (${avg_close:.0f})')

ax.set_title('Scatter: Harga Emas vs Volume Transaksi', color=GOLD, fontsize=13)
ax.set_xlabel('Volume')
ax.set_ylabel('Harga Close (USD)')
ax.legend(fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig('3_scatter_harga_volume.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Rata-rata Close   : ${avg_close:.2f}")
print(f"  Median Volume     : {med_volume:.0f}")
print(f"  Hari 'Underperform': {len(underperform):,} dari {len(df_scatter):,}")
print(f"  Interpretasi: Harga tinggi seringkali justru bertepatan dengan")
print(f"  volume rendah – ini wajar di komoditas (low liquidity = high price)\n")


# ── 4. RFM → PROFIL TAHUNAN EMAS ────────────────────────────
print("── [4] Profil Tahunan Emas (Adaptasi RFM) ───────────")

# Recency   → Selisih hari dari puncak harga tertinggi tahun itu
# Frequency → Jumlah hari harga naik dari hari sebelumnya
# Monetary  → Harga Close tertinggi tahun itu (proxy "nilai puncak")

snapshot_date = df.index.max() + pd.Timedelta(days=1)

rfm = df.groupby('Year').agg(
    Recency   =('Close', lambda x: (snapshot_date - x.idxmax()).days),
    Frequency =('Daily_Return', lambda x: (x > 0).sum()),   # hari naik
    Monetary  =('Close', 'max')
).reset_index()

# Skor 1–5
def safe_qcut(series, labels):
    """qcut dengan fallback jika ada ties parah"""
    try:
        return pd.qcut(series, len(labels), labels=labels, duplicates='drop')
    except Exception:
        return pd.cut(series, len(labels), labels=labels)

rfm['R_Score'] = safe_qcut(rfm['Recency'],  [5, 4, 3, 2, 1])   # recency kecil = skor besar
rfm['F_Score'] = safe_qcut(rfm['Frequency'], [1, 2, 3, 4, 5])
rfm['M_Score'] = safe_qcut(rfm['Monetary'],  [1, 2, 3, 4, 5])
rfm['RFM_Score']= (rfm['R_Score'].astype(int) +
                   rfm['F_Score'].astype(int) +
                   rfm['M_Score'].astype(int))

rfm['Segmen'] = rfm['RFM_Score'].apply(
    lambda s: '🥇 Tahun Terbaik' if s >= 12 else
              ('🥈 Tahun Kuat'   if s >= 9  else
              ('🥉 Tahun Sedang' if s >= 6  else '📉 Tahun Lemah'))
)

print(rfm[['Year','Recency','Frequency','Monetary','RFM_Score','Segmen']].to_string(index=False))
print()

# Visualisasi RFM
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [
    ('Recency',   'Hari sejak Puncak Harga', RED),
    ('Frequency', 'Jumlah Hari Harga Naik',  GREEN),
    ('Monetary',  'Harga Close Tertinggi (USD)', GOLD),
]
for ax, (col, label, color) in zip(axes, metrics):
    bars = ax.bar(rfm['Year'].astype(str), rfm[col], color=color, alpha=0.8)
    ax.set_title(label, color=GOLD, fontsize=11)
    ax.set_xlabel('Tahun')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y')
    for bar, val in zip(bars, rfm[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rfm[col].max()*0.01,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7, color='white')

fig.suptitle('Profil Tahunan Emas – Adaptasi Analisis RFM', color=GOLD, fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('4_rfm_tahunan.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Disimpan: 4_rfm_tahunan.png\n")


# ── 5. EFISIENSI PERGERAKAN PER TAHUN (Bar Chart Horizontal) ─
print("── [5] Efisiensi Pergerakan Tahunan ─────────────────")

# Efisiensi = Net Return Tahunan / Total Range (%) → mirip "revenue vs ad_budget"
yearly = df.groupby('Year').agg(
    Open_First =('Open', 'first'),
    Close_Last =('Close', 'last'),
    Total_Range=('Range', 'sum'),
    Avg_Volume =('Volume', 'mean')
).reset_index()

yearly['Net_Return_Pct'] = (yearly['Close_Last'] - yearly['Open_First']) / yearly['Open_First'] * 100
yearly['Efficiency']     = yearly['Net_Return_Pct'] / (yearly['Total_Range'] / yearly['Open_First'] * 100)
yearly = yearly.sort_values('Efficiency')

colors = [GREEN if e > 0 else RED for e in yearly['Efficiency']]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(yearly['Year'].astype(str), yearly['Efficiency'], color=colors, alpha=0.85)
ax.axvline(0, color=DARK, lw=1.5)
ax.set_title('Efisiensi Pergerakan Tahunan Emas\n(Net Return % ÷ Total Volatilitas %)',
             color=GOLD, fontsize=13)
ax.set_xlabel('Skor Efisiensi')
ax.grid(True, axis='x')
for bar, val in zip(bars, yearly['Efficiency']):
    ax.text(val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
            fontsize=9, color='white')

plt.tight_layout()
plt.savefig('5_efisiensi_tahunan.png', dpi=150, bbox_inches='tight')
plt.close()

print(yearly[['Year','Net_Return_Pct','Efficiency']].to_string(index=False))
print()
print("  → Disimpan: 5_efisiensi_tahunan.png\n")


# ── 6. UJI HIPOTESIS: Volume Tinggi → Return Lebih Besar? ───
print("── [6] Uji Hipotesis: Volume vs Return ──────────────")

df_h = df.dropna(subset=['Daily_Return']).copy()
df_h = df_h[df_h['Volume'] > 0]

median_vol = df_h['Volume'].median()
high_vol   = df_h[df_h['Volume'] >= median_vol]['Daily_Return'].abs()
low_vol    = df_h[df_h['Volume'] <  median_vol]['Daily_Return'].abs()

t_stat, p_val = stats.ttest_ind(high_vol, low_vol, equal_var=False)

print(f"  Median Volume      : {median_vol:.0f}")
print(f"  Rata-rata |Return| Volume Tinggi : {high_vol.mean():.4f}%")
print(f"  Rata-rata |Return| Volume Rendah : {low_vol.mean():.4f}%")
print(f"  T-statistic : {t_stat:.4f}")
print(f"  P-value     : {p_val:.6f}")
print(f"  Kesimpulan  : {'✅ Signifikan (p < 0.05) – Volume tinggi menghasilkan pergerakan lebih besar' if p_val < 0.05 else '❌ Tidak signifikan'}")
print()

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
axes[0].boxplot([high_vol, low_vol], labels=['Volume Tinggi', 'Volume Rendah'],
                patch_artist=True,
                boxprops=dict(facecolor=GOLD, color=DARK),
                medianprops=dict(color=RED, linewidth=2),
                whiskerprops=dict(color=DARK),
                capprops=dict(color=DARK))
axes[0].set_title('Distribusi |Return Harian| per Kelompok Volume', color=GOLD)
axes[0].set_ylabel('|Return Harian| (%)')

# Histogram overlay
axes[1].hist(high_vol, bins=60, alpha=0.6, color=GREEN, label='Volume Tinggi', density=True)
axes[1].hist(low_vol,  bins=60, alpha=0.6, color=RED,   label='Volume Rendah', density=True)
axes[1].set_title(f'Distribusi Return (p-value={p_val:.4f})', color=GOLD)
axes[1].set_xlabel('|Return Harian| (%)')
axes[1].set_ylabel('Densitas')
axes[1].legend()

for ax in axes:
    ax.grid(True)

fig.suptitle('Uji Hipotesis: Volume Tinggi vs Return', color=GOLD, fontsize=13)
plt.tight_layout()
plt.savefig('6_uji_hipotesis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Disimpan: 6_uji_hipotesis.png\n")


# ── 7. REGRESI LINEAR: Open → Close ─────────────────────────
print("── [7] Regresi Linear: Open → Prediksi Close ────────")

X = df[['Open']].dropna()
y = df.loc[X.index, 'Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r2    = model.score(X_test, y_test)
coef  = model.coef_[0]
inter = model.intercept_

print(f"  Koefisien (Open)  : {coef:.6f}")
print(f"  Intercept         : {inter:.4f}")
print(f"  R² Score (test)   : {r2:.6f}")
print(f"  Interpretasi      : Setiap kenaikan $1 harga Open → Close naik ${coef:.4f}")
print()

# Contoh prediksi
for open_price in [1800, 2500, 3000]:
    pred = model.predict([[open_price]])[0]
    print(f"  Open=${open_price:,} → Prediksi Close=${pred:,.2f}")
print()

# Plot regresi
y_pred_all = model.predict(X)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X, y, alpha=0.15, s=5, color=BLUE, label='Data aktual')
ax.plot(X.sort_values('Open'), model.predict(X.sort_values('Open')),
        color=GOLD, lw=2, label=f'Regresi (R²={r2:.4f})')
ax.set_title('Regresi Linear: Harga Open → Prediksi Close', color=GOLD, fontsize=13)
ax.set_xlabel('Harga Open (USD)')
ax.set_ylabel('Harga Close (USD)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('7_regresi_linear.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Disimpan: 7_regresi_linear.png\n")


print("=" * 55)
print("  SEMUA ANALISIS SELESAI!")
print("  File output: 1_tren_bulanan.png")
print("               2_korelasi_heatmap.png")
print("               3_scatter_harga_volume.png")
print("               4_rfm_tahunan.png")
print("               5_efisiensi_tahunan.png")
print("               6_uji_hipotesis.png")
print("               7_regresi_linear.png")
print("=" * 55)