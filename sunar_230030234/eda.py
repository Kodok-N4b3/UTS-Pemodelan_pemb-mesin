# =====================================================
# LOAD DATA + EKSPLORASI DATA + EDA
# =====================================================


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# A. LOAD DATA
# =====================================================

# baca file excel
df = pd.read_excel("dataset_raw_from_webscraping.xlsx")

print("="*60)
print("5 DATA TERATAS")
print("="*60)
print(df.head())

# =====================================================
# INFO DATASET
# =====================================================

print("\n" + "="*60)
print("INFORMASI DATASET")
print("="*60)
print(df.info())

# =====================================================
# DESKRIPSI STATISTIK
# =====================================================

print("\n" + "="*60)
print("STATISTIK DESKRIPTIF")
print("="*60)
print(df.describe())

# =====================================================
# CEK MISSING VALUE
# =====================================================

print("\n" + "="*60)
print("JUMLAH DATA KOSONG")
print("="*60)
print(df.isnull().sum())

# =====================================================
# JUMLAH BARIS DAN KOLOM
# =====================================================

print("\n" + "="*60)
print("UKURAN DATASET")
print("="*60)
print("Jumlah Baris :", df.shape[0])
print("Jumlah Kolom :", df.shape[1])

# =====================================================
# MENAMPILKAN NAMA SEMUA FEATURE
# =====================================================

print("\n" + "="*60)
print("DAFTAR FEATURE / KOLOM")
print("="*60)

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# =====================================================
# ================== EDA 1 ============================
# DISTRIBUSI RATING
# =====================================================

plt.figure(figsize=(8,5))
sns.histplot(df['review_rating'], bins=10, kde=True)
plt.title("Distribusi Rating Laundry")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()

# =====================================================
# ================== EDA 2 ============================
# TOP 10 LAUNDRY DENGAN REVIEW TERBANYAK
# =====================================================

top10 = df[['title', 'review_count']].sort_values(
    by='review_count', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x='review_count', y='title', data=top10)
plt.title("Top 10 Laundry Dengan Review Terbanyak")
plt.xlabel("Jumlah Review")
plt.ylabel("Nama Laundry")
plt.tight_layout()
plt.show()

# =====================================================
# ================== EDA 3 ============================
# HUBUNGAN RATING DAN JUMLAH REVIEW
# =====================================================

plt.figure(figsize=(8,5))
sns.scatterplot(x='review_rating', y='review_count', data=df)
plt.title("Hubungan Rating dan Jumlah Review")
plt.xlabel("Rating")
plt.ylabel("Jumlah Review")
plt.tight_layout()
plt.show()

# =====================================================
# ================== EDA 4 ============================
# STATUS USAHA
# =====================================================

plt.figure(figsize=(7,4))
sns.countplot(x='status', data=df)
plt.title("Distribusi Status Laundry")
plt.xlabel("Status")
plt.ylabel("Jumlah")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# =====================================================
# ================== EDA 5 ============================
# HEATMAP KORELASI DATA NUMERIK
# =====================================================

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Feature Numerik")
plt.tight_layout()
plt.show()

# =====================================================
# KESIMPULAN SINGKAT
# =====================================================

print("\n" + "="*60)
print("KESIMPULAN EDA")
print("="*60)

print("1. Dataset berisi data usaha laundry hasil web scraping.")
print("2. Terdapat rating dan jumlah review tiap laundry.")
print("3. Sebagian data memiliki missing value pada website/status.")
print("4. Beberapa laundry memiliki review sangat tinggi.")
print("5. Data siap masuk tahap preprocessing.")
