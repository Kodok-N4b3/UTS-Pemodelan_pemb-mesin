# =====================================================
# PREPROCESSING DATA
# =====================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load data
df = pd.read_excel("dataset_raw_from_webscraping.xlsx")

print("=================================================")
print("DATA SEBELUM PREPROCESSING")
print("=================================================")
print(df.head())

# =====================================================
# PREPROCESSING 1: HAPUS KOLOM TIDAK DIGUNAKAN
# =====================================================

# kolom yang terlalu banyak missing / tidak relevan
kolom_dihapus = [
    'link','address','open_hours','popular_times','website',
    'plus_code','reviews_per_rating','cid','descriptions',
    'reviews_link','thumbnail','timezone','price_range',
    'data_id','images','reservations','order_online',
    'menu','owner','complete_address','about','user_reviews'
]

df = df.drop(columns=kolom_dihapus)

print("\n=================================================")
print("SETELAH HAPUS KOLOM")
print("=================================================")
print(df.head())

# =====================================================
# PREPROCESSING 2: MENANGANI MISSING VALUE
# =====================================================

# cek missing value
print("\nMissing Value Sebelum:")
print(df.isnull().sum())

# isi missing value
df['phone'] = df['phone'].fillna("tidak_ada")
df['status'] = df['status'].fillna("unknown")

print("\nMissing Value Sesudah:")
print(df.isnull().sum())

# =====================================================
# PREPROCESSING 3: ENCODING DATA KATEGORI
# =====================================================

le = LabelEncoder()

df['category'] = le.fit_transform(df['category'].astype(str))
df['status'] = le.fit_transform(df['status'].astype(str))

print("\n=================================================")
print("SETELAH ENCODING")
print("=================================================")
print(df.head())

# =====================================================
# PREPROCESSING 4 : CEK DUPLIKAT
# =====================================================

duplikat = df.duplicated().sum()
print("\nJumlah Data Duplikat:", duplikat)

df = df.drop_duplicates()

# =====================================================
# PREPROCESSING 5 : CEK OUTLIER SEDERHANA
# =====================================================

print("\nStatistik Review Count:")
print(df['review_count'].describe())

# =====================================================
# HASIL AKHIR
# =====================================================

print("\n=================================================")
print("DATA SETELAH PREPROCESSING SIAP DIGUNAKAN")
print("=================================================")
print(df.head())
print("\nShape:", df.shape)