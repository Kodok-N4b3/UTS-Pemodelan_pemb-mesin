# =====================================================
# FEATURE EXTRACTION
# =====================================================

import pandas as pd

# load data hasil preprocessing
df = pd.read_excel("dataset_raw_from_webscraping.xlsx")

# =====================================================
# 1. MEMBUAT TARGET (LABEL)
# =====================================================

# gunakan median sebagai batas popularitas
median_review = df['review_count'].median()

df['popular'] = (df['review_count'] > median_review).astype(int)

print("=================================================")
print("DATA DENGAN TARGET BARU (POPULAR)")
print("=================================================")
print(df[['review_count','popular']].head())

# =====================================================
# 2. MEMILIH FEATURE YANG DIGUNAKAN
# =====================================================

# fitur yang digunakan untuk model
X = df[['category','review_rating','latitude','longitude','status']]

# target
y = df['popular']

print("\n=================================================")
print("FITUR (X)")
print("=================================================")
print(X.head())

print("\n=================================================")
print("TARGET (y)")
print("=================================================")
print(y.head())

# =====================================================
# 3. CEK DISTRIBUSI TARGET (UNTUK IMBALANCE)
# =====================================================

print("\n=================================================")
print("DISTRIBUSI TARGET")
print("=================================================")
print(y.value_counts())

# =====================================================
# 4. SIMPAN DATA HASIL FEATURE EXTRACTION (OPSIONAL)
# =====================================================

df.to_csv("data_feature_extraction.csv", index=False)

print("\n=================================================")
print("FEATURE EXTRACTION SELESAI")
print("=================================================")