# =====================================================
# SPLIT TRAINING DAN TEST DATA (LENGKAP)
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel("dataset_raw_from_webscraping.xlsx")

# =====================================================
# PREPROCESSING SINGKAT (WAJIB AGAR TIDAK ERROR)
# =====================================================

# pilih kolom yang dipakai
df = df[['category','review_rating','review_count','latitude','longitude','status']]

# handle missing value
df['status'] = df['status'].fillna("unknown")

# encoding kategori
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))
df['status'] = le.fit_transform(df['status'].astype(str))

# =====================================================
# FEATURE EXTRACTION (MEMBUAT TARGET)
# =====================================================

median_review = df['review_count'].median()

df['popular'] = (df['review_count'] > median_review).astype(int)

# fitur dan target
X = df[['category','review_rating','latitude','longitude','status']]
y = df['popular']

# =====================================================
# SPLIT DATA (80% TRAINING, 20% TESTING)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # menjaga distribusi tetap seimbang
)

# =====================================================
# HASIL SPLIT
# =====================================================

print("=================================================")
print("JUMLAH DATA TRAINING DAN TESTING")
print("=================================================")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# =====================================================
# DISTRIBUSI TARGET
# =====================================================

print("\n=================================================")
print("DISTRIBUSI TARGET TRAINING")
print("=================================================")
print(y_train.value_counts())

print("\n=================================================")
print("DISTRIBUSI TARGET TESTING")
print("=================================================")
print(y_test.value_counts())

