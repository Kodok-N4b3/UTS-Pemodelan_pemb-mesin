# =====================================================
# FULL PIPELINE: PREPROCESSING + SPLIT + MODELING
# =====================================================

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel("dataset_raw_from_webscraping.xlsx")

# =====================================================
# PREPROCESSING
# =====================================================

df = df[['category','review_rating','review_count','latitude','longitude','status']]

df['status'] = df['status'].fillna("unknown")

le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))
df['status'] = le.fit_transform(df['status'].astype(str))

# =====================================================
# FEATURE EXTRACTION
# =====================================================

median_review = df['review_count'].median()
df['popular'] = (df['review_count'] > median_review).astype(int)

X = df[['category','review_rating','latitude','longitude','status']]
y = df['popular']

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# SMOTE (IMBALANCE HANDLING)
# =====================================================

print("SEBELUM SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nSESUDAH SMOTE:")
print(pd.Series(y_train).value_counts())

# =====================================================
# GRID SEARCH
# =====================================================

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# =====================================================
# HASIL
# =====================================================

print("\nBEST PARAMETER:")
print(grid.best_params_)

print("\nBEST SCORE:")
print(grid.best_score_)

best_model = grid.best_estimator_


# =====================================================
# EVALUASI HASIL PEMODELAN
# =====================================================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PREDIKSI DATA TEST
# =====================================================

y_pred = best_model.predict(X_test)

# =====================================================
# AKURASI
# =====================================================

print("=================================================")
print("HASIL EVALUASI MODEL")
print("=================================================")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n=================================================")
print("CLASSIFICATION REPORT")
print("=================================================")
print(classification_report(y_test, y_pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# VISUALISASI CONFUSION MATRIX
# =====================================================

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importances = best_model.feature_importances_

# nama fitur (pastikan urutannya sama dengan X)
feature_names = ['category','review_rating','latitude','longitude','status']

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

import joblib

joblib.dump(best_model, "model_laundry.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model dan scaler berhasil disimpan!")

# =====================================================
# PREDIKSI 1 DATA BARU (FILE TERPISAH)
# =====================================================

import pandas as pd
import joblib

# LOAD MODEL & SCALER
model = joblib.load("model_laundry.pkl")
scaler = joblib.load("scaler.pkl")

# DATA BARU
data_baru = pd.DataFrame([
    [2, 4.8, -6.90, 107.60, 1]
], columns=['category','review_rating','latitude','longitude','status'])

print("Data Baru:")
print(data_baru)

# TRANSFORM
data_baru_scaled = scaler.transform(data_baru)

# PREDIKSI
hasil = model.predict(data_baru_scaled)

print("\n===== HASIL PREDIKSI =====")

if hasil[0] == 1:
    print("Laundry diprediksi POPULER")
else:
    print("Laundry diprediksi TIDAK POPULER")