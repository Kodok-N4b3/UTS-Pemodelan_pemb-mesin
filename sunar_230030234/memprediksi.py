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