# ==============================
# PIPELINE DENGAN SCALING
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load dataset
data = pd.read_excel('dataset_raw_from_webscraping.xlsx')

# 2. Feature & Target
X = data[['category', 'review_rating', 'latitude', 'longitude', 'status']]
y = data['popular']

# 3. Split data (WAJIB sebelum scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 5. Training model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Prediksi
y_pred = model.predict(X_test)

# 7. Evaluasi
print("=== DENGAN SCALING ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))