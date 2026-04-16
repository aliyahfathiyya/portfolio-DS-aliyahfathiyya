import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- LANGKAH 1: MUAT DATA ---
# Pastikan file insurance.csv ada di folder yang sama atau tulis path lengkapnya
try:
    df = pd.read_csv(/insurance.csv') 
except FileNotFoundError:
    # Jika kamu pakai dataset dari kagglehub di notebook, 
    # kamu harus download manual csv-nya dan taruh di folder project ini
    print("Error: File insurance.csv tidak ditemukan di folder ini!")
    exit()

# --- LANGKAH 2: PREPROCESSING ---
df_final = df.copy()
df_final.drop_duplicates(inplace=True)

# Mapping kategori ke angka (Label Encoding Sederhana)
df_final['sex'] = df_final['sex'].map({'male': 1, 'female': 0})
df_final['smoker'] = df_final['smoker'].map({'yes': 1, 'no': 0})
df_final['region'] = df_final['region'].map({
    'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3
})

# Tentukan Fitur (X) dan Target (y)
X_final = df_final[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y_final = np.log1p(df_final['charges']) # Pakai Log sesuai model terbaikmu

# --- LANGKAH 3: BUAT PIPELINE & TRAINING ---
# Menggunakan parameter terbaik hasil GridSearchCV-mu
pipeline_asuransi = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))
])

print("Sedang melatih model...")
pipeline_asuransi.fit(X_final, y_final)

# --- LANGKAH 4: SIMPAN MODEL ---
# Simpan di folder yang sama dengan app.py nanti
pickle.dump(pipeline_asuransi, open('insurance_pipeline.sav', 'wb'))
print("Berhasil! File 'insurance_pipeline.sav' telah dibuat.")
