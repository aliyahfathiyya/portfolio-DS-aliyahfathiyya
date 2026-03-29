import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Model dan Scaler (Pastikan file ini ada di folder yang sama)
try:
    model = pickle.load(open('insurance_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan! Pastikan sudah diexport dari notebook.")

# 2. Judul Aplikasi
st.set_page_config(page_title="Insurance Predictor", page_icon="💰")
st.title("💰 Insurance Charge Predictor")
st.write("Masukkan data diri Anda untuk mendapatkan estimasi biaya asuransi.")

st.divider()

# 3. Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Fisik")
    age = st.number_input("Umur", min_value=1, max_value=100, value=25)
    
    # Input Berat dan Tinggi untuk Hitung BMI
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height_cm = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=165.0)
    
    # Kalkulasi BMI Otomatis
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    
    # Menampilkan hasil BMI ke user
    st.info(f"**BMI Anda: {bmi:.2f}**")
    
    children = st.selectbox("Jumlah Anak", [0, 1, 2, 3, 4, 5])

with col2:
    st.subheader("Profil & Wilayah")
    sex = st.selectbox("Jenis Kelamin", ["male", "female"])
    smoker = st.selectbox("Apakah Merokok?", ["yes", "no"])
    region = st.selectbox("Wilayah Tempat Tinggal", ["southeast", "southwest", "northeast", "northwest"])

st.divider()

# 4. Preprocessing Input (Logika Encoding)
# Sesuaikan kolom ini dengan X_train.columns di notebook-mu
input_dict = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex': 1 if sex == 'male' else 0,
    'smoker': 1 if smoker == 'yes' else 0,
    'region_northwest': 1 if region == 'northwest' else 0,
    'region_southeast': 1 if region == 'southeast' else 0,
    'region_southwest': 1 if region == 'southwest' else 0
}

input_df = pd.DataFrame([input_dict])

# 5. Prediksi
if st.button("🚀 Hitung Estimasi Biaya", use_container_width=True):
    # Scaling data input
    input_scaled = scaler.transform(input_df)
    
    # Prediksi menggunakan model (Non-Log sesuai saran sebelumnya)
    prediction = model.predict(input_scaled)
    
    # Tampilkan Hasil
    st.success(f"### Estimasi Biaya Asuransi: **${prediction[0]:,.2f}**")
    
    # --- Analisa Bisnis Singkat ---
    with st.expander("Lihat Analisa Risiko"):
        if smoker == 'yes':
            st.warning("⚠️ **Risiko Merokok:** Status merokok adalah faktor terbesar yang meningkatkan premi Anda hingga 3-4 kali lipat.")
        
        if bmi >= 30:
            st.error(f"⚠️ **Kategori Obesitas (BMI: {bmi:.1f}):** BMI di atas 30 meningkatkan risiko penyakit kronis yang berdampak pada kenaikan premi.")
        elif 25 <= bmi < 30:
            st.warning(f"🔸 **Kategori Overweight (BMI: {bmi:.1f}):** Sedikit di atas normal, perhatikan pola makan untuk menjaga premi tetap rendah.")
        else:
            st.success(f"✅ **Kategori Sehat (BMI: {bmi:.1f}):** BMI Anda ideal, ini membantu menekan biaya asuransi.")
            