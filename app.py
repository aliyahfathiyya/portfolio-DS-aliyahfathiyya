import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Pipeline (Sudah termasuk scaler + model)
# 1. Load Pipeline
try:
    # Menggunakan forward slash agar tidak terjadi escape character error
    with open('data/insurance_pipeline.sav', 'rb') as f:
        model_pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("❌ File 'insurance_pipeline.sav' tidak ditemukan di folder data!")
    st.stop()  # Berhenti di sini jika file tidak ada
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat model: {e}")
    st.stop()
    
# 2. Judul & Header
st.set_page_config(page_title="Insurance Predictor", page_icon="💰")
st.title("💰 Insurance Charge Predictor")
st.write("Aplikasi prediksi biaya asuransi kesehatan berbasis Machine Learning.")

# 3. Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Fisik")
    age = st.number_input("Umur", min_value=1, max_value=100, value=25)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, value=60.0)
    height_cm = st.number_input("Tinggi Badan (cm)", min_value=100.0, value=165.0)
    
    # Hitung BMI otomatis
    bmi = weight / ((height_cm/100)**2)
    st.info(f"**BMI Anda: {bmi:.2f}**")
    
    children = st.selectbox("Jumlah Anak", [0, 1, 2, 3, 4, 5])

with col2:
    st.subheader("Profil & Wilayah")
    sex = st.selectbox("Jenis Kelamin", ["male", "female"])
    smoker = st.selectbox("Apakah Merokok?", ["yes", "no"])
    region = st.selectbox("Wilayah", ["northeast", "northwest", "southeast", "southwest"])

# 4. Proses Prediksi
# 4. Proses Prediksi
if st.button("🚀 Hitung Estimasi Biaya", use_container_width=True):
    # Mapping manual sesuai yang diminta model (asumsi label encoding)
    # Sesuaikan angka ini (0/1) dengan saat kamu training!
    gender_map = {'male': 1, 'female': 0}
    smoker_map = {'yes': 1, 'no': 0}
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [gender_map[sex]],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_map[smoker]],
        'region': [region_map[region]]
    })
    
    # Sekarang input_data berisi angka semua, bukan string 'male' lagi
    pred_log = model_pipeline.predict(input_data)
    final_price = np.expm1(pred_log)[0]
    
    st.success(f"### Estimasi Biaya Asuransi: **${final_price:,.2f}**")
    # Analisa Bisnis Singkat
    
    with st.expander("Lihat Analisa Risiko"):
        if smoker == 'yes':
            st.warning("⚠️ **Perokok:** Biaya meningkat tajam karena risiko kesehatan tinggi.")
        if bmi >= 30:
            st.error("⚠️ **Obesitas:** Disarankan memulai gaya hidup sehat.")