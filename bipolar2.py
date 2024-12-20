import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Fungsi untuk mengatur warna latar belakang aplikasi
def set_background_colors(outer_color, inner_color):
    st.markdown(
        f"""
        <style>
        html {{
            background-color: {outer_color};  /* Warna latar belakang luar */
        }}
        .appview-container {{
            background-color: {inner_color};  /* Warna latar konten */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Fungsi untuk memuat dan memproses dataset
def load_and_preprocess_data():
    try:
        df = pd.read_csv('bipolar.csv')  # Mencoba untuk memuat file CSV
    except FileNotFoundError:
        st.error("File 'bipolar.csv' tidak ditemukan. Pastikan file ada di direktori yang benar.")  # Pesan error jika file tidak ditemukan
        return None

    df = df.dropna(how='all')  # Menghapus baris yang memiliki semua nilai NaN

    df_conditions = df  # Dataset yang sudah diproses

    return df_conditions

# Fungsi untuk menangani nilai yang hilang dengan imputasi menggunakan mean atau nilai paling sering
def handle_missing_values(df):
    imputer_numeric = SimpleImputer(strategy='mean')  # Imputer untuk kolom numerik
    imputer_categorical = SimpleImputer(strategy='most_frequent')  # Imputer untuk kolom kategorikal

    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Menentukan kolom numerik
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns  # Menentukan kolom kategorikal

    if not numeric_cols.empty:
        df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])  # Imputasi untuk kolom numerik
    if not categorical_cols.empty:
        df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])  # Imputasi untuk kolom kategorikal

    return df

# Fungsi untuk mengkodekan variabel kategorikal menjadi numerik menggunakan LabelEncoder
def encode_categorical_variables(df):
    label_encoders = {}  # Kamus untuk menyimpan encoder label untuk setiap kolom
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns  # Menentukan kolom kategorikal

    for col in categorical_cols:
        le = LabelEncoder()  # Membuat encoder label untuk setiap kolom kategorikal
        df[col] = le.fit_transform(df[col].astype(str))  # Fit dan transform kolom
        label_encoders[col] = le  # Menyimpan encoder

    return df, label_encoders

# Fungsi untuk melakukan regresi linier
def perform_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Membagi data menjadi training dan testing

    model = LinearRegression()  # Inisialisasi model regresi linier
    model.fit(X_train, y_train)  # Melatih model dengan data training

    y_pred = model.predict(X_test)  # Memprediksi data testing

    mse = mean_squared_error(y_test, y_pred)  # Menghitung Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Menghitung nilai R-squared

    return model, mse, r2, X_train, X_test, y_train, y_test, y_pred

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.set_page_config(layout="wide")  # Mengatur layout aplikasi Streamlit
    st.title("📊 Analisis Mental Health Bipolar")  # Menampilkan judul aplikasi

    st.sidebar.header("Pengaturan Tampilan")  # Header untuk pengaturan tampilan di sidebar
    outer_background_color = st.sidebar.color_picker("Pilih Warna Latar Belakang Luar", "#f0f8ff")  # Pemilih warna untuk latar belakang luar
    inner_background_color = st.sidebar.color_picker("Pilih Warna Latar Belakang Konten", "#ffffff")  # Pemilih warna untuk latar belakang konten
    set_background_colors(outer_background_color, inner_background_color)  # Mengatur warna latar belakang berdasarkan pilihan pengguna

    df_conditions = load_and_preprocess_data()  # Memuat dan memproses dataset
    if df_conditions is None:
        return  # Menghentikan eksekusi jika file tidak ditemukan

    df_conditions = handle_missing_values(df_conditions)  # Menangani nilai yang hilang dalam dataset

    df_encoded, _ = encode_categorical_variables(df_conditions.copy())  # Mengkodekan variabel kategorikal

    # Menentukan tab untuk berbagai bagian aplikasi
    tab1, tab2, tab3, tab4 = st.tabs([
        "Ringkasan Dataset", 
        "Regresi Linier", 
        "Evaluasi Model", 
        "Visualisasi Prediksi"
    ])

    with tab1:
        st.header("Ringkasan Dataset")  # Tab untuk ringkasan dataset
        col1, col2, col3 = st.columns(3)  # Membuat tiga kolom untuk metrik
        col1.metric("Jumlah Baris", df_encoded.shape[0])  # Menampilkan jumlah baris
        col2.metric("Jumlah Kolom", df_encoded.shape[1])  # Menampilkan jumlah kolom
        col3.metric("Kolom Kategorik", len(df_encoded.select_dtypes(include=['object', 'category']).columns))  # Menampilkan jumlah kolom kategorikal

        st.dataframe(df_encoded)  # Menampilkan dataframe

    with tab2:
        st.header("Regresi Linier")  # Tab untuk regresi linier
        target_column = st.selectbox("Pilih Kolom Target", df_encoded.columns)  # Memilih kolom target
        feature_columns = st.multiselect("Pilih Kolom Fitur", df_encoded.columns.drop(target_column))  # Memilih kolom fitur

        if st.button("Jalankan Regresi"):  # Menjalankan regresi ketika tombol diklik
            X = df_encoded[feature_columns]  # Fitur
            y = df_encoded[target_column]  # Variabel target
            model, mse, r2, X_train, X_test, y_train, y_test, y_pred = perform_linear_regression(X, y)  # Melakukan regresi linier

            st.success(f"Model berhasil dilatih! MSE: {mse:.2f}, R²: {r2:.2f}")  # Menampilkan hasil regresi

    with tab3:
        st.header("Evaluasi Model")  # Tab untuk evaluasi model
        if 'y_test' in locals():  # Memeriksa apakah 'y_test' ada
            st.write("Data Aktual vs Prediksi")  # Menampilkan data aktual vs prediksi
            comparison_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})  # Membuat DataFrame untuk perbandingan
            st.line_chart(comparison_df)  # Menampilkan grafik garis untuk perbandingan

    with tab4:
        st.header("Visualisasi Prediksi")  # Tab untuk visualisasi prediksi
        if 'y_pred' in locals():  # Memeriksa apakah prediksi ada
            plt.figure(figsize=(10, 6))  # Membuat plot
            plt.scatter(y_test, y_pred)  # Plot titik aktual vs prediksi
            plt.xlabel("Nilai Aktual")  # Label sumbu X
            plt.ylabel("Nilai Prediksi")  # Label sumbu Y
            plt.title("Nilai Aktual vs Prediksi")  # Judul plot
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Menambahkan garis merah sebagai referensi
            st.pyplot(plt)  # Menampilkan plot

if __name__ == "__main__":
    main()  # Menjalankan aplikasi
