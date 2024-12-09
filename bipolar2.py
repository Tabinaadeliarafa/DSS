import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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

def load_and_preprocess_data():
    try:
        df = pd.read_csv('bipolar.csv')
    except FileNotFoundError:
        st.error("File 'bipolar.csv' tidak ditemukan. Pastikan file ada di direktori yang benar.")
        return None

    df = df.dropna(how='all')

    df_conditions = df

    return df_conditions

def handle_missing_values(df):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if not numeric_cols.empty:
        df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
    if not categorical_cols.empty:
        df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    return df

def encode_categorical_variables(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

def perform_linear_regression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_train, X_test, y_train, y_test, y_pred

# Streamlit
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Analisis Mental Health Bipolar")

    st.sidebar.header("Pengaturan Tampilan")
    outer_background_color = st.sidebar.color_picker("Pilih Warna Latar Belakang Luar", "#f0f8ff")
    inner_background_color = st.sidebar.color_picker("Pilih Warna Latar Belakang Konten", "#ffffff")
    set_background_colors(outer_background_color, inner_background_color)

    df_conditions = load_and_preprocess_data()
    if df_conditions is None:
        return  

    df_conditions = handle_missing_values(df_conditions)

    df_encoded, _ = encode_categorical_variables(df_conditions.copy())

    tab1, tab2, tab3, tab4 = st.tabs([
        "Ringkasan Dataset", 
        "Regresi Linier", 
        "Evaluasi Model", 
        "Visualisasi Prediksi"
    ])

    with tab1:
        st.header("Ringkasan Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df_encoded.shape[0])
        col2.metric("Jumlah Kolom", df_encoded.shape[1])
        col3.metric("Kolom Kategorik", len(df_encoded.select_dtypes(include=['object', 'category']).columns))

        st.dataframe(df_encoded)

    with tab2:
        st.header("Regresi Linier")
        target_column = st.selectbox("Pilih Kolom Target", df_encoded.columns)
        feature_columns = st.multiselect("Pilih Kolom Fitur", df_encoded.columns.drop(target_column))

        if st.button("Jalankan Regresi"):
            X = df_encoded[feature_columns]
            y = df_encoded[target_column]
            model, mse, r2, X_train, X_test, y_train, y_test, y_pred = perform_linear_regression(X, y)

            st.success(f"Model berhasil dilatih! MSE: {mse:.2f}, R²: {r2:.2f}")

    with tab3:
        st.header("Evaluasi Model")
        if 'y_test' in locals():
            st.write("Data Aktual vs Prediksi")
            comparison_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
            st.line_chart(comparison_df)

    with tab4:
        st.header("Visualisasi Prediksi")
        if 'y_pred' in locals():
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred)
            plt.xlabel("Nilai Aktual")
            plt.ylabel("Nilai Prediksi")
            plt.title("Nilai Aktual vs Prediksi")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
            st.pyplot(plt)

if __name__ == "__main__":
    main()
