import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_custom_css():
    css = """
    <style>
    body {
        background-color: #e5e1da; 
        font-family: 'Times New Roman', serif; 
        color: #B99470;
    }
    .stApp {
        background-color: #B5C0D0; 
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .stButton>button {
        background-color: #CCD5AE;
        color: rgb(204, 211, 202);
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #89A8B2; 
        transform: scale(1.05); 
    }
    .stSelectbox>div {
        background-color: #89A8B2; 
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs button {
        font-weight: bold;
        font-size: 15px;
        border-radius: 10px;
    }
    .stMarkdown h1 {
        text-align: center;
        color: #5C5470; 
        font-family: 'Times New Roman', serif;
        font-size: 48px; 
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.set_page_config(
    page_title="Mental Health Bipolar",
    page_icon="📊",
    layout="wide",
)

load_custom_css()

st.markdown("<h1>📊 Analis is Dataset Bipolar</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://img.freepik.com/free-vector/background-world-mental-health-day-awareness_52683-137722.jpg?t=st=1732555165~exp=1732558765~hmac=b06141202ae057de71e58265a35cb76810afb5bb395f3fa2d7ef00b085cf72e4&w=1060" width="300">
        <img src="https://img.freepik.com/premium-vector/mental-health-illustrations-flat-style_98292-17747.jpg?w=1380" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h3>Unggah dataset Anda (format CSV)</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📄 Data", "📊 Statistik", "🛠️ Kolom", "📈 Visualisasi", "📊 K-Means Clustering"]
        )

        with tab1:
            st.markdown("### Data yang diunggah:")
            st.dataframe(data, use_container_width=True)

        with tab2:
            st.markdown("### Statistik Deskriptif:")
            st.write(data.describe())

        with tab3:
            st.markdown("### Informasi Kolom:")
            col_info = pd.DataFrame({
                "Kolom": data.columns,
                "Tipe Data": data.dtypes,
                "Kosong (%)": (data.isnull().sum() / len(data)) * 100
            })
            st.table(col_info)

        with tab4:
            st.markdown("### Visualisasi Data")
            column_options = st.selectbox("Pilih kolom untuk visualisasi:", data.columns)
            if data[column_options].dtype in ["int64", "float64"]:
                fig = px.histogram(data, x=column_options, color_discrete_sequence=["#7e57c2"])
                st.plotly_chart(fig, use_container_width=True)
            elif data[column_options].dtype == "object":
                fig = px.bar(data[column_options].value_counts(), color_discrete_sequence=["#ff9a9e"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Kolom ini tidak dapat divisualisasikan.")

        with tab5:
            st.markdown("### K-Means Clustering")
            numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
            selected_columns = st.multiselect("Pilih kolom numerik:", numeric_columns)

            if len(selected_columns) >= 2:
                num_clusters = st.slider("Jumlah cluster (K):", min_value=2, max_value=10, value=3)

                imputer = SimpleImputer(strategy="mean")
                data[selected_columns] = imputer.fit_transform(data[selected_columns])

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[selected_columns])

                model = KMeans(n_clusters=num_clusters, random_state=42)
                data["Cluster"] = model.fit_predict(scaled_data)

                st.markdown("#### Hasil Clustering:")
                st.write(data["Cluster"].value_counts())

                fig = px.scatter(
                    x=scaled_data[:, 0],
                    y=scaled_data[:, 1],
                    color=data["Cluster"].astype(str),
                    title="Hasil K-Means Clustering",
                    labels={"x": selected_columns[0], "y": selected_columns[1]},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pilih setidaknya dua kolom numerik untuk melakukan clustering.")
else:
    st.info("Silakan unggah file dataset CSV untuk memulai.")
