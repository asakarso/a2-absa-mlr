import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Aspect-Based Sentiment", layout="wide")

st.markdown("""
    <style>
        .blue-header { background-color: #1f77b4; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .blue-header h1 { color: white; text-align: center; margin: 0; font-size: 1.8rem; }
        .stDataFrame table { font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

MODEL_DATA_FOLDER = "model/"

def softmax_streamlit(z_input):
    z = np.array(z_input)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def predict_binary_aspect_model_streamlit(X_vectorized_dense, weights, bias, threshold=0.5):
    if X_vectorized_dense.shape[0] == 0:
        return np.array([]), np.array([])
    logits_all_classes = np.dot(X_vectorized_dense, weights) + bias
    probs_all_classes = softmax_streamlit(logits_all_classes)
    prob_aspect_present = probs_all_classes[:, 1] if probs_all_classes.shape[1] == 2 else np.array([0.0])
    predictions = (prob_aspect_present >= threshold).astype(int)
    confidences = np.where(predictions == 1, prob_aspect_present, 1 - prob_aspect_present)
    return predictions, confidences

def predict_sentiment_model_streamlit(X_vectorized_dense, weights, bias):
    logits = np.dot(X_vectorized_dense, weights) + bias
    probs = softmax_streamlit(logits)
    return np.argmax(probs, axis=1), np.max(probs, axis=1)

try:
    vectorizer_global = joblib.load(os.path.join(MODEL_DATA_FOLDER, "vectorizer_global_fitted.pkl"))
except:
    st.error("Vectorizer tidak ditemukan.")
    st.stop()

ASPEK_LIST = ['Layanan', 'Fitur', 'Kebermanfaatan', 'Bisnis', 'Non Aspek']

# Load model deteksi aspek
models_detection = {}
models_sentiment = {}
for aspek in ASPEK_LIST:
    try:
        models_detection[aspek] = {
            'weights': np.load(f"{MODEL_DATA_FOLDER}/model_detection_binary_{aspek.lower().replace(' ', '_')}_weights.npy"),
            'bias': np.load(f"{MODEL_DATA_FOLDER}/model_detection_binary_{aspek.lower().replace(' ', '_')}_bias.npy")
        }
    except:
        models_detection[aspek] = None

    try:
        models_sentiment[aspek] = {
            'weights': np.load(f"{MODEL_DATA_FOLDER}/model_{aspek.lower().replace(' ', '_')}_weights.npy"),
            'bias': np.load(f"{MODEL_DATA_FOLDER}/model_{aspek.lower().replace(' ', '_')}_bias.npy")
        }
    except:
        models_sentiment[aspek] = None

# Navigasi
page = st.sidebar.radio("Navigasi Halaman:", ["Home", "Klasifikasi Ulasan"])

# ------------------------ HOME ------------------------
if page == "Home":
    st.markdown('<div class="blue-header"><h1>Aspect-Based Sentiment Analysis (ABSA)</h1></div>', unsafe_allow_html=True)
    st.write("""
    Selamat datang di aplikasi **Aspect-Based Sentiment Analysis (ABSA)**!  
    Sistem ini dikembangkan sebagai bagian dari *Project Kasus Satu* yang mengimplementasikan **Multinomial Logistic Regression** dengan **Grid Search Optimization** untuk menganalisis ulasan pengguna terhadap aplikasi marketplace Indonesia.
    """)
    st.markdown("#### Tujuan Utama")
    st.write("""
    1. Mengidentifikasi aspek yang terkandung dalam setiap ulasan pengguna.  
    2. Mengklasifikasikan sentimen dari masing-masing aspek yang terdeteksi.  
    3. Menyediakan visualisasi dan ringkasan hasil secara interaktif.  
    4. Memberikan kemudahan ekspor hasil klasifikasi.
    """)
    st.caption("Dikembangkan oleh **Kelompok 3A** sebagai bagian dari Project Kasus 1.")

# ------------------------ KLASIFIKASI ------------------------
elif page == "Klasifikasi Ulasan":
    st.markdown('<div class="blue-header"><h1>Klasifikasi Aspect Based Sentiment Analysis</h1></div>', unsafe_allow_html=True)
    ulasan = st.text_area("Masukkan teks ulasan Anda:")

    if st.button("Klasifikasikan") and ulasan.strip():
        cleaned = ulasan.strip().lower()
        X_dense = vectorizer_global.transform([cleaned]).toarray()

        # Deteksi Aspek
        st.subheader("Langkah 1: Deteksi Aspek")
        aspek_terdeteksi = []
        for aspek, model in models_detection.items():
            if model:
                pred, conf = predict_binary_aspect_model_streamlit(X_dense, model['weights'], model['bias'])
                if pred[0] == 1:
                    aspek_terdeteksi.append(aspek)

        if not aspek_terdeteksi:
            aspek_terdeteksi = ['Non Aspek']
            st.info("Tidak ada aspek spesifik terdeteksi. Akan dianalisis sebagai **Non Aspek**.")
        else:
            st.success("Aspek yang terdeteksi: **" + ", ".join(aspek_terdeteksi) + "**")

        # Klasifikasi Sentimen
        st.subheader("Langkah 2: Klasifikasi Sentimen")
        label_map = {0: 'Negatif', 1: 'Non-Sentimen', 2: 'Positif'}
        hasil = {}
        for aspek in aspek_terdeteksi:
            model = models_sentiment.get(aspek)
            if model:
                pred, conf = predict_sentiment_model_streamlit(X_dense, model['weights'], model['bias'])
                hasil[aspek] = {
                    "Label Sentimen": label_map.get(pred[0], "Tidak Diketahui"),
                    "Skor Kepercayaan": f"{conf[0]:.2f}"
                }
            else:
                hasil[aspek] = {"Label Sentimen": "Model Tidak Tersedia", "Skor Kepercayaan": "-"}

        df_hasil = pd.DataFrame.from_dict(hasil, orient='index').reset_index().rename(columns={'index': 'Aspek'})
        st.subheader("Hasil Klasifikasi")
        st.dataframe(df_hasil, use_container_width=True)

        # Visualisasi Confidence
        st.subheader("Visualisasi Confidence per Aspek")
        df_conf = df_hasil.copy()
        df_conf["Confidence"] = df_conf["Skor Kepercayaan"].apply(lambda x: float(x) if x != "-" else 0)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(df_conf["Aspek"], df_conf["Confidence"], color='skyblue')
        for bar, label in zip(bars, df_conf["Label Sentimen"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, label,
                    ha='center', va='bottom')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

        # Download button
        st.download_button("📥 Unduh Hasil sebagai CSV", df_hasil.to_csv(index=False).encode("utf-8"),
                           file_name="hasil_sentimen.csv", mime="text/csv")

        # Riwayat
        st.session_state.history.append({"ulasan": ulasan, "hasil": df_hasil})
        if st.session_state.history:
            st.subheader("Riwayat Analisis Sebelumnya")
            for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                with st.expander(f"Ulasan #{len(st.session_state.history)-idx+1}"):
                    st.text_area("Teks Ulasan", item['ulasan'], height=100, disabled=True, key=f"ulasan_{idx}")
                    st.dataframe(item['hasil'], use_container_width=True)

    # Masukan
    st.markdown("---")
    st.subheader("Berikan Masukan")
    saran = st.text_area("Masukan atau saran Anda:")
    if st.button("Kirim Masukan"):
        if saran.strip():
            st.success("Terima kasih atas masukannya!")
        else:
            st.warning("Masukan tidak boleh kosong.")
