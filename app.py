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
        .stDataFrame th { text-align: center; font-weight: bold; padding: 5px;}
        .stDataFrame td { text-align: right; padding: 5px;}
    </style>
""", unsafe_allow_html=True)

MODEL_DATA_FOLDER = "model/" 

def softmax_streamlit(z_input):
    z = np.array(z_input); 
    if z.ndim == 1: z = z.reshape(1, -1)
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def predict_binary_aspect_model_streamlit(X_vectorized_dense, weights, bias, threshold=0.5):
    if X_vectorized_dense.shape[0] == 0: return np.array([]), np.array([])
    logits_all_classes = np.dot(X_vectorized_dense, weights) + bias
    probs_all_classes = softmax_streamlit(logits_all_classes) 
    prob_aspect_present = probs_all_classes[:, 1] if probs_all_classes.shape[1] == 2 else np.array([0.0] * X_vectorized_dense.shape[0])
    predictions = (prob_aspect_present >= threshold).astype(int)
    confidences = np.where(predictions == 1, prob_aspect_present, 1 - prob_aspect_present)
    return predictions, confidences

def predict_sentiment_model_streamlit(X_vectorized_dense, weights, bias):
    if X_vectorized_dense.shape[0] == 0: return np.array([]), np.array([]) 
    logits = np.dot(X_vectorized_dense, weights) + bias
    probs = softmax_streamlit(logits)
    return np.argmax(probs, axis=1), np.max(probs, axis=1)

vectorizer_global = None
try:
    vectorizer_path = os.path.join(MODEL_DATA_FOLDER, "vectorizer_global_fitted.pkl") 
    vectorizer_global = joblib.load(vectorizer_path)
except: st.error(f"Vectorizer global '{vectorizer_path}' tidak ditemukan."); st.stop()

ASPEK_DENGAN_MODEL_DETEKSI_BINER = ['Layanan', 'Fitur', 'Kebermanfaatan', 'Bisnis', 'Non Aspek'] 
ASPEK_LIST_LENGKAP_UNTUK_SENTIMEN = ASPEK_DENGAN_MODEL_DETEKSI_BINER

models_detection_binary_params = {}
for aspek_det in ASPEK_DENGAN_MODEL_DETEKSI_BINER:
    base_fn = f"model_detection_binary_{aspek_det.replace(' ', '_').lower()}"
    try:
        w = np.load(os.path.join(MODEL_DATA_FOLDER, f"{base_fn}_weights.npy"))
        b = np.load(os.path.join(MODEL_DATA_FOLDER, f"{base_fn}_bias.npy"))
        models_detection_binary_params[aspek_det] = {'weights': w, 'bias': b}
    except: models_detection_binary_params[aspek_det] = None

models_sentiment_params = {} 
for aspek_sent in ASPEK_LIST_LENGKAP_UNTUK_SENTIMEN:
    base_fn = f"model_{aspek_sent.replace(' ', '_').lower()}"
    try:
        w = np.load(os.path.join(MODEL_DATA_FOLDER, f"{base_fn}_weights.npy"))
        b = np.load(os.path.join(MODEL_DATA_FOLDER, f"{base_fn}_bias.npy"))
        models_sentiment_params[aspek_sent] = {'weights': w, 'bias': b}
    except: models_sentiment_params[aspek_sent] = None

page_options = ["Home", "Klasifikasi Ulasan Multi-Label"]
page = st.sidebar.radio("Navigasi Halaman:", page_options)

if page == "Home":
    st.markdown('<div class="blue-header"><h1>Analisis Sentimen Multi-Aspek</h1></div>', unsafe_allow_html=True)
    st.write("Aplikasi ini mendeteksi beberapa aspek dalam ulasan dan menganalisis sentimennya.")
    st.caption("Dibuat oleh Kelompok 3")

elif page == "Klasifikasi Ulasan Multi-Label":
    st.markdown('<div class="blue-header"><h1>Klasifikasi Sentimen Ulasan (Multi-Label)</h1></div>', unsafe_allow_html=True)
    text_input = st.text_area("Masukkan teks ulasan Anda:", height=150)

    if st.button("Klasifikasikan"):
        if not text_input.strip(): st.warning("Masukkan teks ulasan."); st.stop()
        
        with st.spinner("Menganalisis..."):
            cleaned_text = text_input.lower().strip()
            if not vectorizer_global: st.error("Vectorizer belum siap."); st.stop()
            vectorized_input_dense = vectorizer_global.transform([cleaned_text]).toarray()

            st.subheader("Langkah 1: Deteksi Aspek (Multi-Label)")
            detected_aspects_from_binary_models = [] 
            detection_confidences_display = {}

            for aspek_candidate in ASPEK_DENGAN_MODEL_DETEKSI_BINER: 
                params = models_detection_binary_params.get(aspek_candidate)
                if params:
                    try:
                        pred_binary_array, conf_binary_array = predict_binary_aspect_model_streamlit(
                            vectorized_input_dense, params['weights'], params['bias']
                        )
                        if pred_binary_array.size > 0 and pred_binary_array[0] == 1: 
                            detected_aspects_from_binary_models.append(aspek_candidate)
                            detection_confidences_display[aspek_candidate] = f"{conf_binary_array[0]:.2f}"
                    except Exception as e_det_pred:
                        st.caption(f"Peringatan: Error saat deteksi aspek '{aspek_candidate}': {e_det_pred}")
            
            final_aspects_to_analyze_sentiment = []
            if detected_aspects_from_binary_models:
                final_aspects_to_analyze_sentiment = list(set(detected_aspects_from_binary_models)) # Ambil unik
                st.info(f"Aspek terdeteksi oleh model biner: **{', '.join(final_aspects_to_analyze_sentiment)}**")
            else: 
                if 'Non Aspek' in ASPEK_LIST_LENGKAP_UNTUK_SENTIMEN: 
                    final_aspects_to_analyze_sentiment = ['Non Aspek']
                    st.info("Tidak ada aspek spesifik terdeteksi oleh model manapun. Dianalisis sebagai: **Non Aspek** (fallback)")
                else:
                    st.info("Tidak ada aspek yang terdeteksi, dan 'Non Aspek' tidak tersedia untuk fallback.")

            st.subheader("Langkah 2: Klasifikasi Sentimen untuk Aspek Teridentifikasi")
            results_dict_sentiment = {}
            sentiments_overall_summary = {'Positif': 0, 'Non-Sentimen': 0, 'Negatif': 0}
            all_final_sentiment_confidences = []
            sentiment_pred_to_label_map = {0: 'Negatif', 1: 'Non-Sentimen', 2: 'Positif'}

            if final_aspects_to_analyze_sentiment:
                for aspek_to_analyze_sentiment in final_aspects_to_analyze_sentiment: 
                    model_sent_params = models_sentiment_params.get(aspek_to_analyze_sentiment)
                    if model_sent_params:
                        try:
                            pred_sent_num_arr, conf_sent_arr = predict_sentiment_model_streamlit(
                                vectorized_input_dense, model_sent_params['weights'], model_sent_params['bias']
                            )
                            pred_sent_num = pred_sent_num_arr[0]; conf_sent = conf_sent_arr[0]
                            pred_sent_label = sentiment_pred_to_label_map.get(pred_sent_num, 'Tidak Diketahui')
                            results_dict_sentiment[aspek_to_analyze_sentiment] = {'Label Sentimen': pred_sent_label, 'Skor Kepercayaan': f"{conf_sent:.2f}"}
                            if pred_sent_label in sentiments_overall_summary: sentiments_overall_summary[pred_sent_label] += 1
                            all_final_sentiment_confidences.append({'Aspek': aspek_to_analyze_sentiment, 'Confidence': conf_sent})
                        except Exception:
                            results_dict_sentiment[aspek_to_analyze_sentiment] = {'Label Sentimen': 'Error Prediksi', 'Skor Kepercayaan': '-'}
                    else:
                        results_dict_sentiment[aspek_to_analyze_sentiment] = {'Label Sentimen': 'Model Sentimen Tdk Tersedia', 'Skor Kepercayaan': '-'}
            
            st.subheader("Hasil Klasifikasi Sentimen")
            if results_dict_sentiment:
                df_final_results = pd.DataFrame.from_dict(results_dict_sentiment, orient='index').reset_index().rename(columns={'index': 'Aspek'})
                st.dataframe(df_final_results[['Aspek', 'Label Sentimen', 'Skor Kepercayaan']], use_container_width=True, hide_index=True)
            else:
                st.write("Tidak ada hasil sentimen untuk ditampilkan.")