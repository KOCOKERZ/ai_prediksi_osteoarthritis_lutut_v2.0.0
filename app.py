import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Prediksi Osteoarthritis pada Lutut",
    page_icon=":bone:",
    layout="centered",
    initial_sidebar_state='auto'
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1fcdlh1 {padding: 2rem 1rem 10rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict", "About"],
        icons=["house", "search", "book"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Home":
    st.title("Selamat Datang di Prediksi Osteoarthritis pada Lutut")
    st.write("""
        Osteoarthritis adalah kondisi kronis yang menyebabkan kerusakan pada sendi lutut, yang dapat menyebabkan rasa sakit dan penurunan fungsi. Aplikasi ini menggunakan AI untuk menganalisis gambar X-ray lutut dan memprediksi apakah ada tanda-tanda osteoarthritis.

        Cara penggunaan:
        1. Unggah gambar X-ray lutut Anda pada menu "Predict".
        2. AI akan menganalisis gambar dan memberikan prediksi mengenai kondisi lutut Anda.
    """)

if selected == "Predict":
    st.title("Prediksi Osteoarthritis pada Lutut")

    def import_and_predict(image_data, model):
        img_size = 150
        img = image_data.resize((img_size, img_size))
        img = np.asarray(img) / 255.0
        if img.shape[-1] == 1:
            img = np.stack((img,) * 3, axis=-1)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        return prediction

    @st.cache_resource
    def load_model():
        model_path = 'model.h5'
        return tf.keras.models.load_model(model_path)

    model = load_model()

    categories = ['Normal', 'Osteoarthritis']
    descriptions = {
        'Normal': 'Lutut tampak normal tanpa tanda-tanda osteoartritis.',
        'Osteoarthritis': 'Ada tanda-tanda parah osteoartritis, segera periksakan diri anda ke dokter untuk penanganan lebih lanjut.'
    }

    file = st.file_uploader("Unggah gambar X-ray lutut, dan AI akan memprediksi kondisinya", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Silakan unggah file gambar")
    else:
        image = Image.open(file)
        st.image(image, caption='Gambar Terunggah', use_column_width=True)
        st.write("")

        with st.spinner('Tunggu sebentar...'):
            time.sleep(4)
            predictions = import_and_predict(image, model)
            predicted_class = np.argmax(predictions)

        class_names = ['Normal', 'Osteoarthritis']

        string = "Prediksi : " + class_names[np.argmax(predictions)]
        if class_names[np.argmax(predictions)] == 'Normal':
            st.balloons()
            st.sidebar.success(string)
        elif class_names[np.argmax(predictions)] == 'Osteoarthritis':
            st.sidebar.warning(string)

if selected == "About":
    st.title("Tentang Pembuat AI")
    st.write("""
        Aplikasi ini dibuat oleh Fitrah Ali Akbar Setiawan (https://github.com/KOCOKERZ) yang memiliki minat besar dalam pengembangan AI. Anda dapat melihat proyek-proyek lainnya di GitHub saya.
    """)
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KOCOKERZ)")
