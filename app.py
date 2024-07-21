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

        Osteoarthritis adalah kondisi yang menyebabkan kerusakan pada sendi lutut, yang dapat menyebabkan rasa sakit dan penurunan fungsi. Penyakit ini menyebabkan tulang rawan yang melindungi ujung-ujung tulang di sendi menjadi rusak, yang dapat menyebabkan rasa sakit, bengkak, dan keterbatasan gerak.

        - **Penyebab:**
        Penyebab osteoarthritis meliputi penuaan, obesitas, cedera sendi, dan faktor genetik. Aktivitas fisik yang berlebihan atau pekerjaan yang memberi tekanan berulang pada sendi juga dapat berkontribusi.

        - **Gejala:**
        Gejala utama osteoarthritis adalah nyeri sendi dan kekakuan, terutama setelah beristirahat atau tidak bergerak. Gejala lainnya termasuk bengkak, kehilangan fleksibilitas, dan suara 'klik' atau 'retak' saat menggerakkan sendi.

        - **Pengobatan:**
        Pengobatan osteoarthritis meliputi kombinasi dari perubahan gaya hidup, obat-obatan, terapi fisik, dan dalam beberapa kasus, pembedahan. Tujuannya adalah untuk mengurangi rasa sakit, meningkatkan fungsi sendi, dan memperlambat perkembangan penyakit.

        **Aplikasi ini menggunakan AI untuk menganalisis gambar X-ray lutut dan memprediksi apakah ada tanda-tanda osteoarthritis.**

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
        'Normal': 'Selamat! Anda tidak memiliki tanda-tanda osteoarthritis. Lutut anda tampak normal tanpa tanda-tanda osteoartritis. Untuk menjaga kesehatan lutut Anda, pastikan untuk berolahraga secara teratur, menjaga berat badan ideal, dan menghindari cedera pada lutut.',
        'Osteoarthritis': 'Ada tanda-tanda osteoartritis. Disarankan untuk menghindari aktivitas yang membebani lutut, mengonsumsi makanan yang baik untuk kesehatan sendi, melakukan fisioterapi sesuai anjuran dokter, dan  segera periksakan diri anda ke dokter untuk penanganan lebih lanjut'
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

        prediction_text = "Prediksi : " + class_names[predicted_class]
        st.sidebar.write(prediction_text)

        if class_names[predicted_class] == 'Normal':
            st.balloons()
            st.sidebar.success(prediction_text)
            st.sidebar.info(descriptions['Normal'])
        elif class_names[predicted_class] == 'Osteoarthritis':
            st.sidebar.warning(prediction_text)
            st.sidebar.error(descriptions['Osteoarthritis'])

if selected == "About":
    st.title("Tentang Pembuat AI")
    st.write("""
        Aplikasi ini dibuat oleh mahasiswa D4 jurusan Teknik Informatika Universitas Logistik dan Bisnis Internasional. Berikut merupakan mahasiswa yang terlibat dalam pengembangan aplikasi ini:
    """)

    def display_image_with_aspect_ratio(image_path, width, height):
        image = Image.open(image_path)
        image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
        st.image(image, width=200)

    st.markdown("**1. Fitrah Ali Akbar Setiawan**")
    display_image_with_aspect_ratio("img/akbar.png", 300, 400)
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KOCOKERZ)")

    st.markdown("**2. Megah Juliardi Sondara Wicaksana**")
    display_image_with_aspect_ratio("img/megah.png", 300, 400)
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/juliardimegah)")

