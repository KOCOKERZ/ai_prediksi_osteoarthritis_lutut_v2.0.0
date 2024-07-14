# Mari kita jelaskan kode di atas secara lengkap dan mendetail:

### Impor Modul dan Konfigurasi Awal

```python
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
from streamlit_option_menu import option_menu
```
- **`import streamlit as st`**: Mengimpor library Streamlit untuk membuat aplikasi web interaktif.
- **`import tensorflow as tf`**: Mengimpor TensorFlow, library untuk machine learning dan deep learning.
- **`from PIL import Image, ImageOps`**: Mengimpor modul dari Python Imaging Library untuk memanipulasi gambar.
- **`import numpy as np`**: Mengimpor NumPy untuk operasi numerik.
- **`import time`**: Mengimpor modul time untuk penundaan waktu.
- **`from streamlit_option_menu import option_menu`**: Mengimpor modul untuk membuat menu navigasi samping.

### Konfigurasi Halaman

```python
st.set_page_config(
    page_title="Prediksi Osteoarthritis pada Lutut",
    page_icon=":bone:",
    layout="centered",
    initial_sidebar_state='auto'
)
```
- **`st.set_page_config`**: Mengatur konfigurasi halaman Streamlit.
  - **`page_title`**: Judul halaman.
  - **`page_icon`**: Ikon halaman.
  - **`layout`**: Tata letak halaman (tengah).
  - **`initial_sidebar_state`**: Keadaan awal sidebar (otomatis).

### Menyembunyikan Elemen Streamlit

```python
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1fcdlh1 {padding: 2rem 1rem 10rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
```
- **`hide_streamlit_style`**: CSS untuk menyembunyikan menu utama dan footer.
- **`st.markdown`**: Menerapkan gaya CSS ke aplikasi.

### Membuat Sidebar dengan Menu Navigasi

```python
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predict", "About"],
        icons=["house", "search", "book"],
        menu_icon="cast",
        default_index=0
    )
```
- **`with st.sidebar:`**: Membuat elemen sidebar.
- **`option_menu`**: Membuat menu navigasi dengan opsi "Home", "Predict", dan "About" serta ikon terkait.

### Halaman "Home"

```python
if selected == "Home":
    st.title("Selamat Datang di Prediksi Osteoarthritis pada Lutut")
    st.write("""
        Osteoarthritis adalah kondisi kronis yang menyebabkan kerusakan pada sendi lutut, yang dapat menyebabkan rasa sakit dan penurunan fungsi. Aplikasi ini menggunakan AI untuk menganalisis gambar X-ray lutut dan memprediksi apakah ada tanda-tanda osteoarthritis.

        Cara penggunaan:
        1. Unggah gambar X-ray lutut Anda pada menu "Predict".
        2. AI akan menganalisis gambar dan memberikan prediksi mengenai kondisi lutut Anda.
    """)
```
- **`if selected == "Home":`**: Jika menu yang dipilih adalah "Home".
- **`st.title`**: Menampilkan judul halaman.
- **`st.write`**: Menampilkan deskripsi tentang aplikasi dan cara penggunaannya.

### Halaman "Predict"

```python
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
```
- **`if selected == "Predict":`**: Jika menu yang dipilih adalah "Predict".
- **`st.title`**: Menampilkan judul halaman.

#### Fungsi `import_and_predict`

```python
def import_and_predict(image_data, model):
    img_size = 150
    img = image_data.resize((img_size, img_size))
    img = np.asarray(img) / 255.0
    if img.shape[-1] == 1:
        img = np.stack((img,) * 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction
```
- **`img_size = 150`**: Ukuran gambar yang diubah menjadi 150x150 piksel.
- **`image_data.resize((img_size, img_size))`**: Mengubah ukuran gambar.
- **`np.asarray(img) / 255.0`**: Mengubah gambar menjadi array NumPy dan menormalisasi nilai piksel ke rentang [0, 1].
- **`np.stack((img,) * 3, axis=-1)`**: Jika gambar memiliki 1 channel (grayscale), gambar diubah menjadi gambar RGB dengan 3 channel.
- **`np.expand_dims(img, axis=0)`**: Menambahkan dimensi baru pada gambar untuk mencocokkan bentuk input model.
- **`model.predict(img)`**: Memperoleh prediksi dari model.

#### Fungsi `load_model`

```python
@st.cache_resource
def load_model():
    model_path = 'model.h5'
    return tf.keras.models.load_model(model_path)
```
- **`@st.cache_resource`**: Cache model agar tidak perlu memuat ulang setiap kali halaman diakses.
- **`model_path = 'model.h5'`**: Lokasi file model.
- **`tf.keras.models.load_model(model_path)`**: Memuat model dari file.

#### Proses Unggah dan Prediksi Gambar

```python
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
```
- **`st.file_uploader`**: Widget untuk mengunggah file gambar.
- **`if file is None:`**: Menampilkan pesan jika tidak ada file yang diunggah.
- **`Image.open(file)`**: Membuka file gambar.
- **`st.image`**: Menampilkan gambar yang diunggah.
- **`with st.spinner('Tunggu sebentar...'):`**: Menampilkan spinner selama proses prediksi.
- **`time.sleep(4)`**: Menunggu 4 detik (simulasi waktu pemrosesan).
- **`import_and_predict(image, model)`**: Mendapatkan prediksi dari model.
- **`np.argmax(predictions)`**: Mendapatkan indeks kelas dengan probabilitas tertinggi.
- **`class_names`**: Daftar nama kelas.
- **`string`**: Pesan prediksi.
- **`st.balloons()`**: Menampilkan balon jika prediksi "Normal".
- **`st.sidebar.success(string)`**: Menampilkan pesan sukses di sidebar.
- **`st.sidebar.warning(string)`**: Menampilkan pesan peringatan di sidebar.

### Halaman "About"

```python
if selected == "About":
    st.title("Tentang Pembuat AI")
    st.write("""
        Aplikasi ini dibuat oleh Fitrah Ali Akbar Setiawan (https://github.com/KOCOKERZ) yang memiliki minat besar dalam pengembangan AI. Anda dapat melihat proyek-proyek lainnya di GitHub saya.
    """)
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KOCOKERZ)")
```
- **`if selected == "About":`**: Jika menu yang dipilih adalah "About".
- **`st.title`**: Menampilkan judul halaman.
- **`st.write`**: Menampilkan informasi tentang pembuat aplikasi.
- **`st.markdown`**: Menampilkan tautan ke GitHub pembuat.
