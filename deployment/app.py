import streamlit as st
import eda
import prediction

# membuat side bar untuk pilihan menu
page = st.sidebar.selectbox('Pilih Halaman : ', ('EDA', 'Prediction'))

# looping untuk menjalankan menu
if page == 'EDA':
    eda.template_eda()
else:
    prediction.template_prediction()