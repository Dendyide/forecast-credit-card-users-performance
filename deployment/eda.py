import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def template_eda():
    # membuat judul streamlit
    st.title('Default Payment Credit Card')

    # menambahkan gambar
    image = Image.open('keuangan.jpg')
    st.image(image)

    # Show data frame
    data = pd.read_csv('P1G5_Set_1_dendy_dwinanda.csv')
    st.dataframe(data)

    # membuat histogram berdasarkan input user
    st.write('#### Tabel Histogram')
    option = st.selectbox('Pilihan column :', ('default_payment_next_month', 'limit_balance', 'sex', 'education_level', 'marital_status'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data[option], bins = 30, kde = True)
    st.pyplot(fig)

    # menambahkan deskripsi
    st.write('Page ini dibuat oleh DENDY DWINANDA')


if __name__ == '__main__':
    template_eda()