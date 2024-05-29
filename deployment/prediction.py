import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

with open('list_num_cols.txt', 'r') as file_1:
    list_num_col = json.load(file_1)

with open('scaler.pkl', 'rb') as file_2:
    model_scaler = pickle.load(file_2)

with open('model_saving.pkl', 'rb') as file_3:
    model_svc = pickle.load(file_3)

def template_prediction():

    # membuat judul streamlit
    st.title('Simulasi Default Payment')

    # membuat form
    with st.form('Form Default Payment'):
        # set limit balance:
        limit_balance = st.number_input('Limit Balance :', help= 'Diisi dengan nilai limit balance tiap-tiap pemegang kartu.')

        # set gender
        sex = st.number_input('Gender :', min_value=1, max_value=2, help='1 untuk mewakili MALE\n2 untuk mewakili FEMALE')

        # set education level
        education = st.number_input('Education Level :', min_value=1, max_value=4, help='1 untuk mewakili graduate school\n2 untuk mewakili universitas\n3 untuk mewakili high school\n 4 untuk mewakili others')

        # set status pernikahan
        marital_status = st.number_input('Marital Status :', min_value=1, max_value=3, help='1 untuk mewakili status single\n2 untuk mewakili status menikah\n3 untuk mewakili status lainnya')

        # set umur
        age = st.number_input('Age :', min_value=17, max_value=100)

        # set status pembayaran
        pay_0 = st.number_input('Status pembayaran bulan terakhir :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')
        pay_2 = st.number_input('Status pembayaran bulan n-1 :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')
        pay_3 = st.number_input('Status pembayaran bulan n-2 :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')
        pay_4 = st.number_input('Status pembayaran bulan n-3 :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')
        pay_5 = st.number_input('Status pembayaran bulan n-4 :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')
        pay_6 = st.number_input('Status pembayaran bulan n-5 :', min_value=-3, max_value=5, help='Masukan kode status pembayaran!')

        # set jumlah tagihan
        bill_amt_1 = st.number_input('Jumlah tagihan bulan terakhir :', min_value=0, help='Masukan jumlah tagihan!')
        bill_amt_2 = st.number_input('Jumlah tagihan bulan n-1 :', min_value=0, help='Masukan jumlah tagihan!')
        bill_amt_3 = st.number_input('Jumlah tagihan bulan n-2 :', min_value=0, help='Masukan jumlah tagihan!')
        bill_amt_4 = st.number_input('Jumlah tagihan bulan n-3 :', min_value=0, help='Masukan jumlah tagihan!')
        bill_amt_5 = st.number_input('Jumlah tagihan bulan n-4 :', min_value=0, help='Masukan jumlah tagihan!')
        bill_amt_6 = st.number_input('Jumlah tagihan bulan n-5 :', min_value=0, help='Masukan jumlah tagihan!')

        # set jumlah pembayaran bulanan
        pay_amt_1 = st.number_input('Jumlah pembayaran bulan terakhir :', min_value=0, help='Masukan jumlah pembayaran!')
        pay_amt_2 = st.number_input('Jumlah pembayaran bulan n-1 :', min_value=0, help='Masukan jumlah pembayaran!')
        pay_amt_3 = st.number_input('Jumlah pembayaran bulan n-2 :', min_value=0, help='Masukan jumlah pembayaran!')
        pay_amt_4 = st.number_input('Jumlah pembayaran bulan n-3 :', min_value=0, help='Masukan jumlah pembayaran!')
        pay_amt_5 = st.number_input('Jumlah pembayaran bulan n-4 :', min_value=0, help='Masukan jumlah pembayaran!')
        pay_amt_6 = st.number_input('Jumlah pembayaran bulan n-5 :', min_value=0, help='Masukan jumlah pembayaran!')

        # membuat submitted button
        submitted = st.form_submit_button('Predict')

    # memasukan input kedalam data frame
    data_inf = {
        'limit_balance' : limit_balance,
        'sex' : sex,
        'education_level' : education,
        'marital_status' : marital_status,
        'age' : age,
        'pay_0' : pay_0,
        'pay_2' : pay_2,
        'pay_3' : pay_3,
        'pay_4' : pay_4,
        'pay_5' : pay_5,
        'pay_6' : pay_6,
        'bill_amt_1' : bill_amt_1,
        'bill_amt_2' : bill_amt_2,
        'bill_amt_3' : bill_amt_3,
        'bill_amt_4' : bill_amt_4,
        'bill_amt_5' : bill_amt_5,
        'bill_amt_6' : bill_amt_6,
        'pay_amt_1' : pay_amt_1,
        'pay_amt_2' : pay_amt_2,
        'pay_amt_3' : pay_amt_3,
        'pay_amt_4' : pay_amt_4,
        'pay_amt_5' : pay_amt_5,
        'pay_amt_6' : pay_amt_6,        
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # feature scaling
        data_final = model_scaler.transform(data_inf)

        # predict dengan model svc
        y_pred_inf = model_svc.predict(data_final)

        # mengeluarkan hasil predict
        st.write('## Default Payment :', str(int(y_pred_inf)))

if __name__ == '__main__':
    template_prediction()