import pandas as pd
import numpy as np
import streamlit as st
import joblib

# title of your Web Application

#st.markdown(f'<body style="background-color:grey;">', unsafe_allow_html=True)
st.title('Sales Forecasting')

# describe the Web Application
#st.write('Demonstrate how to forecast advertising sales based on ad expenditure.')

# read data
data = pd.read_csv('data/advertising_regression.csv')

# show data
st.subheader('Data')
data = data.drop(columns="Unnamed: 0")
data

# create sidebar
st.sidebar.subheader('Advertising Costs')

# TV slider
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)
# Radio slider
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)
# Newspaper slider
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)

value = st.sidebar.selectbox("Show Distribution", ["Radio", "TV", "Newspaper"])

if value == 'TV':            
	label = 'TV Advertising Cost Distribution'
	hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]
elif value == 'Newspaper':             
	label = 'Newspaper Advertising Cost Distribution'
	hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]
else:
	label = 'Radio Advertising Cost Distribution'
	hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

st.subheader(label)
st.bar_chart(hist_values)

# Load saved machine learning model
saved_model = joblib.load('advertising_model.sav')

predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0]
st.write(f'Predicted sales is {predicted_sales} dollars.')
#st.markdown(f'</div>', unsafe_allow_html=True)
