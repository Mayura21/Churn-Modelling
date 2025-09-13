import pandas as pd
import numpy as np
import pickle
import streamlit as st
from logger import logging

# df = pd.read_csv('dataset/Churn_Modelling.csv')

# df.drop(axis=1, columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

model_path = 'Models\logistic_regression_model.pkl'
scaler_path = 'Models\std_scaler.pkl'
le_path = 'Models\label_encoder.pkl'
ohe_path = 'Models\onehot_encoder.pkl'
ohe = pickle.load(open(ohe_path, 'rb'))
le = pickle.load(open(le_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
logging.info('Models and encoders loaded successfully.')



columns_considered = ['Exited', 'Age', 'Germany', 'Balance', 'France', 'Gender', 'IsActiveMember']

# data = df[columns_considered]

st.title('Customer Churn Prediction')
st.write('Enter the customer details to predict if they will churn or not.')

st.selectbox('Country', ('France', 'Germany', 'Spain'), key='country')
st.number_input('Age', min_value=18, max_value=92, value=30, step=1, key='age')
st.selectbox('Gender', ('Male', 'Female'), key='gender')
st.radio('Is Active Member', ('Yes', 'No'), key='is_active_member')
st.text_input('Balance', value='0.0', key='balance')

with st.form(key='churn_form'):
    submit_button = st.form_submit_button(label='Predict Churn')
    if submit_button:
        geo_df = pd.DataFrame(ohe.transform([[st.session_state.country]]).toarray(), columns=ohe.categories_[0])
        le_df = pd.DataFrame(le.transform([st.session_state.gender]), columns=le.classes_)
        input_data = np.array([[st.session_state.age, st.session_state.is_active_member, float(st.session_state.balance)]])
        input_df = pd.DataFrame(input_data, columns=['Age', 'IsActiveMember', 'Balance',])
        new_df = pd.concat([input_df, geo_df, le_df], axis=1)
        model_input = input_df.copy()
        print(model_input)
        logging.INFO(f'Model Input: {model_input}')
        result = model.predict(scaler.transform(model_input))
        if result[0] == 1:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')