import pandas as pd
import numpy as np
import pickle
import streamlit as st
from logger import logging
import warnings
warnings.filterwarnings('ignore')


model_path = r'Models\logistic_regression_model.pkl'
scaler_path = r'Models\std_scaler.pkl'
le_path = r'Models\label_encoder.pkl'
ohe_path = r'Models\onehot_encoder.pkl'
ohe = pickle.load(open(ohe_path, 'rb'))
le = pickle.load(open(le_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

logging.info('Models and encoders loaded successfully.')


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
        logging.info('Form submitted for prediction.')

        geo_df = pd.DataFrame(ohe.transform([[st.session_state.country]]).toarray(), columns=ohe.categories_[0])
        logging.info(f'One-Hot Encoded Country')
        
        le_df = pd.DataFrame(le.transform([st.session_state.gender]), columns=['Gender'])
        logging.info(f'Label Encoded Gender')

        active_member = 1 if st.session_state.is_active_member == 'Yes' else 0
        input_data = np.array([[st.session_state.age, active_member, float(st.session_state.balance)]])
        input_df = pd.DataFrame(input_data, columns=['Age', 'IsActiveMember', 'Balance'])
        new_df = pd.concat([input_df, geo_df, le_df], axis=1)

        logging.info(f'Dropping column Spain to avoid dummy variable trap.')
        new_df.drop(columns=['Spain'], inplace=True, axis=1)

        model_input = new_df.copy()

        logging.info('Reordering columns to match training data.')
        model_input = model_input[['Age', 'Germany', 'Balance', 'France', 'Gender', 'IsActiveMember']]

        logging.info(f'Scaling Data')
        scaled_data = scaler.transform(model_input)

        result = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)

        logging.info(f'Prediction result: {"The customer is likely to churn." if result[0] == 1 else "The customer is not likely to churn."}')
        logging.info(f'Prediction Probability Not Churn: {prob[0][0]}')
        
        if result[0] == 1:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')
        st.write(f'Prediction Probability Not Churn: {prob[0][0]}')
