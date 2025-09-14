from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from logger import logging
import warnings
warnings.filterwarnings('ignore')


logging.info("Loading models and encoders.")
model_path = r'Models\logistic_regression_model.pkl'
scaler_path = r'Models\std_scaler.pkl'
le_path = r'Models\label_encoder.pkl'
ohe_path = r'Models\onehot_encoder.pkl'
ohe = pickle.load(open(ohe_path, 'rb'))
le = pickle.load(open(le_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))


application = Flask(__name__)


@application.route('/')
def home():
    logging.info('Home route accessed.')
    return render_template('index.html')


@application.route('/details')
def details():
    logging.info('Details route accessed.')
    return render_template('details.html')


@application.route('/predict', methods=['POST'])
def predict_churn():
    logging.info('Predict route accessed.')
    # data = request.get_json(force=True)
    # print(data)
    # logging.info(f'Received data for prediction: {data}')
    age = request.form['age']
    balance = request.form['balance']
    country = request.form['geography']
    gender = request.form['gender']
    is_active_member = request.form['is_active_member']

    geo_df = pd.DataFrame(ohe.transform([[country]]).toarray(), columns=ohe.categories_[0])
    le_df = pd.DataFrame(le.transform([gender]), columns=['Gender'])

    input_df = pd.DataFrame([[int(age), int(is_active_member), float(balance)]], columns=['Age', 'IsActiveMember', 'Balance'])
    new_df = pd.concat([input_df, geo_df, le_df], axis=1)

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

    return render_template('result.html', prediction=result[0], probability=prob[0][0])


if __name__ == '__main__':
    logging.info('Running Flask applicationlication.')
    application.run(host='0.0.0.0')