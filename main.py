from flask import Flask, request, redirect, url_for, jsonify, render_template
import numpy as np
import pandas as pd
import json

from flask_mysqldb import MySQL

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from catboost import CatBoostRegressor
import tensorflow as tf
import tensorflow_probability as tfp

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mohs_hardness_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

features = [ 'log10_allelectrons_Total', 'log10_density_Total',
       'log10_allelectrons_Average', 'log10_val_e_Average',
       'log10_atomicweight_Average', 'log10_ionenergy_Average',
       'log10_el_neg_chi_Average', 'log10_R_vdw_element_Average',
       'log10_R_cov_element_Average', 'log10_zaratio_Average',
       'log10_density_Average', 'log10_ionenergy_zaratio_interaction']



def loss_fn(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=50)
    
def metric_fn(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=100) - tfp.stats.percentile(tf.abs(y_true - y_pred), q=0)

# Load the trained models
catboost_model = CatBoostRegressor()

catboost_model.load_model('./models/catboost_model.cbm')
tf_model = tf.keras.models.load_model('./models/tf_model.h5', custom_objects={
    'loss_fn': loss_fn,
    'metric_fn': metric_fn
})

def log10_bin(x):
    try:
        return np.log10(1+x)
    except:
        return None


# Define expected data types
expected_data_types = {
    'allelectrons_Total': float,
    'density_Total': float,
    'allelectrons_Average': float,
    'val_e_Average': float,
    'atomicweight_Average': float,
    'ionenergy_Average': float,
    'el_neg_chi_Average': float,
    'R_vdw_element_Average': float,
    'R_cov_element_Average': float,
    'zaratio_Average': float,
    'density_Average': float,
}
def preprocess_input(form_data, expected_data_types):
    # Convert form data to a DataFrame
    input_data = {}
    for key, value in form_data.items():
        if key in expected_data_types:
            if expected_data_types[key] == float:
                input_data[key] = [float(value)]
            elif expected_data_types[key] == str:
                input_data[key] = [value]
        else:
            raise ValueError(f"Unexpected feature: {key}")
        
    input_df = pd.DataFrame(input_data)

    input_df['ionenergy_zaratio_interaction'] = input_df['ionenergy_Average'] * input_df['zaratio_Average']

    features = input_df.columns.to_list()
    
    for feature in features:
        if feature in ('Formula', 'Crystal structure','Hardness (Mohs)', 'id'):
            continue
        input_df[f'log10_{feature}'] = input_df[feature].apply(log10_bin)
        input_df = input_df.drop([feature], axis=1)
        print(f'Log10-transformed column {feature}]')
    
        print('-'*15, "Done!", '-'*15)

    return input_df

def save_data(form_data, prediction_float):
    allelectrons_Total = form_data['allelectrons_Total']
    density_Total = form_data['density_Total']
    allelectrons_Average = form_data['allelectrons_Average']
    val_e_Average = form_data['val_e_Average']
    atomicweight_Average = form_data['atomicweight_Average']
    ionenergy_Average = form_data['ionenergy_Average']
    el_neg_chi_Average = form_data['el_neg_chi_Average']
    R_vdw_element_Average = form_data['R_vdw_element_Average']
    R_cov_element_Average = form_data['R_cov_element_Average']
    zaratio_Average = form_data['zaratio_Average']
    density_Average = form_data['density_Average']

    cursor = mysql.connection.cursor()

    cursor.execute(''' INSERT INTO results (allelectrons_Total, density_Total, allelectrons_Average, 
                             val_e_Average, atomicweight_Average, ionenergy_Average,
                             el_neg_chi_Average, R_vdw_element_Average, R_cov_element_Average,
                             zaratio_Average, density_Average, predicted_hardness) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ''', 
                            (allelectrons_Total, density_Total, allelectrons_Average, 
                             val_e_Average, atomicweight_Average, ionenergy_Average,
                             el_neg_chi_Average, R_vdw_element_Average, R_cov_element_Average,
                             zaratio_Average, density_Average, prediction_float))

    mysql.connection.commit()
    cursor.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    form_data = request.form
    input_df = preprocess_input(form_data, expected_data_types)

    # Generate CatBoost features
    catboost_features = catboost_model.predict(input_df).reshape(-1, 1)
    
    # Combine original input with CatBoost features
    combined_features = np.hstack((input_df, catboost_features))

    # Make prediction with TensorFlow model
    prediction = tf_model.predict(combined_features)

    print("Prediction: ",prediction[0][0])

    # # Convert prediction to Python float
    prediction_float = float(prediction[0][0])

    prediction_str = str(prediction_float)

    save_data(form_data, prediction_float)

    # return jsonify({'prediction': prediction_str})

    return render_template("results.html", prediction_str=prediction_str)

@app.route("/history")
def history():

    cursor = mysql.connection.cursor()

    cursor.execute(''' SELECT * FROM results ''')

    previous_preds = cursor.fetchall()

    mysql.connection.commit()
    cursor.close()

    print(previous_preds)

    return render_template("history.html", previous_preds=previous_preds)
        

if __name__ == "__main__":
    app.run(debug=True)