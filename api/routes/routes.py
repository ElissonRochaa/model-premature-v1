from flask import jsonify, request
from api import app
import pandas as pd
from api.models.prediction_model import load_model, predict_test
from datetime import datetime

#custom_model = load_model()

#endpoint de teste
@app.route('/api/')
def index():
    return "Hello, World! this is premature birth model!", 200

# #lista de parametros do modelo
# parameters = ['gestational_risk', 'schooling', 'has_hypertension', 'has_diabetes', 'has_pelvic_surgery','has_urinary_infection', 'has_congenital_malformation', 
#     'has_family_twinship', 'amount_gestation', 'amount_abortion', 'amount_deliveries','amount_cesarean', 'mothers_birth_date', 'date_start_pregnancy', 'date_first_prenatal', 'date_last_delivery']

#endpoint de predição
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        custom_model = load_model()
        preprocessed_data = custom_model.preprocess(data)
        
        X_test = pd.DataFrame(preprocessed_data, index=[0])
        # Prepare os dados de entrada para o modelo XGBoost
        predicao = predict_test(X_test, custom_model)
        return jsonify(predicao), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
      

#endpoint que lista os parâmetros do modelo para o getway
@app.route('/api/parameters', methods=['GET'])
def model_parameters():
    try:
        print("1")
        custom_model = load_model()
        print(custom_model)
        parameters = list(custom_model.attributes_info.keys())
        return parameters, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
