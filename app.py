import os
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'lightgbm_model.pkl')
model = joblib.load(model_path)

# Chargement des encodeurs
encoders = {}
encoder_files = {
    'property_type': 'category_encoder.pkl',
    'city': 'city_encoder.pkl',
    'region': 'region_encoder.pkl'
}

for col, filename in encoder_files.items():
    encoder_path = os.path.join(BASE_DIR, 'encoders', filename)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder {filename} not found at {encoder_path}")
    with open(encoder_path, 'rb') as f:
        encoders[col] = pickle.load(f)

# Route pour l'interface web
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',
                         property_types=encoders['property_type'].classes_,
                         cities=encoders['city'].classes_,
                         regions=encoders['region'].classes_)

# Route API pour les prédictions (identique à votre version)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gère à la fois les requêtes JSON et form-data
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form.to_dict()
        
        # Validation des données
        required_fields = ['room_count', 'bathroom_count', 'size', 
                         'property_type', 'city', 'region', 'type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        # Gestion du type (compatible avec votre version)
        if isinstance(data['type'], int) or data['type'].isdigit():
            type_val = int(data['type'])
            if type_val not in [0, 1]:
                return jsonify({'error': "Type must be 0 (À Louer) or 1 (À Vendre)"}), 400
        else:
            type_mapping = {
                "à vendre": 1, "a vendre": 1, "vendre": 1,
                "à louer": 0, "a louer": 0, "louer": 0
            }
            type_input = data['type'].lower().strip()
            type_val = type_mapping.get(type_input)
            if type_val is None:
                return jsonify({'error': f"Invalid type. Must be 1/0 or: {list(type_mapping.keys())}"}), 400

        # Préparation des données d'entrée
        input_data = [
            float(data['room_count']),
            float(data['bathroom_count']),
            float(data['size']),
            int(encoders['property_type'].transform([data['property_type']])[0]),
            int(encoders['city'].transform([data['city']])[0]),
            int(encoders['region'].transform([data['region']])[0]),
            type_val
        ]

        # Vérification de la cohérence avec le modèle
        if hasattr(model, 'feature_name_'):
            if len(input_data) != len(model.feature_name_):
                return jsonify({'error': f'Feature mismatch. Expected {len(model.feature_name_)} features'}), 400

        # Prédiction
        prediction = model.predict([input_data])
        predicted_price = float(np.expm1(prediction[0]))

        # Réponse adaptée au type de requête
        if request.is_json:
            return jsonify({'predicted_price': predicted_price})
        else:
            return render_template('result.html',
                                price=f"{predicted_price:,.2f} dt",
                                data=data)
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        else:
            return render_template('error.html', error=error_msg), 500

if __name__ == '__main__':
    app.run(debug=True)