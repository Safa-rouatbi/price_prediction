import pandas as pd
import numpy as np
import joblib


model = joblib.load('lightgbm_model.pkl')

new_data = {
    'room_count': [3.0],
    'bathroom_count': [1.0],
    'size': [120.0],
    'Property_type': [0], 
    'Type': [0],           
    'City': [0],           
    'Region': [0]          
}


df_new = pd.DataFrame(new_data)
df_new['size_per_room'] = df_new['size'] / df_new['room_count']
df_new['bathroom_ratio'] = df_new['bathroom_count'] / df_new['room_count']

predicted_log_price = model.predict(df_new)
predicted_price = np.expm1(predicted_log_price)
print(f"Prix prédit : {predicted_price[0]:.2f} ")

def validate_input(data: dict) -> bool:
    required_keys = ['room_count', 'bathroom_count', 'size', 'Property_type', 'Type', 'City', 'Region']
    return all(key in data for key in required_keys)

if not validate_input(new_data):
    print("Erreur : Données manquantes ou format incorrect.")
    exit()