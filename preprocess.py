import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv('property_prices.csv')

encoders = {
    'category': LabelEncoder(),
    'city': LabelEncoder(),
    'region': LabelEncoder(),
    'type': LabelEncoder() 
}

for col in encoders:
    df[col + '_encoded'] = encoders[col].fit_transform(df[col])
    with open(f'encoders/{col}_encoder.pkl', 'wb') as f:
        pickle.dump(encoders[col], f)

df_clean = df.drop(columns=[col for col in encoders])
df_clean.to_csv('cleaned_data.csv', index=False)