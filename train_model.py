import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor


df = pd.read_csv('cleaned_data.csv')
print("Colonnes disponibles:", df.columns.tolist())

df['price'] = np.log1p(df['price'])

features_to_keep = ['room_count', 'bathroom_count', 'size', 
                   'category_encoded', 'city_encoded', 
                   'region_encoded', 'type_encoded']
X = df[features_to_keep]

print("\nDistribution de type_encoded:")
print(df['type_encoded'].value_counts())

print("\nPrix moyen par type:")
print(df.groupby('type_encoded')['price'].mean())

X = df.drop(['price', 'log_price'], axis=1) 
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.7, 
    'min_data_in_leaf': 20,
    'random_state': 42,
    'importance_type': 'gain'
}


model = LGBMRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")

feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nImportance des features:")
print(feature_imp)

type_imp = feature_imp[feature_imp['Feature'] == 'type_encoded']['Importance'].values
if len(type_imp) > 0:
    print(f"\nImportance de type_encoded: {type_imp[0]:.4f}")
    if type_imp[0] < 10:
        print("Attention: type_encoded a peu d'impact!")
else:
    print("\ntype_encoded non trouvé dans les features!")

joblib.dump(model, 'lightgbm_model.pkl')
print("\nModèle sauvegardé avec succès!")