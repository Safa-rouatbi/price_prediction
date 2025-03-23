import pandas as pd 
from sklearn.model_selection import train_test_split

# Charger les données nettoyées depuis le fichier CSV
df_cleaned = pd.read_csv('cleaned_property_prices.csv')

# Séparation des variables indépendantes et de la variable cible
X = df_cleaned.drop('price', axis=1)  # Suppression de la colonne cible
y = df_cleaned['price']  # Colonne cible (prix)

# Séparer en train et test (80% train, 20% test par exemple)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher les dimensions des ensembles
print(f"Ensemble d'entraînement : {X_train.shape}")
print(f"Ensemble de test : {X_test.shape}")

# Sauvegarder les ensembles dans des fichiers CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)