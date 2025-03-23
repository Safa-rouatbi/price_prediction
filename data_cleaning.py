import pandas as pd 
from sklearn.preprocessing import LabelEncoder


#je charge le fichier csv
df=pd.read_csv('property_prices.csv')
#afficher les premières lignes pour verification

print(df.duplicated().sum())

# Remplacer les valeurs -1.0 par NaN dans les colonnes concernées
columns_to_replace=['room_count','bathroom_count','size']
df[columns_to_replace] = df[columns_to_replace].replace(-1.0, pd.NA)


# Imputer les valeurs manquantes avec la médiane de chaque colonne
df['room_count'].fillna(df['room_count'].median(), inplace=True)
df['bathroom_count'].fillna(df['bathroom_count'].median(), inplace=True)
df['size'].fillna(df['size'].median(), inplace=True)

# Vérifier après l'imputation
print(df[['room_count', 'bathroom_count', 'size']].head())
print(df[df.duplicated(keep=False)])
#afficher les doublons 
print(df.duplicated().sum())

df.drop('log_price', axis=1, inplace=True)
#juste pour verifier que si log_prices est supprimé 
print(df.head)
# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Appliquer le LabelEncoder sur les colonnes catégorielles
df['Property_type'] = label_encoder.fit_transform(df['category'])
df['Type'] = label_encoder.fit_transform(df['type'])
df['City'] = label_encoder.fit_transform(df['city'])
df['Region'] = label_encoder.fit_transform(df['region'])

# Supprimer les colonnes textuelles si elles sont inutiles
df = df.drop(columns=['category','type','city','region']) 

# Afficher le DataFrame après Label Encoding
print(df)

# Calculer l'IQR pour chaque colonne numérique
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Filtrer les données pour enlever les valeurs aberrantes (en dehors de 1.5 * IQR)
df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]



# Affichage du DataFrame nettoyé
print(df_cleaned.head())
df_cleaned.to_csv('cleaned_property_prices.csv', index=False)