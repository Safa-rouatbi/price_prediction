import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('cleaned_property_prices.csv')

# Histogramme du prix
plt.figure(figsize=(8, 6))
df['price'].hist(bins=20)
plt.title('Histogramme des prix')
plt.xlabel('Prix')
plt.ylabel('Fréquence')
plt.show()


# Diagramme en boîte du prix par région
plt.figure(figsize=(8, 6))
sns.boxplot(x='Region', y='price', data=df)
plt.title('Prix par région')
plt.xlabel('Region')
plt.ylabel('Prix')
plt.xticks(rotation=90)
plt.show()

# Diagramme de dispersion du prix en fonction de la taille
plt.figure(figsize=(8, 6))
plt.scatter(df['size'], df['price'])
plt.title('Prix en fonction de la taille')
plt.xlabel('Taille')
plt.ylabel('Prix')
plt.show()

# Diagramme de dispersion du prix en fonction du nombre de chambres et de salles de bain
plt.figure(figsize=(8, 6))
plt.scatter(df['room_count'], df['price'], label='Nombre de chambres')
plt.scatter(df['bathroom_count'], df['price'], label='Nombre de salles de bain')
plt.title('Prix en fonction du nombre de pièces')
plt.xlabel('Nombre de pièces')
plt.ylabel('Prix')
plt.legend()
plt.show()

# Prix moyen par type de propriété et par région
prix_moyens = df.groupby(['Property_type', 'Region'])['price'].mean().unstack()
prix_moyens.plot(kind='bar', figsize=(12, 6))
plt.title('Prix moyen par type de propriété et par région')
plt.xlabel('Type de propriété')
plt.ylabel('Prix moyen')
plt.show()

# Avant la transformation
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['size'], kde=True, color='blue')
plt.title('Histogramme avant log-transform sur Size')

plt.subplot(1, 3, 2)
sns.histplot(df['room_count'], kde=True, color='green')
plt.title('Histogramme avant log-transform sur Room Count')

plt.subplot(1, 3, 3)
sns.histplot(df['bathroom_count'], kde=True, color='red')
plt.title('Histogramme avant log-transform sur Bathroom Count')

plt.tight_layout()
plt.show()