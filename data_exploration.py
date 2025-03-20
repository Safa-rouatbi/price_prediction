import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('property_prices.csv')
print(data.dtypes)

# Sélectionner + afficher+calculer la corrélation
numeric_data = data.select_dtypes(exclude='object') 
print(numeric_data.head())
correlation_matrix = numeric_data.corr()

# Créer une heatmap des corrélations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap des corrélations')
plt.show()

# Visualiser la distribution de 'price'
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, bins=30, color='blue')
plt.title('Distribution du Prix')
plt.xlabel('Prix')
plt.ylabel('Fréquence')
plt.show()

# Visualiser la distribution de 'size'
plt.figure(figsize=(10, 6))
sns.histplot(data['size'], kde=True, bins=30, color='green')
plt.title('Distribution de la Taille (Size)')
plt.xlabel('Taille')
plt.ylabel('Fréquence')
plt.show()

# Scatter plot pour la relation entre le prix et la taille
plt.figure(figsize=(10, 6))
sns.scatterplot(x='size', y='price', data=data, color='purple')
plt.title('Relation entre Prix et Taille (Size)')
plt.xlabel('Taille (Size)')
plt.ylabel('Prix')
plt.show()

# Scatter plot pour la relation entre le prix et le nombre de chambres
plt.figure(figsize=(10, 6))
sns.scatterplot(x='room_count', y='price', data=data, color='red')
plt.title('Relation entre Prix et Nombre de Chambres')
plt.xlabel('Nombre de Chambres')
plt.ylabel('Prix')
plt.show()

# Boxplot pour voir la distribution des prix selon la catégorie
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='price', data=data)
plt.title('Distribution des Prix par Catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Prix')
plt.xticks(rotation=45) 
plt.show()

# Boxplot pour la distribution de la taille (size) par catégorie
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='size', data=data)
plt.title('Distribution de la Taille par Catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Taille')
plt.xticks(rotation=45)
plt.show()

# Boxplot pour la distribution du nombre de chambres (room_count) par catégorie
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='room_count', data=data)
plt.title('Distribution du Nombre de Chambres par Catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Nombre de Chambres')
plt.xticks(rotation=45)
plt.show()


# Histogramme pour le nombre de chambres
plt.figure(figsize=(10, 6))
sns.histplot(data['room_count'], kde=True, bins=20, color='orange')
plt.title('Distribution du Nombre de Chambres')
plt.xlabel('Nombre de Chambres')
plt.ylabel('Fréquence')
plt.show()






