Plateforme de Prédiction des Prix Immobiliers
📖 Description
Ce projet propose une plateforme interactive pour estimer les prix des biens immobiliers.
En s'appuyant sur des algorithmes de machine learning, cette solution offre des prédictions précises à partir des critères fournis par l'utilisateur,
tels que le nombre de chambres, la taille, la région et la ville.

Fonctionnalités
Interface utilisateur intuitive : Permet aux utilisateurs de saisir facilement les informations nécessaires et d’obtenir une estimation en un clic.
Nettoyage des données : Gestion des doublons, des valeurs manquantes et des valeurs aberrantes pour garantir des prédictions fiables.
Modèle avancé : Utilisation de LightGBM pour maximiser la précision des estimations.
Visualisation des tendances : Génération de graphiques pour mieux comprendre les relations entre les différentes variables.

Technologies Utilisées
Langage : Python
Framework Machine Learning : LightGBM
Bibliothèques Principales :
Pandas : Manipulation et analyse de données.
NumPy : Calculs mathématiques.
Scikit-learn : Prétraitement des données et évaluation des modèles.
Matplotlib et Seaborn : Visualisations et graphiques.
Joblib : Sauvegarde et chargement du modèle entraîné.

Architecture du Projet
Voici les fichiers principaux et leurs rôles respectifs :
data_cleaning.py :
Nettoie les données brutes en gérant les valeurs manquantes et en supprimant les doublons.
Exporte les données nettoyées dans cleaned_property_prices.csv.

exploration.py :
Génère des visualisations pour explorer les tendances des prix et leurs relations avec d'autres variables.

trainmodel.py :
Entraîne un modèle LightGBM sur les données nettoyées.
Évalue les performances du modèle et sauvegarde le modèle final dans lightgbm_model.pkl.

predict.py :
Implémente un système de prédiction en temps réel à partir des entrées utilisateur.

preprocess.py :
Encode les variables catégoriques (région, type, catégorie) pour les rendre exploitables par les algorithmes.

Étapes à suivre :
Lancez data_cleaning.py pour préparer les données.
Analysez les données avec exploration.py.
Entraînez le modèle avec trainmodel.py.
Effectuez des prédictions en exécutant predict.py.
tester l'interface avec app.py

Résultats
Performances du Modèle :

MSE : 0.112
R² : 0.91

Ces résultats mettent en avant la précision du modèle sur l'ensemble de test.

Améliorations Futures
Intégrer des données temporelles pour capturer les tendances des prix à long terme.
Enrichir les données avec des facteurs économiques (taux d'intérêt, inflation).
Proposer une interface plus interactive avec des visualisations dynamiques.
