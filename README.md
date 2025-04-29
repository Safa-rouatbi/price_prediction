#  Plateforme de Prédiction des Prix Immobiliers

##  Description

Ce projet propose une plateforme interactive permettant d’estimer les prix de biens immobiliers. En s’appuyant sur des algorithmes de machine learning, cette solution fournit des prédictions précises à partir de critères saisis par l’utilisateur tels que :

- Le nombre de chambres
- La taille du bien
- La région
- La ville

---

##  Fonctionnalités

- **Interface utilisateur intuitive** : saisie rapide des informations et estimation instantanée.
- **Nettoyage des données** : suppression des doublons, gestion des valeurs manquantes et aberrantes.
- **Modèle de machine learning performant** : entraînement avec LightGBM pour une haute précision.
- **Visualisation des tendances** : graphiques générés pour mieux comprendre les relations entre les variables.

---

## 🛠️ Technologies Utilisées

- **Langage** : Python
- **Framework ML** : LightGBM

###  Bibliothèques principales :

- **Pandas** : manipulation et nettoyage des données
- **NumPy** : calculs numériques
- **Scikit-learn** : prétraitement et évaluation du modèle
- **Matplotlib & Seaborn** : visualisations
- **Joblib** : sauvegarde et chargement du modèle

---

##  Architecture du Projet

Voici les fichiers principaux et leurs rôles respectifs :

-data_cleaning.py :
--Nettoie les données brutes en gérant les valeurs manquantes et en supprimant les doublons.
--Exporte les données nettoyées dans cleaned_property_prices.csv.

-exploration.py :
--Génère des visualisations pour explorer les tendances des prix et leurs relations avec d'autres variables.

-trainmodel.py :
--Entraîne un modèle LightGBM sur les données nettoyées.
--Évalue les performances du modèle et sauvegarde le modèle final dans lightgbm_model.pkl.

-predict.py :

--Implémente un système de prédiction en temps réel à partir des entrées utilisateur.

-preprocess.py :

--Encode les variables catégoriques (région, type, catégorie) pour les rendre exploitables par les algorithmes.


---

##  Étapes pour exécuter le projet

1. Nettoyer les données avec `data_cleaning.py`
2. Explorer les tendances avec `exploration.py`
3. Encoder et entraîner le modèle avec `trainmodel.py`
4. Effectuer des prédictions avec `predict.py`
5. Lancer l'interface utilisateur avec `app.py`

---

##  Résultats

- **MSE** : 0.112  
- **R² Score** : 0.91  

Ces scores illustrent la capacité du modèle à prédire les prix de manière précise et fiable.

---

##  Améliorations Futures

- Ajouter des données temporelles pour analyser les tendances à long terme.
- Enrichir le dataset avec des facteurs économiques (inflation, taux d’intérêt…).
- Améliorer l’interface utilisateur avec des visualisations dynamiques et interactives.

---




