# Modèle de Prédiction de la Température

Ce projet présente un modèle prédictif simple, développé à partir de zéro, pour prévoir la température de demain en fonction du jour de la semaine. Le modèle utilise Python et s'exécute sur Google Colab, en utilisant un petit ensemble de données météorologiques. L'objectif principal de ce modèle est de fournir une compréhension de base de la construction et de l'évaluation d'un modèle de régression linéaire pour la prédiction de données numériques.

## Table des Matières
1. [Aperçu](#aperçu)
2. [Jeu de Données](#jeu-de-données)
3. [Flux de Travail du Modèle](#flux-de-travail-du-modèle)
4. [Prérequis](#prérequis)
5. [Comment Exécuter](#comment-exécuter)
6. [Résultats et Visualisation](#résultats-et-visualisation)
7. [Travaux Futurs](#travaux-futurs)

---

## Aperçu
Ce modèle est un modèle de régression linéaire simple qui prévoit la température du jour suivant en fonction du jour de la semaine. L'ensemble de données utilisé comprend des températures historiques associées aux jours de la semaine. En entraînant le modèle sur ces données, nous pouvons faire des prévisions pour les jours à venir.

### Technologies Utilisées :
- **Python** : Langage de programmation pour le prétraitement des données, la construction du modèle et la visualisation.
- **Google Colab** : Environnement basé sur le cloud pour exécuter le script Python.
- **scikit-learn** : Bibliothèque pour l'apprentissage automatique et le prétraitement des données.
- **Matplotlib** : Bibliothèque pour la visualisation des données.

---

## Jeu de Données
L'ensemble de données est stocké dans un fichier CSV nommé `weather_data.csv`. Il contient les colonnes suivantes :
- **Day** : Le jour de la semaine (par exemple, Monday, Tuesday).
- **Temperature** : La température enregistrée pour ce jour.

### Hypothèses :
- La colonne `Day` est convertie en valeurs numériques (par exemple, Monday = 1, Tuesday = 2, etc.) pour l'entraînement du modèle.

---

## Flux de Travail du Modèle
1. **Importer les Bibliothèques** : Importation des bibliothèques essentielles telles que pandas, matplotlib et scikit-learn.
2. **Charger le Jeu de Données** : Le jeu de données météorologiques est chargé dans un DataFrame pandas.
3. **Prétraitement des Données** :
   - La colonne `Day` est associée à des valeurs numériques.
   - Les caractéristiques (`X`) et les étiquettes (`y`) sont extraites.
4. **Division des Données** : L'ensemble de données est divisé en sous-ensembles d'entraînement (80%) et de test (20%) à l'aide de `train_test_split`.
5. **Entraînement du Modèle** :
   - Un modèle `LinearRegression` de scikit-learn est entraîné sur les données d'entraînement.
6. **Prédiction et Évaluation** :
   - Le modèle prévoit les températures pour l'ensemble de test.
   - L'erreur quadratique moyenne (MSE) est calculée pour évaluer la performance du modèle.
7. **Visualisation** :
   - Un graphique de dispersion des températures réelles et une ligne de tendance des températures prédites sont tracés.
8. **Prévision** :
   - La température du jour suivant est prévue à l'aide du modèle entraîné.

---

## Prérequis
- Python 3.x
- Google Colab
- Bibliothèques requises :
  - pandas
  - matplotlib
  - scikit-learn

---

## Comment Exécuter
1. **Télécharger le Jeu de Données** :
   - Assurez-vous que le fichier `weather_data.csv` est téléchargé dans votre environnement Google Colab.
2. **Exécuter le Script** :
   - Copiez le code Python fourni dans un notebook Colab.
   - Ajustez le chemin du fichier pour le jeu de données si nécessaire.
3. **Exécutez le Notebook** :
   - Exécutez chaque cellule séquentiellement pour entraîner le modèle et visualiser les résultats.

---

## Résultats et Visualisation
- **Erreur Quadratique Moyenne (MSE)** : Affiche la mesure d'erreur pour évaluer la précision du modèle.
- **Graphique de Dispersion et Ligne de Tendance** : Une visualisation des températures réelles et prédites sur la semaine.
- **Prédiction de la Température de Demain** : Affiche la température prévue pour le jour suivant.

---

## Travaux Futurs
- **Améliorer le Jeu de Données** : Intégrer des caractéristiques supplémentaires telles que l'humidité, la vitesse du vent et les données saisonnières.
- **Améliorer le Modèle** : Expérimenter avec des modèles d'apprentissage automatique avancés (par exemple, arbres de décision, forêts aléatoires).
- **Étendre les Prévisions** : Étendre le modèle pour prévoir plusieurs jours dans le futur.

---

Ce projet démontre les bases de la construction et de l'interprétation d'un modèle prédictif pour les données temporelles. Il constitue un point de départ précieux pour

