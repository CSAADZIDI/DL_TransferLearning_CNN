# 🩺 Détection de Pneumonie sur Radiographies Thoraciques utilisant le Transfer Learning (à partir de modèles pré-entraînés de type CNN) et MLflow

## 🧠 Présentation du Projet
Ce projet vise à développer un modèle d’apprentissage profond capable de classifier automatiquement des radiographies thoraciques en deux catégories : Pneumonie et Normal. Il s’appuie sur des techniques de transfer learning avec des réseaux de neurones convolutifs pré-entraînés pour améliorer les performances et accélérer l'entraînement. Le suivi des expériences est assuré par MLflow, ce qui facilite la reproductibilité et la gestion des versions du modèle.

## 🔍 Contexte
La pneumonie est une infection grave des poumons qui peut mettre la vie en danger. Un diagnostic précoce à partir de radiographies peut grandement améliorer le pronostic. Cependant, l’analyse manuelle des images est lente et sujette à l’erreur humaine. Ce projet propose une solution automatisée pour aider les professionnels de santé à diagnostiquer plus rapidement et avec plus de précision.
Les performances obtenues ont été atteintes grâce au transfert d’apprentissage, apprentissage en ensemble et au fine-tuning des couches supérieures.

## 🧰 Technologies utilisées

Langage de programmation : Python

Framework Deep Learning : TensorFlow 

Modèles pré-entraînés : VGG16, DenseNet121, etc.

Suivi des expériences : MLflow

Données : Chest X-Ray Images (Pneumonia) - Kaggle
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

## 📁 Structure du Projet

├── data/                   # Données (train, val, test) (ignored)

├── figures/                 # Matrice de confusion, courbes, métriques (ignored)

├── notebooks/              # Notebooks Jupyter pour l'exploration et le prototypage

├── models/                 # Modèles sauvegardés


├── mlruns/                 # Répertoire utilisé par MLflow (ignored)


├── requirements.txt        # Dépendances du projet


└── README.md               # Présentation du projet

## 🧪 Suivi des expériences avec MLflow
MLflow permet de :

Suivre les métriques d'entraînement et de validation

Enregistrer les hyperparamètres et artefacts

Comparer les différents essais

Gérer les versions des modèles

Pour lancer l’interface graphique de MLflow :

Changer de dossier (cd notebooks/)

Ecrire dans le terminal:

mlflow ui


Puis ouvrez http://localhost:5000 dans votre navigateur.

## 🚀 Lancement du projet

1. Cloner le dépôt :

git clone https://github.com/CSAADZIDI/DL_TransferLearning_CNN.git

2. Changer de répertoire

cd DL_TransferLearning_CNN

3. Noublier pas de crèer l'environnment virtuel

3. Installer les dépendances :

pip install -r requirements.txt

3. Tester le modèle avec les données à télécharger de Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

4. Tester le projet via l'interface fournie en utilisant streamlit

    Changer de dossier (cd web_app)

    Ecrire streamlit run app.py

    Puis ouvrez http://localhost:8502 dans votre navigateur. (Généralement, votre navigateur s'ouvre automatiquement)


## 🧪 Test du projet avec Streamlit

Une fois la page http://localhost:8502 est chargée, choisir une image de test.

Une réponse sera affichée s'il s'agit d'une pneumonie ou non.

Pour l'explinabilité du modèle, nous fournissons à l'expert une image montrant les zones les plus analysées par le modèle.


## 📌 Pistes d’amélioration


Combinaison et personalisation de plusieurs modèles 

Extension à la classification multi-classes (COVID-19, TB, etc.)

## 🤝 Contributions
Les contributions sont les bienvenues ! N’hésitez pas à ouvrir une issue ou à soumettre une pull request.
