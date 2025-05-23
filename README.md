# 🩺 Détection de Pneumonie sur Radiographies Thoraciques avec Transfer Learning et MLflow
## 🧠 Présentation du Projet
Ce projet vise à développer un modèle d’apprentissage profond capable de classifier automatiquement des radiographies thoraciques en deux catégories : Pneumonie et Normal. Il s’appuie sur des techniques de transfer learning avec des réseaux de neurones convolutifs pré-entraînés pour améliorer les performances et accélérer l'entraînement. Le suivi des expériences est assuré par MLflow, ce qui facilite la reproductibilité et la gestion des versions du modèle.

## 🔍 Contexte
La pneumonie est une infection grave des poumons qui peut mettre la vie en danger. Un diagnostic précoce à partir de radiographies peut grandement améliorer le pronostic. Cependant, l’analyse manuelle des images est lente et sujette à l’erreur humaine. Ce projet propose une solution automatisée pour aider les professionnels de santé à diagnostiquer plus rapidement et avec plus de précision.

## 🧰 Technologies utilisées

Langage de programmation : Python

Framework Deep Learning : TensorFlow ou PyTorch

Modèles pré-entraînés : ResNet50, DenseNet121, etc.

Suivi des expériences : MLflow

Données : Chest X-Ray Images (Pneumonia) - Kaggle
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

## 📁 Structure du Projet

<<<<<<< HEAD
=======

>>>>>>> 14a47a5 (updating README)
├── data/                   # Données (train, val, test)

├── notebooks/              # Notebooks Jupyter pour l'exploration et le prototypage

├── models/                 # Modèles sauvegardés


├── mlruns/                 # Répertoire utilisé par MLflow


├── requirements.txt        # Dépendances du projet


└── README.md               # Présentation du projet

## 🧪 Suivi des expériences avec MLflow
MLflow permet de :

Suivre les métriques d'entraînement et de validation

Enregistrer les hyperparamètres et artefacts

Comparer les différents essais

Gérer les versions des modèles

Pour lancer l’interface graphique de MLflow :

Taper

mlflow ui


Puis ouvrez http://localhost:5000 dans votre navigateur.

## 🚀 Lancement du projet
1. Cloner le dépôt :

git clone https://github.com/CSAADZIDI/DL_TransferLearning_CNN.git

2. Installer les dépendances :

pip install -r requirements.txt

3. Entraîner et Évaluer le modèle :



Les performances obtenues ont été atteintes grâce au transfert d’apprentissage et au fine-tuning des couches supérieures.

## 📌 Pistes d’amélioration

Déploiement d’une API REST (FastAPI ou Flask)


Combinaison de plusieurs modèles (ensemble learning)

Extension à la classification multi-classes (COVID-19, TB, etc.)

## 🤝 Contributions
Les contributions sont les bienvenues ! N’hésitez pas à ouvrir une issue ou à soumettre une pull request.
