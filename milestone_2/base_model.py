import os
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Récupérer la clé API depuis la variable d'environnement
api_key = os.getenv("WANDB_API_KEY")

# Authentifier Wandb avec la clé API récupérée
wandb.login(key=api_key)

# Initialisation de Wandb avec nom de projet
wandb.init(project="IFT6758.2024-A03", name="exp_logistic_regression")

# Chargement des données d'entraînement à partir du fichier train_data.csv
train_data = pd.read_csv('train_data.csv')

# Suppose que la colonne "distance" est la caractéristique et "result" est la cible
X = train_data[['distance']].values  # Caractéristiques (par exemple, 'distance')
y = train_data['result'].values  # Cible (par exemple, 'result')

# Split des données en train et validation (80% pour l'entraînement, 20% pour la validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation des hyperparamètres dans wandb.config
wandb.config.update({
    "test_size": 0.2,
    "random_state": 42,
    "solver": "lbfgs",  # Paramètre de régression logistique, défini par défaut à 'lbfgs'
    "max_iter": 100,    # Nombre maximal d'itérations
})

# Entraînement du modèle Logistic Regression
clf = LogisticRegression(solver=wandb.config.solver, max_iter=wandb.config.max_iter)
clf.fit(X_train, y_train)

# Prédictions sur l'ensemble de validation
y_pred = clf.predict(X_val)

# Calcul de la précision (accuracy)
accuracy = accuracy_score(y_val, y_pred)
print(f"Précision sur l'ensemble de validation : {accuracy:.4f}")

# Enregistrement de la précision dans Wandb
wandb.log({"validation_accuracy": accuracy})

# Créer un DataFrame pour visualiser les prédictions et les vraies valeurs dans Wandb
predictions_df = pd.DataFrame({
    "distance": X_val.flatten(),  # Caractéristique de validation (distance)
    "true_label": y_val,          # Valeurs réelles
    "predicted_label": y_pred     # Prédictions du modèle
})

# Enregistrer le tableau de données dans Wandb pour analyse
wandb.log({"predictions_table": wandb.Table(dataframe=predictions_df)})

# Visualisation des prédictions vs vérité terrain
plt.figure(figsize=(10, 6))
plt.scatter(X_val, y_val,marker='o', color='blue', label='Vérité terrain')  # Points réels
plt.scatter(X_val, y_pred,marker='x', color='red', label='Prédictions', alpha=0.5)  # Prédictions
plt.title('Prédictions vs Réalité pour Logistic Regression')
plt.xlabel('Distance')
plt.ylabel('Classe (0 ou 1)')
plt.legend()
plt.show()

# Enregistrement du graphique dans Wandb
wandb.log({"predictions_vs_true_plot": plt})

# Sauvegarde du modèle
model_filename = "logistic_regression_model.pkl"
joblib.dump(clf, model_filename)

# Utilisation de log_artifact pour enregistrer le modèle sur Wandb
artifact = wandb.Artifact("logistic_regression_model", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# Fin de la session Wandb
wandb.finish()
