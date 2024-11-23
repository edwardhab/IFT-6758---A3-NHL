import os
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, brier_score_loss
from sklearn.calibration import CalibrationDisplay
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

# Suppose que la colonne "distance" est la caractéristique et "target" est la cible
X = train_data[['distance']].values  # Caractéristiques (par exemple, 'distance')
y = train_data['result'].values  # Cible (par exemple, 'target')

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
y_pred_proba = clf.predict_proba(X_val)[:, 1]  # Probabilité pour la classe 1 (but)

# Calcul de la précision (accuracy)
accuracy = accuracy_score(y_val, y_pred)
print(f"Précision sur l'ensemble de validation : {accuracy:.4f}")

# Enregistrement de la précision dans Wandb
wandb.log({"validation_accuracy": accuracy})

# --- 1. Courbe ROC et AUC --- 
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Création du graphique avec matplotlib
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', label="Classificateur aléatoire (AUC = 0.5)")
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')

# Sauvegarder le graphique en tant que fichier image
roc_curve_path = "roc_curve.png"
plt.savefig(roc_curve_path)

# Loguer l'image dans Wandb
wandb.log({"roc_curve": wandb.Image(roc_curve_path)})

# --- 2. Taux de buts par centile de la probabilité ---
percentiles = np.percentile(y_pred_proba, np.arange(0, 101, 10))
goal_rate_by_percentile = [np.mean(y_val[y_pred_proba >= p]) for p in percentiles]

# Création du graphique avec matplotlib
plt.figure(figsize=(10, 6))
plt.plot(percentiles, goal_rate_by_percentile, marker='o', color='blue')
plt.title("Taux de buts par centile de la probabilité")
plt.xlabel("Centile de la probabilité")
plt.ylabel("Taux de buts")

# Sauvegarder le graphique en tant que fichier image
goal_rate_path = "goal_rate_by_percentile.png"
plt.savefig(goal_rate_path)

# Loguer l'image dans Wandb
wandb.log({"goal_rate_by_percentile": wandb.Image(goal_rate_path)})

# --- 3. Proportion cumulée de buts ---
cumulative_goal_rate = [np.sum(y_val <= p) / len(y_val) for p in percentiles]

# Création du graphique avec matplotlib
plt.figure(figsize=(10, 6))
plt.plot(percentiles, cumulative_goal_rate, marker='o', color='green')
plt.title("Proportion cumulée de buts par centile de la probabilité")
plt.xlabel("Centile de la probabilité")
plt.ylabel("Proportion cumulée de buts")

# Sauvegarder le graphique en tant que fichier image
cumulative_goal_rate_path = "cumulative_goal_rate.png"
plt.savefig(cumulative_goal_rate_path)

# Loguer l'image dans Wandb
wandb.log({"cumulative_goal_rate": wandb.Image(cumulative_goal_rate_path)})

# --- 4. Diagramme de fiabilité (calibration curve) ---
calibration_display = CalibrationDisplay.from_estimator(clf, X_val, y_val)
wandb.log({"calibration_curve": calibration_display.figure_})

# Sauvegarde du modèle
model_filename = "logistic_regression_model.pkl"
joblib.dump(clf, model_filename)

# Utilisation de log_artifact pour enregistrer le modèle sur Wandb
artifact = wandb.Artifact("logistic_regression_model", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# Fin de la session Wandb
wandb.finish()
