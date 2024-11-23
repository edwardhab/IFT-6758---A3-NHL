import os
import joblib
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Récupérer la clé API depuis la variable d'environnement
api_key = os.getenv("WANDB_API_KEY")

# Authentifier Wandb avec la clé API récupérée
wandb.login(key=api_key)

# Initialisation de Wandb avec nom de projet
wandb.init(project="IFT6758.2024-A03", name="exp_logistic_regression_distance_angle", entity="michel-wilfred-essono-university-of-montreal")

# Chargement des données
df = pd.read_csv('train_data.csv')
X_df = df[['shotDistance', 'shotAngle']]
y_df = df['result']

# Conversion en numpy array
X_data = X_df.to_numpy()
y_data = y_df.to_numpy()

# Division des données en ensembles d'entraînement et de validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_data, y_data, test_size=0.25, random_state=10, shuffle=True
)

# Fonction pour entraîner un modèle de régression logistique
def logreg_fit(X_train, X_valid, y_train, y_valid):
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_valid.ndim == 1:
        X_valid = X_valid.reshape(-1, 1)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    predicted_prob = clf.predict_proba(X_valid)[:, 1]
    return clf, y_pred, predicted_prob

# Entraînement des modèles
logreg_dist, y_pred_dist, prob_dist = logreg_fit(X_train[:, 0], X_valid[:, 0], y_train, y_valid)
logreg_ang, y_pred_ang, prob_ang = logreg_fit(X_train[:, 1], X_valid[:, 1], y_train, y_valid)
logreg_comb, y_pred_comb, prob_comb = logreg_fit(X_train, X_valid, y_train, y_valid)

# Sauvegarde et enregistrement des modèles dans des artifacts séparés
model_names = ["logreg_dist", "logreg_ang", "logreg_comb"]
models = [logreg_dist, logreg_ang, logreg_comb]

for model_name, model in zip(model_names, models):
    # Sauvegarder localement
    model_path = f"models/{model_name}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    
    # Enregistrer dans un artifact wandb
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

# Baseline aléatoire
prob_rand = np.random.uniform(0, 1, prob_comb.shape)
y_pred_rand = np.where(prob_rand > 0.5, 1, 0)

# Modèles et probabilités
model_list = ["Distance", "Angle", "Distance + Angle", "Random"]
models_prob = [prob_dist, prob_ang, prob_comb, prob_rand]
models_pred = [y_pred_dist, y_pred_ang, y_pred_comb, y_pred_rand]

# Calcul des courbes ROC et AUC
fpr_list, tpr_list, roc_auc_list = [], [], []
for prob, model in zip(models_prob, model_list):
    fpr, tpr, _ = roc_curve(y_valid, prob)
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

# Tracé des courbes ROC
plt.figure(figsize=(10, 6))
for fpr, tpr, roc_auc, model in zip(fpr_list, tpr_list, roc_auc_list, model_list):
    plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Baseline")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbes ROC")
plt.legend(loc="lower right")

# Sauvegarde de la courbe ROC en local
roc_plot_path = "figures/roc_curve.png"
plt.savefig(roc_plot_path)

# Log de la figure dans wandb
wandb.log({"ROC Curves": wandb.Image(roc_plot_path)})
plt.close()

# Définir les intervalles de probabilités
bins = list(np.arange(0, 105, 5))
bin_centers = list(np.arange(2.5, 100, 5.0))

# Préparation des dataframes pour les plots b, c, d
df_prob_list = []
df_prob_bined_list = []

for i in range(len(model_list)):
    # Créer un DataFrame avec les prédictions et les probabilités
    df_prob = pd.DataFrame(list(zip(models_pred[i], y_valid, models_prob[i]*100)), columns=['goal_pred', 'goal', 'goal_Prob'])
    df_prob['shot'] = 1  # Chaque ligne représente un tir
    sum_goal = df_prob['goal'].sum()  # Somme des buts
    df_prob['percentile'] = df_prob['goal_Prob'].rank(pct=True) * 100  # Rank des probabilités en percentile
    df_prob['goal_perc_bins'] = pd.cut(df_prob['percentile'], bins, labels=bin_centers)  # Binning des percentiles

    # Groupement des données par bins de probabilité
    df_prob_bined = df_prob[['goal_perc_bins', 'shot', 'goal']].groupby(['goal_perc_bins']).sum().reset_index()
    df_prob_bined['goal_rate'] = (df_prob_bined['goal'] / df_prob_bined['shot'])  # Calcul du taux de buts
    df_prob_bined['goal_cum'] = (df_prob_bined['goal'] / sum_goal)  # Calcul du cumul de buts
    df_prob_bined['goal_cumsum'] = 1 - df_prob_bined['goal_cum'].cumsum()  # Cumul inverse des buts

    df_prob_list.append(df_prob)
    df_prob_bined_list.append(df_prob_bined)

# Tracé du graphique du taux de buts
goal_rate_plot = plt.figure(figsize=(10, 5))
plt.title("Goal Rate")

# Tracer pour chaque modèle
for i in range(len(model_list)):
    ax = sns.lineplot(
        x='goal_perc_bins',
        y='goal_rate',
        data=df_prob_bined_list[i],
        legend=False,
        linewidth=2.5,
        label=f"{model_list[i]}"
    )

plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goals / (Shots + Goals)')
ax.set_xlim(left=101, right=-1)  # Limites inversées pour l'axe X
ax.set_ylim(bottom=0, top=1)  # Limites de l'axe Y
goal_rate_plot.legend(bbox_to_anchor=(0.9, 0.88))
plt.xticks(np.arange(0, 120, 20))  # Ajustement des ticks de l'axe X

# Sauvegarde du graphique du taux de buts en local
goal_rate_plot_path = "figures/goal_rate_plot.png"
goal_rate_plot.savefig(goal_rate_plot_path)

# Log de la figure dans wandb
wandb.log({"Goal Rate Plot": wandb.Image(goal_rate_plot_path)})

# Calcul du taux cumulé des buts
cum_rate_plot = plt.figure(figsize=(10, 5))
plt.title("Cumulative % of Goal")
for i in range(len(model_list)):
    ax = sns.lineplot(
        x='goal_perc_bins',
        y='goal_cumsum',
        data=df_prob_bined_list[i],
        legend=False,
        label=f"{model_list[i]}",
        linewidth=2.5
    )

plt.legend(loc="lower right")
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Proportion')
ax.set_xlim(left=101, right=-1)  # Limites inversées pour l'axe X
ax.set_ylim(bottom=0, top=1)  # Limites de l'axe Y
plt.xticks(np.arange(0, 120, 20))  # Ajustement des ticks de l'axe X

# Sauvegarde du graphique du taux cumulé des buts en local
cum_rate_plot_path = "figures/cum_rate_plot.png"
cum_rate_plot.savefig(cum_rate_plot_path)

# Log de la figure dans wandb
wandb.log({"Cumulative Goal Rate Plot": wandb.Image(cum_rate_plot_path)})

# Calcul et tracé de la courbe de calibration
calibration_plot = plt.figure(figsize=(7, 7))
plt.title("Calibration Curve")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for i in range(len(model_list)):
    prob_true, prob_pred = calibration_curve(df_prob_list[i]['goal'], df_prob_list[i]['goal_Prob']/100, n_bins=20)
    plt.plot(prob_pred, prob_true, "s-", label=f"{model_list[i]}")

plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Proportion')
plt.legend(loc="upper left", ncol=2)
plt.xticks(np.arange(0, 1.2, 0.2))
plt.yticks(np.arange(0, 1.2, 0.2))

# Sauvegarder localement la courbe de calibration
calibration_plot_path = "figures/calibration_curve.png"
calibration_plot.savefig(calibration_plot_path)

# Log de la courbe de calibration dans wandb
wandb.log({"Calibration Curve": wandb.Image(calibration_plot_path)})

# Affichage de la courbe de calibration
plt.show()
