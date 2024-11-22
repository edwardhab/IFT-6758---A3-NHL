import os
import wandb
import pandas as pd
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,classification_report,log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

def temporal_split(data, n_first_games=20):
    data_sorted = data.sort_values(['gameId', 'timeInPeriod'])
    teams = data['teamId'].unique()
    mask = pd.Series(True, index=data.index)
    
    for team in teams:
        team_games = data[data['teamId'] == team].sort_values('gameId')
        first_games = team_games.head(n_first_games).index
        mask[first_games] = False
    
    return data[mask]

class PlayerTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.player_stats = {}
        self.global_mean = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            # Gestion des colonnes pour les tireurs et les gardiens
            for column in X.columns:
                self.player_stats[column] = {}
                
                player_counts = X[column].value_counts()
                # Calcul de la moyenne globale pour ce type de joueur
                self.global_mean = np.mean(y)
                
                for player in player_counts.index:
                    mask = (X[column] == player)
                    shots = np.sum(mask)
                    goals = np.sum(y[mask])
                    
                    # Taux de réussite lissé
                    smoothed_rate = (goals + self.smoothing * self.global_mean) / (shots + self.smoothing)
                    self.player_stats[column][player] = smoothed_rate
        
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = np.zeros((len(X), len(X.columns)))
            
            for idx, column in enumerate(X.columns):
                result[:, idx] = X[column].map(self.player_stats[column]).fillna(self.global_mean)
            
            return result
        
        return X.map(self.player_stats).fillna(self.global_mean).to_numpy().reshape(-1, 1)

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = None

    def fit(self, X, y):
        # S'assurer que X est une série pandas
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Calcul de la moyenne globale
        self.global_mean = np.mean(y)
        
        # Calcul des moyennes pour chaque catégorie avec lissage
        counts = X.value_counts()
        for category in counts.index:
            cat_mask = (X == category)
            n = counts[category]
            cat_mean = np.mean(y[cat_mask])
            # Application du lissage
            smoothed_mean = (n * cat_mean + self.smoothing * self.global_mean) / (n + self.smoothing)
            self.target_means[category] = smoothed_mean
            
        return self

    def transform(self, X):
        # S'assurer que X est une série pandas
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.map(self.target_means).fillna(self.global_mean).to_numpy().reshape(-1, 1)

# Conversion du temps en période en secondes
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return None

def calculate_score_differential(row):
    if row['HomevsAway'] == 'home':
        return row['home_goals'] - row['away_goals']
    else:
        return row['away_goals'] - row['home_goals']

def custom_scorer(y_true, y_pred_proba, beta=1.0):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # F-beta score pour donner plus de poids à la précision ou au rappel
        if precision + recall > 0:
            score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            best_score = max(best_score, score)
            
    return best_score

#custom_scorer = make_scorer(custom_scorer, needs_proba=True)
# Initialisation de Wandb
wandb.init(project="IFT6758.2024-A03", name="exp_random_forest_optimized",entity="michel-wilfred-essono-university-of-montreal")

# Chargement des données
train_data = pd.read_csv('train_data.csv')
train_data=temporal_split(train_data)

# Étape 1 : Trier les événements chronologiquement
train_data['powerplay'] = ((train_data['HomevsAway'] == 1) & (train_data['powerplayHome'] == 1) | 
                          (train_data['HomevsAway'] == 0) & (train_data['powerplayAway'] == 1)).astype(int) - \
                         ((train_data['HomevsAway'] == 1) & (train_data['powerplayAway'] == 1) | 
                          (train_data['HomevsAway'] == 0) & (train_data['powerplayHome'] == 1)).astype(int)

# Ajouter le cumul des tirs par période
train_data['teamId'] = np.where(
        train_data['HomevsAway'] == 'home',
        train_data['homeTeamId'],
        train_data['awayTeamId']
    )

# Remplir les valeurs manquantes avec la médiane par groupe
for col in ['speed', 'distanceFromLastEvent', 'lastEventXCoord', 'lastEventYCoord']:
    median_by_group = train_data.groupby('lastEventType')[col].transform('median')
    train_data[col] = train_data[col].fillna(median_by_group).fillna(train_data[col].median())
# Calcul du cumul des tirs par période
train_data['shots_period_cumsum'] = train_data.groupby(['gameId', 'period', 'teamId'])['eventType'].cumcount()
train_data['timeInPeriod'] = train_data['timeInPeriod'].apply(convert_time_to_seconds)
train_data = train_data.sort_values(['gameId', 'period', 'timeInPeriod'])

# Étape 2 : Calcul de la différence de temps entre les événements consécutifs
train_data['time_diff'] = train_data.groupby(['gameId', 'period'])['timeInPeriod'].diff().fillna(0)

# Étape 3 : Attribution du temps à l'équipe contrôlant la rondelle
train_data['team_possession_time'] = train_data['time_diff']

# Étape 4 : Somme cumulative du temps de possession pour chaque équipe
train_data['cumulative_possession_time'] = (
    train_data.groupby(['gameId', 'teamId'])['team_possession_time']
    .cumsum()
)

# Ajout du temps depuis le dernier tir
train_data['time_since_last_shot'] = train_data.groupby(['gameId', 'period', 'teamId'])['timeInPeriod'].diff().fillna(0)

# Calcul des pénalités consécutives
train_data['consecutive_penalties'] = (
        train_data.groupby(['gameId', 'teamId'])
        ['powerplay']
        .cumsum()
    )

# Calcul des buts à domicile et à l'extérieur
train_data['home_goals'] = ((train_data['HomevsAway'] == 'home') & (train_data['eventType'] == 'goal')).groupby(train_data['gameId']).cumsum()
train_data['away_goals'] = ((train_data['HomevsAway'] == 'away') & (train_data['eventType'] == 'goal')).groupby(train_data['gameId']).cumsum()
train_data['score_differential'] = train_data.apply(calculate_score_differential, axis=1)

# Calcul de la fréquence pour chaque catégorie
frequency = train_data['lastEventType'].value_counts(normalize=True)
train_data['lastEventType'] = train_data['lastEventType'].map(frequency)
frequency = train_data['shotType'].value_counts(normalize=True)
train_data['shotType'] = train_data['shotType'].map(frequency)

# Mise à jour des buts avec décalage
train_data['home_goals'] = ((train_data['HomevsAway'] == 'home') & (train_data['eventType'] == 'goal')).groupby(train_data['gameId']).cumsum().shift(1, fill_value=0)
train_data['away_goals'] = ((train_data['HomevsAway'] == 'away') & (train_data['eventType'] == 'goal')).groupby(train_data['gameId']).cumsum().shift(1, fill_value=0)
train_data['score_differential'] = train_data.apply(calculate_score_differential, axis=1)

# Définition des groupes de caractéristiques
numerical_features = [
    'shotDistance', 'shotAngle', 'timeInPeriod', 'xCoord', 'yCoord','cumulative_possession_time',
    'timeElapsedSinceLastEvent', 'distanceFromLastEvent',
    'speed', 'powerplay', 'shots_period_cumsum', 'time_since_last_shot', 
    'consecutive_penalties','home_goals','away_goals',
    'score_differential','lastEventXCoord','lastEventYCoord','shotType','lastEventType'
]

categorical_features = [
    'offensiveSide', 'HomevsAway'
]

# Suppression de la colonne eventType si elle existe
if 'eventType' in train_data.columns and 'goal' in train_data['eventType'].unique():
    train_data = train_data.drop(columns=['eventType'])

binary_features = [
    'emptyNetGoal', 'rebound'
]
shooter_features = ['shooter']

# Conversion des caractéristiques binaires en entiers
for col in binary_features:
    train_data[col] = train_data[col].astype(int)

# Suppression des lignes avec des valeurs manquantes
train_data = train_data.dropna(subset=numerical_features + categorical_features + binary_features +shooter_features)

# Préparation des caractéristiques et de la cible
X = train_data[numerical_features + categorical_features + binary_features +shooter_features]
y = train_data['result'].astype(int)

# Division des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Application du sous-échantillonnage à l'ensemble d'entraînement
sampling_strategy = 0.35   # Ajuster ce ratio selon les besoins
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

# Création du pipeline de prétraitement
numeric_transformer =  'passthrough'
frequency_transformer = Pipeline(steps=[
    ('scaler', TargetEncoder(smoothing=10))
])

# Correction du transformateur catégoriel en ajoutant 'handle_unknown'
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('shooter', PlayerTargetEncoder(smoothing=30), shooter_features),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

# Ajout des poids de classe pour équilibrer l'importance des cas positifs (buts)
class_weights = {0: 1, 1: 2}  # Ajuster ces valeurs pour donner plus de poids aux buts
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight=class_weights
    ))
])

# Définition de la grille de paramètres
param_grid = {
    # Paramètres principaux
    'classifier__n_estimators': [100, 200, 300, 500],
    
    # Paramètres de régularisation
    'classifier__max_depth': [10, 15, 20, 25, None],  # Limite la profondeur de l'arbre
    'classifier__min_samples_split': [5, 10, 15, 20],  # Échantillons minimum avant division
    'classifier__min_samples_leaf': [4, 6, 8, 10],    # Échantillons minimum dans les feuilles
    'classifier__max_features': [0.5, 0.7, 'sqrt', 'log2'],  # Limite les caractéristiques pour les divisions
    'classifier__max_leaf_nodes': [50, 100, 200, None],  # Limite le nombre total de feuilles
    'classifier__min_impurity_decrease': [0.0001, 0.0005, 0.001],  # Amélioration minimum nécessaire pour la division
    
    'classifier__bootstrap': [True],
    'classifier__oob_score': [True],  # Estimation du score out-of-bag
}

# Configuration de la validation croisée
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Exécution de la recherche aléatoire
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=cv_splitter,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("Début de l'ajustement du modèle...")
# Ajustement du modèle
random_search.fit(X_train, y_train)

# Récupération des meilleurs paramètres
best_params = random_search.best_params_

# Formatage des paramètres pour Wandb
wandb_params = {
    'max_depth': best_params['classifier__max_depth'],
    'n_estimators': best_params['classifier__n_estimators'],
    'min_samples_split': best_params['classifier__min_samples_split'],
    'min_samples_leaf': best_params['classifier__min_samples_leaf'],
    'max_features': best_params['classifier__max_features'],
    'max_leaf_nodes':  best_params['classifier__max_leaf_nodes'],  # Définition d'une valeur initiale ou par défaut
    'min_impurity_decrease': best_params['classifier__min_impurity_decrease'],  # Définition d'une valeur par défaut ou choisie
    'bootstrap': True,  # Définition comme True puisque c'est fixé dans param_grid
    'oob_score': True  # Définition comme True pour l'estimation hors-sac
}

# Enregistrement des paramètres individuellement
for param_name, param_value in wandb_params.items():
    wandb.log({f"best_parameter/{param_name}": param_value})

# Enregistrement également comme dictionnaire complet
wandb.log({"best_parameters": wandb_params})

# Enregistrement de la configuration complète
wandb.config.update({
    "model_type": "RandomForest",
    "best_parameters": wandb_params,
    "sampling_strategy": sampling_strategy,
    "class_weights": class_weights
})

# Vous pouvez également créer une table Wandb pour une meilleure visualisation
wandb_table = wandb.Table(
    columns=["Parameter", "Value"],
    data=[[k, str(v)] for k, v in wandb_params.items()]
)
wandb.log({"best_parameters_table": wandb_table})

# Récupération du meilleur modèle
best_model = random_search.best_estimator_

# Réalisation des prédictions
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
custom_threshold = 0.65
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

# Calcul des métriques
metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_pred_proba),
    'classification_report':classification_report(y_val,y_pred),
    'log_loss':log_loss(y_val,y_pred)
}

# Enregistrement des métriques
wandb.log(metrics)
print("Métriques:", metrics)

# Définition de la liste des modèles et de leurs probabilités
model_list = ["Random Forest"]  # Vous pouvez ajouter d'autres modèles si vous en avez
models_prob = [y_pred_proba]   # Ajoutez d'autres probabilités si vous avez d'autres modèles

# Tracé de la courbe "Goal Rate"
bins = list(np.arange(0, 105, 5))
bin_centers = list(np.arange(2.5, 100, 5.0))
df_prob_list, df_prob_bined_list = [], []

for i in range(len(model_list)):
    df_prob = pd.DataFrame(
        list(zip(models_prob[i] * 100, y_val)), columns=["goal_Prob", "goal"]
    )
    df_prob["shot"] = 1
    sum_goal = df_prob["goal"].sum()
    df_prob["percentile"] = df_prob["goal_Prob"].rank(pct=True) * 100
    df_prob["goal_perc_bins"] = pd.cut(df_prob["percentile"], bins, labels=bin_centers)
    df_prob_bined = (
        df_prob.groupby("goal_perc_bins", observed=False)[["shot", "goal"]].sum().reset_index()
    )
    df_prob_bined["goal_rate"] = df_prob_bined["goal"] / df_prob_bined["shot"]
    df_prob_bined_list.append(df_prob_bined)

plt.figure(figsize=(10, 5))
plt.title("Goal Rate")
for i, model in enumerate(model_list):
    sns.lineplot(
        x="goal_perc_bins",
        y="goal_rate",
        data=df_prob_bined_list[i],
        label=model,
        linewidth=2,
    )
plt.xlabel("Shot Probability Model Percentile")
plt.ylabel("Goals / (Shots + Goals)")
plt.xlim([101, -1])
plt.ylim([0, 1])
plt.legend(loc="upper left")
plt.grid()

# Enregistrement de la figure dans Wandb
wandb.log({"Goal Rate Curve": wandb.Image(plt)})
plt.close()

# Ajout du tracé cumulatif des pourcentages de buts
df_prob_bined_list = []
for prob in models_prob:
    df_temp = pd.DataFrame({"goal_prob": prob, "goal": y_val})
    df_temp['goal_perc_bins'] = pd.qcut(df_temp['goal_prob'], q=100, labels=False,duplicates='drop') + 1
    df_temp = df_temp.sort_values(by="goal_perc_bins", ascending=False)
    df_temp['goal_cumsum'] = df_temp['goal'].cumsum() / df_temp['goal'].sum()
    df_prob_bined_list.append(df_temp)

cum_rate_plot = plt.figure(figsize=(10, 5))
plt.title("Cumulative % of Goal")
for i in range(len(model_list)):
    ax = sns.lineplot(
        x="goal_perc_bins", 
        y="goal_cumsum", 
        data=df_prob_bined_list[i], 
        label=f"{model_list[i]}",
        linewidth=2.5)
plt.legend(loc="lower right")
plt.xlabel("Shot probability model percentile")
plt.ylabel("Proportion")
ax.set_xlim(left=101, right=-1)
ax.set_ylim(bottom=0, top=1)
plt.xticks(np.arange(0, 120, 20))

# Enregistrement de la figure dans Wandb
wandb.log({"Cumulative Goal Rate": wandb.Image(cum_rate_plot)})
plt.close()

# Récupération des noms de caractéristiques et de leur importance
preprocessor = best_model.named_steps['preprocessor']
rf_classifier = best_model.named_steps['classifier']

# Récupération des noms de caractéristiques transformées
cat_feature_names = []
for i, (name, trans, cols) in enumerate(preprocessor.transformers_):
    if name == 'cat':
        for j, col in enumerate(cols):
            cats = trans.named_steps['onehot'].categories_[j][1:]  # Ignorer la première catégorie en raison de drop='first'
            cat_feature_names.extend([f"{col}_{cat}" for cat in cats])
    elif name == 'num':
        cat_feature_names.extend(cols)
    elif name == 'bin':
        cat_feature_names.extend(cols)
    elif name == 'shooter':
        cat_feature_names.extend(cols)

# Récupération de l'importance des caractéristiques
feature_importance = pd.DataFrame({
    'feature': cat_feature_names,
    'importance': rf_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Tracé de l'importance des caractéristiques
plt.figure(figsize=(15, 8))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 des caractéristiques les plus importantes')
plt.tight_layout()
wandb.log({"feature_importance_plot": wandb.Image(plt)})
plt.show()

# Création de la matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.ylabel('Valeur réelle')
plt.xlabel('Valeur prédite')
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.show()

# Tracé de la courbe de calibration
CalibrationDisplay.from_predictions(y_val, y_pred_proba, n_bins=10, strategy='uniform')
plt.plot([0, 1], [0, 1], linestyle='--', label='Calibration parfaite')
plt.xlabel("Probabilité moyenne prédite")
plt.ylabel("Fraction de positifs")
plt.title("Courbe de calibration")
plt.legend()
plt.savefig("RandomFrest_calibration_curve.png")
wandb.log({"Calibration Curve": wandb.Image("RandomForest_calibration_curve.png")})

# Tracé de la courbe ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
wandb.log({"roc_curve": wandb.Image(plt)})
plt.show()
plt.close()

# Calcul des centiles
percentiles = np.percentile(y_pred_proba, np.linspace(0, 100, 11))

# Proportion cumulative de buts
cumulative_goal_rate = [np.sum(y_val <= p) / len(y_val) for p in percentiles]
plt.figure()
plt.plot(percentiles, cumulative_goal_rate, marker='o', color='green')
plt.xlabel("Centile du modèle de probabilité de tir")
plt.ylabel("Proportion")
plt.title("Proportion cumulative de buts")
plt.legend()
plt.yticks(np.linspace(0, 1, 11), labels=[f"{int(x*100)}%" for x in np.linspace(0, 1, 11)])
plt.gca().invert_xaxis()
plt.savefig("RandomForest_cumulative_goals_proportion.png")
wandb.log({"Cumulative percentage of goals": wandb.Image("RandomForest_cumulative_goals_proportion.png")})

# Enregistrement du modèle
model_filename = "random_forest_best_model.pkl"
joblib.dump(best_model, model_filename)

# Enregistrement de l'artefact du modèle
artifact = wandb.Artifact("random_forest_best_model", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# Enregistrement de l'importance des caractéristiques
feature_importance.to_csv('feature_importance.csv')
artifact = wandb.Artifact("feature_importance", type="dataset")
artifact.add_file('feature_importance.csv')
wandb.log_artifact(artifact)

# Fin de l'enregistrement de Wandb
wandb.finish()
