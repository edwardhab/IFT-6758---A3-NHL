import os
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay

# Classe PlayerTargetEncoder modifiée pour mieux gérer plusieurs types de joueurs
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

# Classe TargetEncoder pour encoder avec lissage des cibles
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = None

    def fit(self, X, y):
        # Assurez-vous que X est une pandas Series
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Calcul de la moyenne globale
        self.global_mean = np.mean(y)
        
        # Calcul des moyennes pour chaque catégorie avec un lissage
        counts = X.value_counts()
        for category in counts.index:
            cat_mask = (X == category)
            n = counts[category]
            cat_mean = np.mean(y[cat_mask])
            # Appliquer un lissage
            smoothed_mean = (n * cat_mean + self.smoothing * self.global_mean) / (n + self.smoothing)
            self.target_means[category] = smoothed_mean
            
        return self

    def transform(self, X):
        # Assurez-vous que X est une pandas Series
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.map(self.target_means).fillna(self.global_mean).to_numpy().reshape(-1, 1)

# Initialisation de Wandb
wandb.init(project="IFT6758.2024-A03", name="exp_knn_optimized", entity="michel-wilfred-essono-university-of-montreal")

# Chargement des données
train_data = pd.read_csv('train_data.csv')

# Ajouter une colonne pour identifier les powerplays (supériorité ou infériorité numérique)
train_data['powerplay'] = ((train_data['HomevsAway'] == 1) & (train_data['powerplayHome'] == 1) | (train_data['HomevsAway'] == 0) & (train_data['powerplayAway'] == 1)).astype(int) - ((train_data['HomevsAway'] == 1) & (train_data['powerplayAway'] == 1) | (train_data['HomevsAway'] == 0) & (train_data['powerplayHome'] == 1)).astype(int)

# Convertir le temps en secondes
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return None

# Calculer la différence de score
def calculate_score_differential(row):
    if row['HomevsAway'] == 'home':
        return row['home_goals'] - row['away_goals']
    else:
        return row['away_goals'] - row['home_goals']

# Définir un scorer personnalisé basé sur F-beta
def custom_scorer(y_true, y_pred_proba, beta=1.0):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Calcul du F-beta score
        if precision + recall > 0:
            score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            best_score = max(best_score, score)
            
    return best_score

# Ajouter cumulatif des tirs par période et équipe
train_data['teamId'] = np.where(
        train_data['HomevsAway'] == 'home',
        train_data['homeTeamId'],
        train_data['awayTeamId']
    )
for col in ['speed', 'distanceFromLastEvent', 'lastEventXCoord', 'lastEventYCoord']:
    median_by_group = train_data.groupby('lastEventType')[col].transform('median')
    train_data[col] = train_data[col].fillna(median_by_group).fillna(train_data[col].median())
train_data['shots_period_cumsum'] = train_data.groupby(['gameId', 'period', 'teamId'])['eventType'].cumcount()
train_data['timeInPeriod'] = train_data['timeInPeriod'].apply(convert_time_to_seconds)

# Ajouter temps écoulé depuis le dernier tir
train_data['time_since_last_shot'] = train_data.groupby(['gameId', 'period', 'teamId'])['timeInPeriod'].diff().fillna(0)
train_data['powerplay'] = ((train_data['HomevsAway'] == 'home') & (train_data['powerplayHome'] == 1) | (train_data['HomevsAway'] == 'away') & (train_data['powerplayAway'] == 1)).astype(int) - ((train_data['HomevsAway'] == 'home') & (train_data['powerplayAway'] == 1) | (train_data['HomevsAway'] == 'away') & (train_data['powerplayHome'] == 1)).astype(int)

train_data['consecutive_penalties'] = (
        train_data.groupby(['gameId', 'teamId'])
        ['powerplay']
        .cumsum()
    )

# Caractéristiques des groupes définis
numerical_features = [
    'shotDistance', 'shotAngle', 'timeInPeriod', 'xCoord', 'yCoord',
    'timeElapsedSinceLastEvent', 'distanceFromLastEvent', 'changeInShotAngle', 
    'speed', 'powerplay', 'shots_period_cumsum', 'time_since_last_shot', 
    'consecutive_penalties'
]

frequency_features= [
    'shotType'
]

categorical_features = [
    'lastEventType', 'offensiveSide', 'HomevsAway'
]
if 'eventType' in train_data.columns and 'goal' in train_data['eventType'].unique():
    train_data = train_data.drop(columns=['eventType'])
binary_features = [
    'emptyNetGoal', 'rebound'
]
shooter_features = ['shooter','goalie']

# Conversion des colonnes binaires en entiers
for col in binary_features:
    train_data[col] = train_data[col].astype(int)

# Suppression des lignes avec des valeurs manquantes
train_data = train_data.dropna(subset=numerical_features + categorical_features + binary_features + frequency_features+shooter_features)

# Préparer les données d'entraînement et de validation
X = train_data[numerical_features + categorical_features + binary_features + frequency_features+shooter_features]
y = train_data['result'].astype(int)

# Division en ensemble d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Sous-échantillonnage pour équilibrer les classes
sampling_strategy = 0.5  # Ajuster ce ratio selon les besoins
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

# Pipeline de prétraitement
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

frequency_transformer = Pipeline(steps=[
    ('scaler', TargetEncoder(smoothing=10))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('shooter', PlayerTargetEncoder(smoothing=30), shooter_features),
        ('num', numeric_transformer, numerical_features),
        ('freq', frequency_transformer, frequency_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

# Pipeline KNN
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Définir la grille des hyperparamètres pour KNN
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
    'classifier__p': [1, 2],  # p=1 pour manhattan, p=2 pour euclidienne
    'classifier__leaf_size': [20, 30, 40]
}

cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Recherche aléatoire des hyperparamètres avec RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=cv_splitter,
    scoring='precision',
    random_state=42,
    n_jobs=8,
    verbose=2
)

print("L'entraînement du modèle commence...")
# Ajustement du modèle
random_search.fit(X_train, y_train)

# Obtenir les meilleurs paramètres
best_params = random_search.best_params_

# Formatter les paramètres pour Wandb
wandb_params = {
    'n_neighbors': best_params['classifier__n_neighbors'],
    'weights': best_params['classifier__weights'],
    'metric': best_params['classifier__metric'],
    'p': best_params['classifier__p'],
    'leaf_size': best_params['classifier__leaf_size']
}

# Journalisation des paramètres individuellement
for param_name, param_value in wandb_params.items():
    wandb.log({f"best_parameter/{param_name}": param_value})

# Enregistrement des paramètres complets
wandb.log({"best_parameters": wandb_params})

# Configuration de Wandb
wandb.config.update({
    "model_type": "KNN",
    "best_parameters": wandb_params,
    "sampling_strategy": sampling_strategy
})

# Création d'une table pour Wandb
wandb_table = wandb.Table(
    columns=["Parameter", "Value"],
    data=[[k, str(v)] for k, v in wandb_params.items()]
)
wandb.log({"best_parameters_table": wandb_table})

# Obtenir le meilleur modèle
best_model = random_search.best_estimator_

# Tracer les performances du KNN en fonction du nombre de voisins
neighbors_range = range(1, 31)  # Tester k de 1 à 30
results = []

for n in neighbors_range:
    knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=n))
    ])
    knn.fit(X_train, y_train)
    
    # Prédire les probabilités
    y_pred_proba = knn.predict_proba(X_val)[:, 1]
    y_pred = knn.predict(X_val)
    
    # Calculer les métriques de performance
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results.append({
        'n_neighbors': n,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    })

# Convertir les résultats en DataFrame pour la visualisation
results_df = pd.DataFrame(results)

# Tracer les métriques en fonction du nombre de voisins
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_neighbors'], results_df['accuracy'], label='Accuracy', marker='o')
plt.plot(results_df['n_neighbors'], results_df['precision'], label='Precision', marker='o')
plt.plot(results_df['n_neighbors'], results_df['recall'], label='Recall', marker='o')
plt.plot(results_df['n_neighbors'], results_df['roc_auc'], label='ROC AUC', marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Performance Metric')
plt.title('KNN Performance vs Number of Neighbors')
plt.xticks(neighbors_range)
plt.legend()
plt.grid()
plt.show()
plt.close()

# Faire des prédictions
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
custom_threshold = 0.65
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

# Calculer les métriques
metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_pred_proba)
}

# Journalisation des métriques
wandb.log(metrics)
print("Metrics:", metrics)

# Créer une matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.show()

# Créer une courbe ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
wandb.log({"roc_curve": wandb.Image(plt)})
plt.show()
plt.close()

# Tracé de la courbe de calibration
CalibrationDisplay.from_predictions(y_val, y_pred_proba, n_bins=10, strategy='uniform')
plt.plot([0, 1], [0, 1], linestyle='--', label='Calibration parfaite')
plt.xlabel("Probabilité moyenne prédite")
plt.ylabel("Fraction de positifs")
plt.title("Courbe de calibration")
plt.legend()
plt.savefig("KNN_calibration_curve.png")
wandb.log({"Calibration Curve": wandb.Image("KNN_calibration_curve.png")})

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
# Sauvegarder le modèle
model_filename = "knn_best_model.pkl"
joblib.dump(best_model, model_filename)

# Journalisation du modèle en tant qu'artefact
artifact = wandb.Artifact("knn_best_model", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# Terminer l'exécution de Wandb
wandb.finish()