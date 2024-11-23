import os
import joblib
import wandb
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,classification_report, PrecisionRecallDisplay, brier_score_loss
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def separate_seasons(df):
    # Vérification de la longueur minimale de 'gameId'
    if not df['gameId'].astype(str).str.len().ge(6).all():
        raise ValueError("Certains 'gameId' n'ont pas au moins 6 caractères.")
    
    # Convertir le 'gameId' en chaîne pour pouvoir extraire les 6ème et 7ème chiffres
    df['gameId_str'] = df['gameId'].astype(str)
    
    # Extraire les 6ème et 7ème chiffres pour déterminer si c'est une saison régulière ou des playoffs
    df['season_type'] = df['gameId_str'].str[4:6]

    # Séparer en saison régulière (02) et playoffs (03)
    regular_season_df = df[df['season_type'] == '02'].copy()
    playoffs_df = df[df['season_type'] == '03'].copy()
    
    # Nettoyage des colonnes auxiliaires
    df.drop(columns=['gameId_str', 'season_type'], inplace=True, errors='ignore')
    regular_season_df.drop(columns=['gameId_str', 'season_type'], inplace=True, errors='ignore')
    playoffs_df.drop(columns=['gameId_str', 'season_type'], inplace=True, errors='ignore')

    return regular_season_df, playoffs_df


def Logreg_feature_engineering(df):
    X_test = df[['shotDistance', 'shotAngle']]
    y_test = df['result']
    return X_test.to_numpy(),y_test.to_numpy()

def MLP_feature_engineering(df):
    y = df['result']

    df = df[['shotDistance','shotAngle','xCoord','yCoord','lastEventType','emptyNetGoal','lastEventXCoord','lastEventYCoord',
         'timeElapsedSinceLastEvent','distanceFromLastEvent','rebound','speed','timeInPeriod','shotType']]

    # Conversion des colonnes booléennes en entiers
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    
    # Transformation des colonnes catégoriques
    frequency_map = df['lastEventType'].value_counts() / len(df)
    df.loc[:, 'lastEventType_encoded'] = df['lastEventType'].map(frequency_map)
    

    frequency_map2 = df['shotType'].value_counts() / len(df)
    df.loc[:, 'shotType'] = df['shotType'].map(frequency_map2)
    
    # Sélection uniquement des colonnes numériques
    df = df.select_dtypes(exclude=['object'])

    # Suppression de colonnes spécifiques si elles existent
    columns_to_drop = ['changeInShotAngle', 'timeInPeriod']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Calcul des corrélations avec la colonne 'result'
    if 'result' in df.columns:  # Vérifie que la colonne 'result' existe
        df_temp = df.copy()
        correlations = df_temp.corr()['result'].drop('result').sort_values(ascending=False)
        # Filtrage des colonnes avec corrélation absolue > 0.04
        threshold = 0.05
        selected_features = correlations[correlations.abs() > threshold].index
        df = df[selected_features.union(['result'])]  # Conserve 'result' dans le DataFrame
    
    return df,y

def Xgboost_feature_engineering(df):
    
    # Vérifier si toutes les colonnes nécessaires sont présentes
    required_columns = [
        'gameId', 'changeInShotAngle', 'result', 'timeInPeriod', 'shooter',
        'goalie', 'HomevsAway', 'shotType', 'emptyNetAway', 'emptyNetHome',
        'powerplayHome', 'powerplayAway', 'offensiveSide', 'rebound', 'lastEventType'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne requise '{col}' est absente du DataFrame.")

    
    # Gestion des valeurs manquantes
    df['changeInShotAngle'] = df['changeInShotAngle'].fillna(0)  # Valeur par défaut : 0
    
    
    df = df.dropna()  # Supprimer les lignes restantes avec des valeurs manquantes

    print(df['result'])

    # Encodage des colonnes catégoriques
    categorical_features = [
        'timeInPeriod', 'shooter', 'goalie', 'HomevsAway', 'shotType',
        'emptyNetAway', 'emptyNetHome', 'powerplayHome', 'powerplayAway',
        'offensiveSide', 'rebound', 'lastEventType'
    ]

    label_encoders = {}  # Dictionnaire pour sauvegarder les encodeurs
    for feature in categorical_features:
        if df[feature].dtype == 'object':  # Si le type est 'object'
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))  # Encodage
            label_encoders[feature] = le

    # Créer la cible (y)
    y = df['result'].astype(int)
    
    selected_columns = [
        'shotDistance', 'yCoord', 'shotType', 'timeElapsedSinceLastEvent',
        'shotAngle', 'eventId', 'speed', 'powerplayAway', 'gameId', 'shooter'
    ]
    df = df[selected_columns]
    # Retourner le DataFrame prétraité (X) et les labels (y)
    return df, y



def preprocess_and_evaluate(models, data, phase="regular"):
    """
    Prépare les données pour les modèles, effectue les prédictions, évalue les performances,
    et trace les courbes ROC.

    Parameters:
        models (dict): Dictionnaire des modèles chargés.
        data (DataFrame): Données de la saison (régulière ou playoffs).
        phase (str): Phase de la saison ("regular" ou "playoffs").

    Returns:
        None
    """
    # Préparation des données
    X_test_logreg, y_test_logreg = Logreg_feature_engineering(data)
    X_test_dist = X_test_logreg[:, 0]
    X_test_ang = X_test_logreg[:, 1]
    X_test_comb = X_test_logreg

    X_test_xgb, y_test_xgb = Xgboost_feature_engineering(data)
    X_test_MLP, y_test_MLP = MLP_feature_engineering(data)

    X_test_data = {
        "logreg_dist": (X_test_dist, y_test_logreg),
        "logreg_ang": (X_test_ang, y_test_logreg),
        "logreg_comb": (X_test_comb, y_test_logreg),
        "xg_boost": (X_test_xgb, y_test_xgb),
        "MLP": (X_test_MLP, y_test_MLP),
    }

    # Listes pour stocker les résultats ROC
    model_list = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    y_pred_list =[]
    y_pred_proba_list =[]

    # Évaluation des modèles et collecte des métriques
    for model_name, model in models.items():
        if model_name not in X_test_data:
            print(f"Les données pour le modèle {model_name} ne sont pas disponibles pour {phase}.")
            continue

        X_test, y_test = X_test_data[model_name]
        if X_test.ndim == 1:
           X_test = X_test.reshape(-1, 1)
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilité pour la classe positive
        y_pred_proba_list.append(y_pred_proba)
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nÉvaluation du modèle : {model_name} ({phase})")
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy : {accuracy:.4f}")

        # Log des résultats sur Wandb
        wandb.log({
            f"{model_name}_accuracy_{phase}": accuracy,
            f"{model_name}_classification_report_{phase}": class_report
        })

        # Calcul des points pour la courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Stockage des résultats
        model_list.append(model_name)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

    # Tracé de la courbe ROC pour tous les modèles
    roc_auc_plot = plt.figure(figsize=(10, 6))
    lw = 2
    for i in range(len(model_list)):
        plt.plot(
            fpr_list[i],
            tpr_list[i],
            lw=lw,
            label="ROC curve %s (area = %0.2f)" % (model_list[i], roc_auc_list[i])
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, label="Ideal Random Baseline", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({phase.title()})")
    plt.legend(loc="lower right")

    # Sauvegarde du graphique
    output_dir = "./figures"
    os.makedirs(output_dir, exist_ok=True)
    roc_file = os.path.join(output_dir, f"roc_curve_{phase}.png")
    plt.savefig(roc_file)
    plt.close(roc_auc_plot)

    # Log du graphique sur Wandb
    wandb.log({f"roc_curve_{phase}": wandb.Image(roc_file)})

    print(f"Courbe ROC pour {phase} sauvegardée dans {roc_file} et loggée sur Wandb.")

    bins = list(np.arange(0, 105, 5))
    bin_centers = list(np.arange(2.5, 100, 5.0))

    # Préparation des DataFrames pour les courbes
    df_prob_list = []
    df_prob_bined_list = []

    for i, model_name in enumerate(model_list):
        X_test, y_test = X_test_data[model_name]
        # Création du DataFrame de base pour un modèle
        df_prob = pd.DataFrame(
            list(zip(y_pred_list[i], y_test, y_pred_proba_list[i] * 100)),
            columns=["goal_pred", "goal", "goal_Prob"]
        )
        df_prob["shot"] = 1
        sum_goal = df_prob["goal"].sum()

        # Calcul des percentiles et binning
        df_prob["percentile"] = df_prob["goal_Prob"].rank(pct=True) * 100
        df_prob["goal_perc_bins"] = pd.cut(df_prob["percentile"], bins, labels=bin_centers)
        df_prob_bined = (
            df_prob[["goal_perc_bins", "shot", "goal"]]
            .groupby(["goal_perc_bins"])
            .sum()
            .reset_index()
        )
        df_prob_bined["goal_rate"] = df_prob_bined["goal"] / df_prob_bined["shot"]
        df_prob_bined["goal_cum"] = df_prob_bined["goal"] / sum_goal
        df_prob_bined["goal_cumsum"] = 1 - df_prob_bined["goal_cum"].cumsum()

        # Stockage des DataFrames
        df_prob_list.append(df_prob)
        df_prob_bined_list.append(df_prob_bined)

        

    # Tracé des Goal Rates
    sns.set(style="whitegrid")
    goal_rate_plot = plt.figure(figsize=(10, 5))
    plt.title(f"Goal Rate ({phase.title()})")

    for i in range(len(model_list)):
        ax = sns.lineplot(
            x="goal_perc_bins",
            y="goal_rate",
            data=df_prob_bined_list[i],
            legend=False,
            linewidth=2.5,
            label="%s" % model_list[i]
        )

    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Goals / (Shots + Goals)")
    ax.set_xlim(left=101, right=-1)
    ax.set_ylim(bottom=0, top=1)
    plt.xticks(np.arange(0, 120, 20))
    plt.legend(bbox_to_anchor=(0.9, 0.88))

    # Sauvegarde du graphique
    output_dir = "./figures"
    os.makedirs(output_dir, exist_ok=True)
    goal_rate_file = os.path.join(output_dir, f"goal_rate_{phase}.png")
    plt.savefig(goal_rate_file)
    plt.close(goal_rate_plot)

    # Log du graphique sur Wandb
    wandb.log({f"goal_rate_{phase}": wandb.Image(goal_rate_file)})

    print(f"Graphique Goal Rate ({phase}) sauvegardé dans {goal_rate_file} et loggé sur Wandb.")

# Tracé Cumulative %
    cum_rate_plot = plt.figure(figsize=(10, 5))
    plt.title(f"Cumulative % of Goal ({phase.title()})")
    for i in range(len(model_list)):
        ax = sns.lineplot(
            x="goal_perc_bins", y="goal_cumsum",
            data=df_prob_bined_list[i],
            label=f"{model_list[i]}", linewidth=2.5
        )
    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Cumulative % of Goal")
    ax.set_xlim(left=101, right=-1)
    plt.legend(loc="lower right")
    plt.savefig(f"./figures/cumulative_goal_{phase}.png")
    plt.close(cum_rate_plot)

    # Log des courbes sur Wandb
    wandb.log({
       
        f"cumulative_goal_{phase}": wandb.Image(f"./figures/cumulative_goal_{phase}.png")
    })

    print(f"Graphiques Cumulative % sauvegardé et loggé sur Wandb.")

    # Ajout du calcul et tracé des courbes de calibration
    calibration_plot = plt.figure(figsize=(7, 7))
    plt.title(f"Calibration Curve ({phase.title()})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model_name in model_list:
        prob_true, prob_pred = calibration_curve(
            X_test_data[model_name][1],  # Les vraies étiquettes
            y_pred_proba_list[model_list.index(model_name)],  # Les probabilités prédites
            n_bins=20
        )
        plt.plot(
            prob_pred, prob_true, "s-", label=f"{model_name}"
        )
    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Proportion")
    plt.legend(loc="upper left", ncol=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    calibration_file = os.path.join("./figures", f"calibration_curve_{phase}.png")
    plt.savefig(calibration_file)
    plt.close(calibration_plot)
    wandb.log({f"calibration_curve_{phase}": wandb.Image(calibration_file)})

    print(f"Courbe de calibration sauvegardée dans {calibration_file} et loggée sur Wandb.")