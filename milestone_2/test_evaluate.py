import os
import joblib
import wandb
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,classification_report, PrecisionRecallDisplay, brier_score_loss
from sklearn.metrics import roc_curve, auc
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import pickle
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utils_test import separate_seasons,preprocess_and_evaluate



def main():
    # Authentification et configuration Wandb
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        raise ValueError("La clé API WANDB_API_KEY n'est pas définie dans les variables d'environnement.")

    wandb.login(key=api_key)

    entity = "michel-wilfred-essono-university-of-montreal"
    project = "IFT6758.2024-A03"
    run_id = "mpuu11sp"

    wandb.init(project=project, name="Evaluate_on_Test", entity=entity)

    # Charger les données de test
    root_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
    data_dir = os.path.join(root_directory, 'data')
    df = pd.read_csv(f'{data_dir}/test_data.csv')



    # Dossier contenant les modèles
    model_directory = './models'

    # Liste pour stocker les modèles
    models = {}

    for file_name in os.listdir(model_directory):
        file_path = os.path.join(model_directory, file_name)
        
        if os.path.isfile(file_path):
            if file_name.endswith('.pkl'):
                try:
                    model = joblib.load(file_path)
                    model_name = os.path.splitext(file_name)[0]  # Nom du fichier sans extension
                    models[model_name] = model  # Ajouter le modèle au dictionnaire
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_name} : {e}")
            else:
                print(f"Le fichier {file_name} n'est pas un fichier de modèle compatible.")


    # Séparer les saisons
    reg_data, series_data = separate_seasons(df)

   
    



    # Prétraitement et évaluation pour la saison régulière
    print("\n--- Évaluation sur la saison régulière ---")
    preprocess_and_evaluate(models, reg_data, phase="regular")

    # Prétraitement et évaluation pour les playoffs
    print("\n--- Évaluation sur les playoffs ---")
    preprocess_and_evaluate(models, series_data, phase="playoffs")

    print("Évaluation terminée avec succès.")


if __name__ == "__main__":
    main()