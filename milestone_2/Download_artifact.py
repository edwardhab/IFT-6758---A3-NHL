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

# Authentification et configuration Wandb
api_key = os.getenv("WANDB_API_KEY")
if api_key is None:
    raise ValueError("La clé API WANDB_API_KEY n'est pas définie dans les variables d'environnement.")

wandb.login(key=api_key)

entity = "michel-wilfred-essono-university-of-montreal"
project = "IFT6758.2024-A03"
run_id = "mpuu11sp"

wandb.init(project=project, name="Evaluate_on_Test", entity=entity)



# Liste des artefacts à télécharger
artifacts = [
    {"name": "logreg_dist", "type": "model", "version": "v0"},
    {"name": "logreg_ang", "type": "model", "version": "v0"},
    {"name": "logreg_comb", "type": "model", "version": "v0"},
    {"name": "xg_boost_shap_features_optimized_best_model", "type": "model", "version": "v0"},
    {"name": "MLP_models", "type": "model", "version": "v0"}
]




# Initialisation de l'API et du run
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")


output_dir="./models"
# Création du répertoire de sortie
os.makedirs(output_dir, exist_ok=True)

# Téléchargement et chargement des artefacts
models = {}
for artifact_info in artifacts:
    artifact_name = artifact_info["name"]
    artifact_type = artifact_info["type"]
    artifact_version = artifact_info["version"]

    try:
        # Récupérer l'artefact depuis Wandb
        run = wandb.init()
        artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}:{artifact_version}",type='model')
        download_path = artifact.download(root=output_dir)

        # Charger les fichiers de l'artefact
        model_loaded = False
        for file in os.listdir(download_path):
            if file.endswith('.pkl'):  # Vérifier les fichiers `.pkl`
                model_path = os.path.join(download_path, file)
                models[artifact_name] = joblib.load(model_path)
                model_loaded = True
                print(f"Modèle chargé : {artifact_name} depuis {file}")

        if not model_loaded:
            print(f"Aucun modèle trouvé dans l'artefact : {artifact_name}")

    except wandb.errors.CommError as e:
        print(f"Erreur de communication avec Wandb pour l'artefact {artifact_name}: {e}")
    except Exception as e:
        print(f"Erreur lors du téléchargement ou du chargement de l'artefact {artifact_name}: {e}")

# Vérification des modèles chargés
if not models:
    raise RuntimeError("Aucun modèle n'a été chargé. Vérifiez les artefacts ou leur disponibilité sur Wandb.")
else:
    print(f"{len(models)} modèles ont été chargés avec succès.")
    
    