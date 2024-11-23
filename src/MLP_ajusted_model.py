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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import os 
import pickle
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Récupérer la clé API depuis la variable d'environnement
api_key = os.getenv("WANDB_API_KEY")

# Authentifier Wandb avec la clé API récupérée
wandb.login(key=api_key)

# Initialisation de Wandb avec nom de projet
wandb.init(project="IFT6758.2024-A03", name="MLP_best_model", entity="michel-wilfred-essono-university-of-montreal")

df = pd.read_csv('train_data.csv')
df = df.drop(columns=['changeInShotAngle'])


target = 'result'
df = df[[col for col in df.columns if col != target] +[target]]

df['home_goals'] = ((df['HomevsAway'] == 'home') & (df['eventType'] == 'goal')).groupby(df['gameId']).cumsum()
df['away_goals'] = ((df['HomevsAway'] == 'away') & (df['eventType'] == 'goal')).groupby(df['gameId']).cumsum()
def calculate_score_differential(row):
    if row['HomevsAway'] == 'home':
        return row['home_goals'] - row['away_goals']
    else:
        return row['away_goals'] - row['home_goals']
    
df['score_differential'] = df.apply(calculate_score_differential, axis=1)

X_train = df.drop(columns=['result'])

y_train = df['result']

for col in X_train.select_dtypes(include='bool').columns:
    X_train[col] = X_train[col].astype(int)

# Convertir une colonne contenant des durées (hh:mm:ss) en secondes
X_train['timeInPeriod'] = X_train['timeInPeriod'].apply(lambda x: sum(int(i) * 60 ** idx for idx, i in enumerate(reversed(x.split(':')))))

frequency_map = X_train['lastEventType'].value_counts() / len(X_train)
X_train['lastEventType'] = X_train['lastEventType'].map(frequency_map)


frequency_map = X_train['shotType'].value_counts() / len(X_train)
X_train['shotType'] = X_train['shotType'].map(frequency_map)

shooter_freq = X_train['shooter'].value_counts() / len(X_train)

# Appliquer l'encodage de fréquence à la colonne 'shooter'
X_train['shooter'] = X_train['shooter'].map(shooter_freq)

X_train = X_train.select_dtypes(exclude=['object'])

# Ajouter temporairement `result` dans X_train pour calculer les corrélations
corr_df = X_train.copy()
corr_df['result'] = y_train

# Calcul des corrélations
correlations = corr_df.corr()['result'].drop('result').sort_values(ascending=False)


threshold = 0.05  # Par exemple, garder les corrélations > 0.05
selected_features = correlations[correlations.abs() > threshold].index
X_train = X_train[selected_features]



X_tr, X_val, y_tr, y_val = train_test_split(X_train.fillna(0),y_train, test_size=0.25,random_state=10, shuffle = True)



def pipeline(classifier,feature_selection,encoder):

    enc = {
        'Label': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        'OneHot': OneHotEncoder(handle_unknown='ignore')
    }
    numeric_transformer = Pipeline([('Impt', SimpleImputer(strategy='mean')),('scaler', StandardScaler()),]) #Scaling of numerical col.
    categorical_transformer = Pipeline([('Impt', SimpleImputer(strategy='most_frequent')),('encoder', enc[encoder]),]) #Encoding of categorical col.  
    preprocessor = ColumnTransformer(
        transformers=[
            ("Numerical_transform", numeric_transformer, make_column_selector(dtype_include="number")),
            ("Categorical_transform", categorical_transformer, make_column_selector(dtype_exclude="number")),
        ]
    )  
    return Pipeline(steps=[('preprocessor', preprocessor)] + [('feature_select', feature_selection)] + [('classifier', classifier)])

classifer=MLPClassifier(hidden_layer_sizes='hidden_layer_sizes', 
                        activation = 'activation',solver='solver',
                        alpha='alpha',learning_rate='learning_rate',
                        max_iter=1000)
feature_selection= SelectFromModel(estimator=LinearSVC(C=0.1, penalty="l1", dual=False))
ecoder='Label'
pipe = pipeline(classifer,feature_selection,ecoder)

parameter_space = {
    'classifier__hidden_layer_sizes': [(50,50)],
    'classifier__activation': ['relu'],
    'classifier__solver': ['adam'],
    'classifier__alpha': [0.0001],
    'classifier__learning_rate': ['constant'],
}

# Search for best hyperparamters
cv_strategy = StratifiedKFold(n_splits=2, shuffle=True) # Cross-validation Method
search = RandomizedSearchCV(pipe, parameter_space, n_jobs=-1, cv=cv_strategy, scoring='f1_weighted').fit(X_tr, y_tr
                                                                                                         )

pipe = search.best_estimator_
MLP=pipe.fit(X_tr, y_tr)

# Sauvegarde des modèles avec joblib
os.makedirs("MLP_models", exist_ok=True)
joblib.dump(MLP, "MLP_models/MLP.pkl")

# Enregistrement des modèles sur Wandb
artifact = wandb.Artifact("MLP_models", type="model")
artifact.add_dir("MLP_models")
wandb.log_artifact(artifact)

y_pred_val = pipe.predict(X_val) 
predictions=pipe.predict_proba(X_val)

report = classification_report(y_val, y_pred_val, output_dict=True)
results = pd.DataFrame(report).transpose()

f_matrix=confusion_matrix(y_val, y_pred_val)

sns.heatmap(f_matrix, annot=True,fmt=".0f", cmap=sns.cubehelix_palette(as_cmap=True),linewidth=.5)

fig = plt.figure(figsize=(10, 10))
# Get FPR, TPR and AUC
fpr, tpr, _ = roc_curve(y_val,predictions[:,1].ravel())
roc_auc = auc(fpr, tpr)
sns.set()
plt.plot(fpr, tpr, label="ROC curve(area = %0.3f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="red",label="Random Baseline", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC CURVE")
plt.legend(loc="lower right", fontsize='small')

# Log de la figure dans wandb
wandb.log({"ROC Curves": wandb.Image(plt)})
plt.close()



bins = list(np.arange(0, 105,  5))
bin_centers = list(np.arange(2.5, 100,  5.0))

df_prob = pd.DataFrame(list(zip(y_pred_val ,y_val,predictions[:,1]*100)), columns = ['goal_pred', 'goal','goal_Prob'])
df_prob['shot'] = 1
sum_goal = df_prob['goal'].sum()
df_prob['percentile'] = df_prob['goal_Prob'].rank(pct=True) * 100
df_prob['goal_perc_bins'] = pd.cut(df_prob['percentile'], bins, labels = bin_centers)
df_prob_bined = df_prob[['goal_perc_bins', 'shot', 'goal' ]].groupby(['goal_perc_bins']).sum().reset_index()
df_prob_bined['goal_rate'] = (df_prob_bined['goal']/df_prob_bined['shot'])
df_prob_bined['goal_cum'] = (df_prob_bined['goal']/sum_goal)
df_prob_bined['goal_cumsum'] = 1-df_prob_bined['goal_cum'].cumsum()


fig = plt.figure(figsize = (15,5))
plt.title(f"Goal Rate")
ax = sns.lineplot(x = 'goal_perc_bins', y = 'goal_rate', data = df_prob_bined, linewidth = 2)
fig.legend(loc="upper right")
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('shot probability model percentile')
plt.ylabel('goals/(shots+goals)')
ax.set_xlim(left=101, right=-1)
ax.set_ylim(bottom=0, top=1)
plt.yticks(np.arange(0,1.1,.1))
yvals = ax.get_yticks()
ax.set_yticklabels(["{:,.0%}".format(y) for y in yvals], fontsize=12)
plt.xticks(np.arange(0,110,10))

# Log de la figure dans wandb
wandb.log({"Goal Rate Curve": wandb.Image(plt)})
plt.close()



fig = plt.figure(figsize = (15,5))
plt.title(f"Cumulative % of goal")
ax = sns.lineplot(x = 'goal_perc_bins', y = 'goal_cumsum', data = df_prob_bined, legend = False, linewidth = 2)
fig.legend(loc="upper right")
plt.grid(color='gray', linestyle='--', linewidth=.5)
plt.xlabel('shot probability model percentile')
plt.ylabel('proportion')
ax.set_xlim(left=101, right=-1)
ax.set_ylim(bottom=0, top=1)
plt.yticks(np.arange(0,1.2,.1))
yvals = ax.get_yticks()
ax.set_yticklabels(["{:,.0%}".format(y) for y in yvals], fontsize=12)
plt.xticks(np.arange(0,110,10))

# Log de la figure dans wandb
wandb.log({"Cumulative Goal Rate": wandb.Image(fig)})
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))

disp = CalibrationDisplay.from_predictions(y_val, predictions[:,1].ravel(), n_bins=10, ax=ax)
plt.grid(color='gray', linestyle='--', linewidth=.5)
plt.legend(loc="center right")
plt.title(f"calibration curve")

# Log de la figure dans wandb
wandb.log({"Calibration Curves": wandb.Image(plt)})
plt.close()



# Finalisation de wandb
wandb.finish()