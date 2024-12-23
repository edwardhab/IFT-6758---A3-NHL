import os
import pandas as pd
import numpy as np
import cupy as cp
from sklearn.model_selection import train_test_split
import xgboost as xgb
import wandb
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import seaborn as sns
import joblib

class XGBoostModelTrainer:
    def __init__(self, data_dir):
        """
        Initializes the model trainer class.
        
        Args:
            data_dir (str): The directory containing the data file.
        """
        self.data_dir = data_dir
        self.df = self.load_data()
        self.preprocess_data()
        cp.cuda.Device(0).use()  # Use GPU 0
        self.train_df, self.val_df = self.split_train_val()
        

    def load_data(self):
        """
        Loads the data from the specified directory.
        
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = os.path.join(self.data_dir, 'enhanced_parsed_shot_events.csv')
        return pd.read_csv(file_path)
        
    def preprocess_data(self):
        """
        Handle preprocessing.
        """
        # Fill the 'changeInShotAngle' column with 0s
        self.df = self.df[self.df['gameId'].astype(str).str[5] == '2']
        self.df['changeInShotAngle'] = self.df['changeInShotAngle'].fillna(0)
        # Drop other rows
        self.df = self.df.dropna()
        # Encode non numerical features
        self.df['result'] = self.df['result'].map({'goal': 1, 'no goal': 0}).fillna(0)
        categorical_features = [
            'timeInPeriod', 'shooter', 'goalie', 'HomevsAway', 'shotType',
            'emptyNetAway', 'emptyNetHome', 'powerplayHome', 'powerplayAway',
            'offensiveSide', 'rebound', 'lastEventType'
        ]
        self.encode_features(categorical_features)

    def encode_features(self, categorical_features):
        """
        Encodes categorical features in the DataFrame using LabelEncoder.
        """
        # Apply LabelEncoder to each categorical feature
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            self.df[feature] = le.fit_transform(self.df[feature].astype(str))  # Ensure data type is string
            label_encoders[feature] = le  # Store encoder for potential inverse transform (optional)
        
        self.label_encoders = label_encoders 

    def split_train_val(self):
        """
        Splits the DataFrame into training and validation sets by excluding the 20202021 season,
        then splitting the remaining data into 80% training and 20% validation.
        
        Returns:
            pd.DataFrame, pd.DataFrame: Training and validation DataFrames.
        """
        train_val_df = self.df[self.df['season'] != 20202021]
        train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=42)
        return train_df, val_df

    def transfer_to_gpu(self, dataframe):
        """
        Transfers a Pandas DataFrame to GPU using CuPy.
        
        Args:
            dataframe (pd.DataFrame): The input DataFrame.
        
        Returns:
            cp.ndarray: The GPU array representation of the DataFrame.
        """
        return cp.asarray(dataframe.values)
    
    def train_xgboost_base(self, features, target, project_name="IFT6758.2024-A03", run_name="XGBoost_Run_Base"):
        """
        Trains an XGBoost classifier using specified features and logs metrics with Weights & Biases.
        
        Args:
            features (list): List of feature column names.
            target (str): The target column name.
            run_name (str): Name for the WandB run.
            project_name (str): The name of the Weights & Biases project.
        """
        # Prepare data
        X_train = self.train_df[features].astype(float)
        y_train = self.train_df[target].astype(int)
        X_val = self.val_df[features].astype(float)
        y_val = self.val_df[target].astype(int)

        # Initialize WandB
        wandb.init(project=project_name, config={"features": features, "model": "XGBoost"}, name=run_name)

        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {"objective": "binary:logistic", 
                  "eval_metric": "logloss",
                  "tree_method": "hist",
                  "device": "cuda"}
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Save and log model using `.xgb` and `joblib`
        model_file_xgb = f"{run_name}_model.xgb"
        model_file_pkl = f"{run_name}_model.pkl"
        
        model.save_model(model_file_xgb)
        joblib.dump(model, model_file_pkl)

        artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
        artifact.add_file(model_file_xgb)  # Add .xgb
        artifact.add_file(model_file_pkl)  # Add .pkl
        wandb.log_artifact(artifact)

        # Make predictions and evaluate
        y_pred_proba = model.predict(dval)
        self.plot_and_log_metrics(y_val, y_pred_proba, run_name)
        self.plot_best_params(params, run_name)


    def train_xgboost_optimized(self, features, target, project_name="IFT6758.2024-A03", run_name="XGBoost_Optimized"):
        """
        Trains an XGBoost classifier using specified features with Bayesian optimization for hyperparameter tuning.
        Logs metrics and saves only the best model among all boosters.
        """
        # Prepare data
        X_train = self.train_df[features].astype(float)
        y_train = self.train_df[target].astype(int)
        X_val = self.val_df[features].astype(float)
        y_val = self.val_df[target].astype(int)

        # Transfer data to GPU
        X_train_gpu = self.transfer_to_gpu(X_train)
        y_train_gpu = self.transfer_to_gpu(y_train)
        X_val_gpu = self.transfer_to_gpu(X_val)
        y_val_gpu = self.transfer_to_gpu(y_val)

        # Track the best booster and its performance
        best_model = None
        best_params = None
        best_booster = None
        best_roc_auc = 0

        # Function to optimize parameters for a given booster
        def xgb_evaluate(max_depth=None, learning_rate=None, n_estimators=None, subsample=None, colsample_bytree=None, booster=None):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": learning_rate,
                "n_estimators": int(n_estimators),
                "booster": booster,
                "device": "cuda"  # Enable GPU training
            }
            if booster in ["gbtree", "dart"]:
                params.update({
                    "max_depth": int(max_depth),
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "tree_method": "hist",  # GPU-compatible method
                })
            elif booster == "gblinear":
                # Remove tree-specific parameters for gblinear
                params.pop("max_depth", None)
                params.pop("subsample", None)
                params.pop("colsample_bytree", None)

            xgb_model = xgb.XGBClassifier(**params)
            # Cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(xgb_model, cp.asnumpy(X_train_gpu), cp.asnumpy(y_train_gpu), cv=skf, scoring='roc_auc')
            return np.mean(cv_scores)  # Return mean ROC AUC across folds

        # Booster testing loop
        for booster in ["gbtree", "gblinear", "dart"]:
            print(f"\nOptimizing for booster: {booster}")
            param_bounds = {"learning_rate": (0.01, 0.3), "n_estimators": (50, 300)}

            if booster in ["gbtree", "dart"]:
                param_bounds.update({
                    "max_depth": (3, 10),
                    "subsample": (0.5, 1.0),
                    "colsample_bytree": (0.5, 1.0),
                })

            # Perform Bayesian optimization
            optimizer = BayesianOptimization(
                f=lambda learning_rate, n_estimators, **kwargs: xgb_evaluate(
                    max_depth=kwargs.get("max_depth", None),
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=kwargs.get("subsample", None),
                    colsample_bytree=kwargs.get("colsample_bytree", None),
                    booster=booster
                ),
                pbounds=param_bounds,
                random_state=42,
                verbose=2
            )
            optimizer.maximize(init_points=5, n_iter=10)

            # Use the best parameters found
            current_params = optimizer.max["params"]
            current_params["n_estimators"] = int(current_params["n_estimators"])
            if "max_depth" in current_params:  # Only convert max_depth if it exists
                current_params["max_depth"] = int(current_params["max_depth"])
            current_params["booster"] = booster

            # Train and evaluate the current booster model
            current_model = xgb.XGBClassifier(**current_params, objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda")
            current_model.fit(cp.asnumpy(X_train_gpu), cp.asnumpy(y_train_gpu))
            y_pred_proba = current_model.predict_proba(cp.asnumpy(X_val_gpu))[:, 1]
            current_roc_auc = roc_auc_score(cp.asnumpy(y_val_gpu), y_pred_proba)

            print(f"Validation ROC AUC for booster {booster}: {current_roc_auc}")

            # Update the best model if the current one is better
            if current_roc_auc > best_roc_auc:
                best_model = current_model
                best_params = current_params
                best_booster = booster
                best_roc_auc = current_roc_auc

        # Log the best model and metrics
        print(f"\nBest booster: {best_booster} with ROC AUC: {best_roc_auc}")
        wandb.init(project=project_name, config=best_params, name=f"{run_name}_best")
        self.plot_best_params(best_params, f"{run_name}_best")

        # Save the best model

        best_model_file_xgb = f"{run_name}_best_model.xgb"
        best_model_file_pkl = f"{run_name}_best_model.pkl"

        # Save in XGBoost's native format
        best_model.save_model(best_model_file_xgb)
        # Save with joblib in .pkl format
        joblib.dump(best_model, best_model_file_pkl)

        # Log both files as artifact
        artifact = wandb.Artifact(name=f"{run_name}_best_model", type="model")
        artifact.add_file(best_model_file_xgb)  # Add .xgb
        artifact.add_file(best_model_file_pkl)  # Add .pkl
        wandb.log_artifact(artifact)

        # Plot and log metrics for the best model
        y_pred_proba = best_model.predict_proba(cp.asnumpy(X_val_gpu))[:, 1]
        self.plot_and_log_metrics(cp.asnumpy(y_val_gpu), y_pred_proba, f"{run_name}_best")

    def plot_and_log_metrics(self, y_val, y_pred_proba, run_name):
        """
        Plots and logs various evaluation metrics.
        
        Args:
            y_val (array-like): True labels for the validation set.
            y_pred_proba (array-like): Predicted probabilities for the validation set.
            run_name (str): The name of the run for naming output files.
        """
        # ROC Curve
        display = RocCurveDisplay.from_predictions(y_val, y_pred_proba)
        display.plot()
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier (AUC = 0.5)')
        plt.title(f"ROC Curve - {run_name}")
        plt.legend()
        wandb.log({"ROC Curve Display": wandb.Image(plt)})
        plt.close()
        
        # Goal Rate
        percentiles = np.percentile(y_pred_proba, np.linspace(0, 100, 11))
        percentile_values = np.linspace(0, 100, len(percentiles))
        goal_rates = [np.mean(y_val[y_pred_proba >= p]) for p in percentiles]
        plt.figure()
        plt.plot(percentile_values, goal_rates, marker='o')
        plt.xlabel("Shot probability model percentile")
        plt.ylabel("Goals / (Shots + Goals)")
        plt.title(f"Goal Rate - {run_name}")
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), labels=[f"{int(x*100)}%" for x in np.linspace(0, 1, 11)])
        plt.legend()
        plt.gca().invert_xaxis()
        wandb.log({"Goal Rate": wandb.Image(plt)})
        plt.close()

        # Cumulative Proportion of Goals
        df_temp = pd.DataFrame({"goal_prob": y_pred_proba, "goal": y_val})
        df_temp['goal_perc_bins'] = pd.qcut(df_temp['goal_prob'], q=100, labels=False, duplicates='drop') + 1
        df_temp = df_temp.sort_values(by="goal_perc_bins", ascending=False)
        df_temp['goal_cumsum'] = df_temp['goal'].cumsum() / df_temp['goal'].sum()

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            x="goal_perc_bins", 
            y="goal_cumsum", 
            data=df_temp, 
            label=run_name,
            linewidth=2.5
        )
        plt.title(f"Cumulative % of Goal - {run_name}")
        plt.legend(loc="lower right")
        plt.xlabel("Shot probability model percentile")
        plt.ylabel("Proportion")
        plt.xlim(left=101, right=-1)  # Reverse x-axis
        plt.ylim(bottom=0, top=1)
        plt.xticks(np.arange(0, 120, 20))
        wandb.log({"Cumulative Goal Rate": wandb.Image(plt)})
        plt.close()

        # Calibration Curve
        plt.figure()
        CalibrationDisplay.from_predictions(y_val, y_pred_proba, n_bins=10, strategy='uniform')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration Curve - {run_name}")
        plt.legend()
        wandb.log({"Calibration Curve": wandb.Image(plt)})
        plt.close()
    
    def plot_best_params(self, best_params, run_name):
        """
        Creates a bar chart of the best parameters.

        Args:
            best_params (dict): A dictionary containing parameter names and their best values.
        """
        # Convert all parameter values to strings to avoid type mismatch issues in WandB
        rows = [[key, str(value)] for key, value in best_params.items()]
        table = wandb.Table(data=rows, columns=["Parameter", "Value"])
        
        # Log the table to WandB
        wandb.log({f"{run_name} Best Parameters": table})


    def perform_feature_selection(self, features, target):
        """
        Applies feature selection to reduce the number of input features.
        
        Args:
            features (list): List of feature column names.
            target (str): The target column name.
        
        Returns:
            list: The selected features.
        """
        X_train = self.train_df[features].astype(float)
        y_train = self.train_df[target].astype(int)

        # Example with SelectKBest
        selector = SelectKBest(score_func=f_classif, k='all')  # You can specify 'k' to limit the number of features
        selector.fit(X_train, y_train)
        feature_scores = selector.scores_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Score': feature_scores})
        feature_importance_df = feature_importance_df.sort_values(by='Score', ascending=False)

        # Log feature selection results to WandB
        wandb.init(project="IFT6758.2024-A03", name="xg_boost_selected_features_optimized")
        wandb_table = wandb.Table(dataframe=feature_importance_df)
        wandb.log({f"Feature Importance": wandb_table})

        # Select the top N features
        selected_features = feature_importance_df['Feature'].head(10).tolist() 
        return selected_features
    
    def run_shap_analysis(self, model, features, top_k=10):
        """
        Uses SHAP values to interpret feature importance in the trained model and extracts top features.

        Args:
            model (xgb.XGBClassifier): The trained XGBoost model.
            features (list): List of feature column names.
            top_k (int): Number of top features to extract.

        Returns:
            list: Top `top_k` features ranked by their average absolute SHAP values.
        """
        explainer = shap.Explainer(model, self.val_df[features])
        X_val = self.val_df[features].astype(float)
        shap_values = explainer(X_val, check_additivity=False)

        wandb.init(project="IFT6758.2024-A03", name="xg_boost_shap_features_best_model")

        # Generate SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_val, show=False)
        shap_plot_path = f"shap_summary.png"
        plt.savefig(shap_plot_path, bbox_inches='tight')
        shap_plot_path = f"shap_summary.png"
        wandb.log({f"Shap Values": wandb.Image(shap_plot_path)})
        plt.close()

        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Mean_SHAP_Value': mean_shap_values
        }).sort_values(by='Mean_SHAP_Value', ascending=False)
        
        # Log feature importance as a table
        
        wandb_table = wandb.Table(dataframe=feature_importance)
        wandb.log({f" Shap - Feature Importance Table": wandb_table})

        # Extract top features
        top_features = feature_importance.head(top_k)['Feature'].tolist()
        print(f"Top {top_k} SHAP features: {top_features}")
        
        return top_features


    def run_experiment(self, name, features, target, optimized=False):
        """
        Runs an experiment to train XGBoost with specified features and either base or optimized parameters.
        
        Args:
            name (str): Run name for WandB logging.
            features (list): List of feature column names.
            target (str): The target column name.
            optimized (bool): Whether to use the optimized XGBoost training method.
        """
        if optimized:
            run_name = f"{name}_optimized"
            self.train_xgboost_optimized(features, target, run_name=run_name)
        else:
            run_name = f"{name}_base"
            self.train_xgboost_base(features, target, run_name=run_name)

def main():
    root_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
    data_dir = os.path.join(root_directory, 'data')  # Adjusted to correctly reference data_dir
    
    trainer = XGBoostModelTrainer(data_dir)
   
    features_base = ['shotDistance', 'shotAngle']
    features_II = ['timeInPeriod', 'period', 'xCoord', 'yCoord', 'shotDistance', 'shotAngle', 'lastEventType', 'lastEventXCoord','lastEventYCoord','timeElapsedSinceLastEvent', 'distanceFromLastEvent', 'rebound', 'changeInShotAngle', 'speed']
    all_features = [col for col in trainer.df.columns if col != 'result' and col != 'coordinates' and col != 'eventType']
    
    target = 'result'

    #UNCOMMENT ONLY 1 CONFIG
    #Base XGBoost: Line 392
    #Optimized XGBoost: Line 396
    #Feature Selection + Optimization: Line 400-401
    #SHAP Analysis: Line 405-412


    # Run base experiment
    #trainer.run_experiment(name='xg_boost_distance_angle', features=features_base, target=target, optimized=False)

    # Run optimized experiment
    #trainer.run_experiment(name='xg_boost_features_II', features=features_II, target=target, optimized=True)

    #Run with feature selection + optimization
    #selected_features = trainer.perform_feature_selection(all_features, target)
    #trainer.run_experiment(name='xg_boost_selected_features_optimized', features=selected_features, target=target, optimized=True)

    # Train model
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
    X_train = trainer.train_df[all_features].astype(float)
    y_train = trainer.train_df[target].astype(int)
    model.fit(X_train, y_train)

    #Shap
    top_shap_features = trainer.run_shap_analysis(model=model, features=all_features, top_k=10)
    trainer.run_experiment(name='xg_boost_shap_features', features=top_shap_features, target=target, optimized=True)

main()