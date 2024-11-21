import os
import pandas as pd
import numpy as np
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
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Save and log model
        model_file = f"{run_name}_model.xgb"
        model.save_model(model_file)
        artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)

        # Make predictions and evaluate
        y_pred_proba = model.predict(dval)
        self.plot_and_log_metrics(y_val, y_pred_proba, run_name)
        self.plot_best_params(params, run_name)
        # SHAP Analysis
        #self.run_shap_analysis(model, features)

    def train_xgboost_optimized(self, features, target, project_name="IFT6758.2024-A03", run_name="XGBoost_Optimized"):
        """
        Trains an XGBoost classifier using specified features with Bayesian optimization for hyperparameter tuning and logs metrics.
        """
        # Prepare data
        X_train = self.train_df[features].astype(float)
        y_train = self.train_df[target].astype(int)
        X_val = self.val_df[features].astype(float)
        y_val = self.val_df[target].astype(int)

        # Define the function to optimize
        def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": int(max_depth),
                "learning_rate": learning_rate,
                "n_estimators": int(n_estimators),
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "use_label_encoder": False
            }
            xgb_model = xgb.XGBClassifier(**params)
            # Cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='roc_auc')
            return np.mean(cv_scores)  # Return mean ROC AUC across folds

        # Define the parameter bounds for optimization
        param_bounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 300),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0)
        }

        # Perform Bayesian optimization
        optimizer = BayesianOptimization(
            f=xgb_evaluate,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=3, n_iter=15)

        # Use the best parameters found
        best_params = optimizer.max["params"]
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])

        # Print the best parameters
        print("Best parameters found by Bayesian Optimization:")
        print(best_params)
        
        # Log the best parameters to WandB
        wandb.init(project=project_name, config=best_params, name=run_name)
        self.plot_best_params(best_params, run_name)

        # Train the best model
        best_model = xgb.XGBClassifier(**best_params, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
        best_model.fit(X_train, y_train)

        # Save and log the best model as an artifact
        model_file = f"{run_name}_model.xgb"
        best_model.save_model(model_file)
        artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)

        # Evaluate on the validation set
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation ROC AUC: {roc_auc}")
        wandb.log({"ROC AUC": roc_auc})
        self.plot_and_log_metrics(y_val, y_pred_proba, run_name)


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
        # Extract parameter names and values
        rows = [[key, value] for key, value in best_params.items()]
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

        wandb.init(project="IFT6758.2024-A03", name="xg_boost_shap_features")

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