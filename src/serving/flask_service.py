from flask import Flask, request, jsonify
import logging
import os
import joblib
import wandb
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Environment variable for log file
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

# Global variable for the loaded model
current_model = None
current_model_name = None

@app.before_first_request
def initialize_application():
    """
    Initializes logging, loads the default model, and preprocesses the data.
    """
    global current_model, current_model_name

    # Setup logging
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
    # Load the default model if specified
    default_model_path = os.path.join(os.getcwd(), 'artifacts', 'logreg_comb-v0', 'logreg_comb.pkl')
    container_default_model_path = os.path.join(os.getcwd(), 'serving', 'artifacts', 'logreg_comb-v0', 'logreg_comb.pkl')
    if default_model_path and (os.path.exists(default_model_path) or os.path.exists(container_default_model_path)):
        try:
            if(os.path.exists(default_model_path)):
                current_model = joblib.load(default_model_path)
            else:
                current_model = joblib.load(container_default_model_path)
            current_model_name = 'logreg_comb:latest'
            app.logger.info('Successfully loaded default model: logreg_comb:latest')
        except Exception as e:
            app.logger.error(f"Error loading default model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint: Accepts JSON payload or reads data from a CSV file and returns predictions.
    """
    if current_model is None:
        return jsonify({'error': 'No model loaded. Please load a model first using /download_registry_model.'}), 400

    try:
        # Check if JSON payload is provided
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'No valid input provided. Please provide a JSON payload.'}), 400

        # Predict probabilities
        predictions = current_model.predict_proba(df.to_numpy())[:, 1]
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
def logs():
    """
    Logs endpoint: Displays application logs.
    """
    try:
        with open(LOG_FILE, 'r') as log_file:
            logs = log_file.readlines()
        return "<br>".join(logs)
    except Exception as e:
        app.logger.error(f"Error reading logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_registry_model', methods=['POST'])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The wandDB API key should be retrieved from the ${wandDB} environment variable.

    Request -> json with the schema:

        {
            "artifact_name": "logreg_dist:latest" (Include the tag to specify which version you want to use)
        }
    
    """
    global current_model, current_model_name
    try:
        # Parse input JSON
        data = request.get_json()
        artifact_name = data['artifact_name']  # e.g., "logreg_dist:latest"
        workspace_name = data['workspace_name']  # e.g., "IFT6758.2024-A03"

        # Check if the requested model is already loaded in memory
        if current_model_name == artifact_name:
            app.logger.info(f"Model {artifact_name} is already loaded.")
            return jsonify({'message': f'Model {artifact_name} is already loaded.'})

        app.logger.info(f"Migrating to model {artifact_name}...")
        # Check if the model artifact is already downloaded
        artifact_base_name = artifact_name.split(':')[0]  # Extract base name (e.g., "logreg_dist")
        artifacts_path = os.path.join(os.getcwd(), 'artifacts')
        model_file = None

        for root, dirs, files in os.walk(artifacts_path):
            for file in files:
                if file == f"{artifact_base_name}.pkl":
                    model_file = os.path.join(root, file)
                    app.logger.info(f"Model {artifact_name} already exists locally at {model_file}.")
                    break

        if not model_file:
            # Initialize WandB API and download the artifact
            app.logger.info(f"Downloading model artifact {artifact_name}...")
            api = wandb.Api()
            project_name = os.environ.get("WANDB_PROJECT", workspace_name)
            entity_name = os.environ.get("WANDB_ENTITY", "michel-wilfred-essono-university-of-montreal")  
            artifact = api.artifact(f"{entity_name}/{project_name}/{artifact_name}")
            model_path = artifact.download()

            # Locate the model file in the newly downloaded artifact
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file == f"{artifact_base_name}.pkl":
                        model_file = os.path.join(root, file)
                        break

            if not model_file:
                app.logger.error(f"No file named {artifact_base_name}.pkl found in the downloaded artifact.")
                raise FileNotFoundError(f"No file named {artifact_base_name}.pkl found in the downloaded artifact.")

        # Load the model
        with open(model_file, 'rb') as file:
            current_model = joblib.load(file)  # Use joblib to load the model
        current_model_name = artifact_name

        app.logger.info(f"Loaded model {artifact_name} successfully.")
        return jsonify({'message': f'Model {artifact_name} loaded successfully.'})
    
    except Exception as e:
        app.logger.error(f"Error downloading/loading model artifact: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_application()
    app.run(host='0.0.0.0', port=5000)
