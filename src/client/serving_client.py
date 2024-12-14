import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=['distance', 'angle']):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        self.features = features


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        try:
            #Filter the required columns
            filtered_df = X[self.features]
            
            filtered_df = filtered_df.apply(pd.to_numeric, errors="coerce")
            if filtered_df.empty:
                logger.info("Data Frame for prediction is empty")
            #Convert input to json
            json_data = json.loads(filtered_df.to_json(orient="records"))

            #Send the POST request to the /predict endpoint
            url = f"{self.base_url}/predict"
            response = requests.post(url, json=json_data)
            predictions = response.json().get("predictions", [])
            # Handle response errors
            if response.status_code != 200:
                logger.error(f"Prediction service error: {response.status_code} {response.text}")
                response.raise_for_status()
            #Format response into dataframe
            prediction_df = pd.DataFrame({"prediction": predictions}, index=filtered_df.index)
            return prediction_df
        
        except requests.RequestException as e:
            logger.error(f"Error fetching prediction from server: {e}")
            raise
        

    def logs(self) -> dict:
        """Get server logs"""
        try:
            #Construct the URL
            url = f"{self.base_url}/logs"
            #Send the GET request to the /logs endpoint
            response = requests.get(url)
            return response
        except requests.RequestException as e:
            logger.error(f"Error fetching logs from server: {e}")
            raise

    def download_registry_model(self, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        try: 
            #Construct the URL
            url = f"{self.base_url}/download_registry_model"
            if(model == 'logreg_comb'):
                self.features = ['distance', 'angle']
            else:
                self.features = ['distance']
                
            #Prepare the payload
            artifact_name = f"{model}:{version}"
            payload = {"artifact_name": artifact_name}

            #Send the POST request to the /download_registry_model endpoint
            response = requests.post(url, json=payload)

            #Handle response errors
            if response.status_code != 200:
                logger.error(f"Failed to download the model: {response.status_code} {response.text}")
                response.raise_for_status()
            
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Error communicating with the service: {e}")
            raise
        