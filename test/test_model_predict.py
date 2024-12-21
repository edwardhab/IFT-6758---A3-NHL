import requests
import pandas as pd
import json
import os

# Paths to your data files
module_path = os.path.abspath(os.path.join('..', '..', 'data', 'enhanced_parsed_shot_events.csv'))
parsed_module_path = os.path.abspath(os.path.join('..', '..', 'data', 'test_predict_logreg_comb.csv'))

# Read the input data
df = pd.read_csv(module_path)

# Ensure only numerical columns are passed to the model
columns_to_extract = [
    'shotDistance',  # Distance to goal
    'shotAngle',     # Angle to goal
]

# Filter the required columns
filtered_df = df[columns_to_extract]

# Check for and handle missing or non-numeric values
filtered_df = filtered_df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, setting invalid values to NaN
filtered_df = filtered_df.dropna()  # Drop rows with NaN values

# Save the cleaned data to a new CSV for verification
filtered_df.to_csv(parsed_module_path, index=False)

# Convert the filtered DataFrame to JSON format
json_data = json.loads(filtered_df.to_json(orient="records"))

# URL of the predict endpoint
url = "http://127.0.0.1:5000/predict"  # Replace <PORT> if your app runs on a different port

# Send the POST request to the /predict endpoint
response = requests.post(url, json=json_data)

# Print the response
print(response.status_code)
print(response.json())
