import json

def extract_unique_event_types(file_path):
    # Charger le fichier JSON
    with open(file_path, 'r') as f:
        all_games_data = json.load(f)
    
    # Créer un ensemble pour stocker les valeurs uniques
    unique_event_types = set()

    # Parcourir chaque match et ses événements
    for game_id, game_data in all_games_data.items():
        # Vérifier si la clé "plays" existe dans le match
        if 'plays' in game_data:
            for event in game_data['plays']:
                # Ajouter "typeDescKey" dans l'ensemble des types uniques
                unique_event_types.add(event['typeDescKey'])
    
    # Convertir l'ensemble en liste et retourner
    return list(unique_event_types)

# Appeler la fonction en passant le chemin du fichier
file_path = nhl_games_file_path = 'C:\\Users\\Admin\\Documents\\IFT6758-A3\\IFT-6758---A3-NHL\\nhl_data\\all_games_data.json'
unique_event_types = extract_unique_event_types(file_path)

# Afficher les types d'événements uniques
print(unique_event_types)

