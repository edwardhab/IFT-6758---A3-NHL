import os 
import requests
import json
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

class LNHDataDownload:
    def __init__(self, start_year, end_year, save_dir='data'):
        # Récupérer le répertoire depuis la variable d'environnement ou utiliser la valeur par défaut
        self.save_dir = os.environ.get('NHL_DATA_DIR', 'data') 
        self.start_year = start_year
        self.end_year = end_year
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def get_end_game(self, season):
        url = 'https://api.nhle.com/stats/rest/en/season'
        response = requests.get(url).json()
        
        for item in response['data']:
            id_value = item.get('id', '')
            total_regular_season_games = item.get('totalRegularSeasonGames', 'N/A')
            id_string = str(id_value)
            if id_string.startswith(season):
                return int(total_regular_season_games)
        return 0

    def get_game_id(self, season, game_type, game_number):
        return f"{season}{game_type}{str(game_number).zfill(4)}"
    
    def get_playoff_gameId(self, season, stage):
        next_year = int(season) + 1
        season_str = f"{season}{next_year}"
        url = f'https://api-web.nhle.com/v1/schedule/playoff-series/{season_str}/{stage}'
        response = requests.get(url)
        if response.status_code != 200:
            return None
        
        data = response.json()
        game_id = [game['id'] for game in data['games']]
        return game_id

    def download_game_data(self, game_id):
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        local_file = os.path.join(self.save_dir, f"{game_id}.json")
        
        if os.path.exists(local_file):
            with open(local_file, 'r') as json_file:
                print(f"Chargement des données depuis le cache pour le match {game_id}")
                return json.load(json_file)
        
        response = requests.get(url)
        if response.status_code == 200:
            game_data = response.json()
            print(f"Données téléchargées pour le match {game_id}")
            return game_data
        else:
            print(f"Échec du téléchargement des données pour le match {game_id}")
            return None

    def download_games_parallel(self, game_ids):
        all_games_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.download_game_data, game_id): game_id for game_id in game_ids}
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    game_data = future.result()
                    if game_data:
                        all_games_data[game_id] = game_data
                        local_file = os.path.join(self.save_dir, f"{game_id}.json")
                        # Sauvegarder les données téléchargées
                        with open(local_file, 'w') as json_file:
                            json.dump(game_data, json_file, indent=4)
                except Exception as e:
                    print(f"Erreur pour le match {game_id}: {e}")
        
        return all_games_data
    
    def download_Regseason_data(self, start_game=1):
        all_games_data = {}
        game_types = ['02', '03']
        
        for game_type in game_types:
            for season in range(self.start_year, self.end_year + 1):
                if game_type == '02':
                    game_ids = [
                        self.get_game_id(season, game_type, game_number)
                        for game_number in range(start_game, self.get_end_game(str(season)) + 1)
                    ]
                    # Exécution en parallèle pour les matchs de la saison régulière
                    regseason_data = self.download_games_parallel(game_ids)
                    all_games_data.update(regseason_data)
                else:
                    for stage in string.ascii_lowercase:
                        playoff_gamesId = self.get_playoff_gameId(season, stage)
                        if playoff_gamesId:
                            # Exécution en parallèle pour les matchs des playoffs
                            playoff_data = self.download_games_parallel(playoff_gamesId)
                            all_games_data.update(playoff_data)
                        else:
                            print(f"Aucun match de playoffs trouvé pour la saison {season}, étape {stage}")

        # Sauvegarder toutes les données dans un seul fichier JSON
        with open(os.path.join(self.save_dir, 'all_games_data.json'), 'w') as json_file:
            json.dump(all_games_data, json_file, indent=4)

# Utilisation
downloader = LNHDataDownload(start_year=2016, end_year=2023)
downloader.download_Regseason_data(start_game=1)
