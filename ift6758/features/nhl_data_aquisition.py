import os 
import requests
import json
import string

class LNHDataDownload:
    def __init__(self,start_year,end_year,save_dir='data'):
        # Récupérer le répertoire depuis la variable d'environnement ou utiliser la valeur par défaut
        self.save_dir = os.environ.get('NHL_DATA_DIR', 'data') 
        self.start_year = start_year
        self.end_year = end_year
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def get_end_game(self,season):
        
        self.season = season

        url = 'https://api.nhle.com/stats/rest/en/season'
        response = requests.get(url).json()
        
        for item in response['data']:
            id_value = item.get('id', '')
            total_regular_season_games = item.get('totalRegularSeasonGames', 'N/A')
            id_string = str(id_value)
            if id_string.startswith(self.season):
                return int(total_regular_season_games)
        return 0

    def get_game_id(self, season, game_type, game_number):
        """
        Génère le GAME_ID pour un match donné.
        
        :param season: La première année de la saison (par ex. 2016 pour la saison 2016-17).
        :param game_type: '01' pour la saison régulière, '02' pour les séries éliminatoires.
        :param game_number: Numéro du match dans la saison/séries (1-1271 pour saison régulière).
        :return: Le GAME_ID formé.
        """
        return f"{season}{game_type}{str(game_number).zfill(4)}"
    
    def get_playoff_gameId(self,season,stage):
   
      next_year = int(season) + 1
      season = int(f"{season}{next_year}")
      url =f'https://api-web.nhle.com/v1/schedule/playoff-series/{season}/{stage}'
      
     
      response = requests.get(url)
      if response.status_code != 200:
                
        return None  # Move to the next season if this URL doesn't exist
            
    # If the URL exists, process the data
      data = response.json()
      
      game_id = [game['id'] for game in data['games']]
      return game_id

    def download_game_data(self, game_id):
        """
        Télécharge les données d'un match donné et les sauvegarde localement.
        
        :param game_id: L'identifiant du match (formaté selon la méthode get_game_id).
        :return: Le contenu JSON des données du match.
        """
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        local_file = os.path.join(self.save_dir, f"{game_id}.json")
        
        # Vérifier si les données sont déjà en cache
        if os.path.exists(local_file):
            with open(local_file, 'r') as f:
                print(f"Chargement des données depuis le cache pour le match {game_id}")
                return json.load(f)
        
        # Sinon, télécharger depuis l'API
        response = requests.get(url)
        
        if response.status_code == 200:
            game_data = response.json()
            # Sauvegarder les données localement
            with open(local_file, 'w') as f:
                json.dump(game_data, f)
            print(f"Données téléchargées et sauvegardées pour le match {game_id}")
            return game_data
        else:
            print(f"Échec du téléchargement des données pour le match {game_id}")
            return None

    def download_Regseason_data(self,start_game =1):
        """
        Télécharge les données de toute une saison (ou série).
        
        :param season_start: Année de début de la saison (ex: 2016 pour 2016-17).
        :param game_type: '01' pour la saison régulière, '02' pour les séries éliminatoires.
        :param start_game: Numéro du premier match à télécharger (défaut : 1).
        :param end_game: Numéro du dernier match à télécharger (défaut : 1271 pour la saison régulière).
        """
        
        game_types = ['02','03']
       
        for game_type in game_types:
           for season in range(self.start_year,self.end_year+1):
              if game_type == '02':
                 for game_number in range(start_game, self.get_end_game(str(season))+ 1):
                     game_id = self.get_game_id(season, game_type, game_number)
                     self.download_game_data(game_id)
              else:
                  for stage in string.ascii_lowercase:
                    playoff_gamesId = self.get_playoff_gameId(season, stage)
                    if playoff_gamesId:  # Check if playoff_gamesId is not None
                        for playoff_gameId in playoff_gamesId:
                            self.download_game_data(playoff_gameId)
                    else:
                        print(f"No playoff games found for season {season}, stage {stage}")
                          

# Utilisation
downloader = LNHDataDownload(start_year=2016, end_year=2023)
downloader.download_Regseason_data(start_game=1)





