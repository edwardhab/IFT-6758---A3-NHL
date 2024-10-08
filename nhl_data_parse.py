import requests
import pandas as pd
import json
import os
from pandasgui import show


class NHLDataParser:
    def __init__(self, nhl_games_file_path):
        self.nhl_games_file_path = nhl_games_file_path
        self.player_names = {}

    def get_player_name(self, player_id: int):
        if player_id is None:
            return None
        
        if str(player_id) in self.player_names:
            return self.player_names[str(player_id)]
        
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url)
        
        if response.status_code == 200:
            player_data = response.json()
            first_name = player_data.get('firstName', {}).get('default', '')
            last_name = player_data.get('lastName', {}).get('default', '')
            full_name = f"{first_name} {last_name}".strip()

            self.player_names[str(player_id)] = full_name
            return full_name
        else:
            return None

    def get_season_data(self, season, stage):
        """
        Parcours les évènements d'une saison donnée à partir du fichier JSON 'all_games_data.json'.
        
        Args:
            season (str): La saison à analyser (par exemple '2016' pour la saison 2016-17).
            stage (str): 'season' pour la saison régulière, 'playoffs' pour les séries éliminatoires.
        
        Returns:
            list: Liste de tous les événements trouvés pour la saison donnée.
        """
        # Charger le fichier JSON contenant tous les matchs
        all_games_file = os.path.join(self.nhl_games_file_path, 'all_games_data.json')
        
        if not os.path.exists(all_games_file):
            print("Fichier 'all_games_data.json' non trouvé.")
            return []

        with open(all_games_file, 'r') as json_file:
            all_games_data = json.load(json_file)

        # Filtrer les matchs de la saison donnée
        season_games = []
        for game_id, game_details in all_games_data.items():
            game_season = game_details.get('season')
            game_type = game_details.get('gameType')  # Type de match (saison ou playoffs)
            
            if game_season == season and (stage == 'season' and game_type == '02') or (stage == 'playoffs' and game_type == '03'):
                # print(f"Match trouvé pour la saison {season}, game ID: {game_id}")
                season_games.append(game_details)
            # else:
                # print(f"Match ignoré pour game ID: {game_id}, season {game_season}, game type {game_type}")

        return season_games
    
    def parse_season_events(self, season, stage):
        """
        Analyse et retourne les événements des matchs d'une saison donnée.
        
        Args:
            season (str): La saison à analyser (par exemple '2016' pour la saison 2016-17).
            stage (str): 'season' pour la saison régulière, 'playoffs' pour les séries éliminatoires.
        
        Returns:
            list: Liste d'événements pour la saison donnée.
        """
        season_games = self.get_season_data(season, stage)
        
        shot_events = []
        for game_details in season_games:
            all_plays = game_details.get('plays', [])
            
            for play in all_plays:
                event_type = play.get('typeCode')
                if event_type in [505, 506]:  # Filtrer pour les tirs et buts
                    shooter_id = play.get('details', {}).get('scoringPlayerId') if event_type == 505 else play.get('details', {}).get('shootingPlayerId')
                    goalie_id = play.get('details', {}).get('goalieInNetId')
                    situation_code = play.get('situationCode')

                    shot_details = {
                        'season': game_details.get('season'),
                        'gameId': game_details.get('gameId'),
                        'eventId': play.get('eventId'),
                        'period': play.get('periodDescriptor', {}).get('number'),
                        'timeInPeriod': play.get('timeInPeriod'),
                        'eventType': play.get('typeDescKey'),
                        'teamId': play.get('details', {}).get('eventOwnerTeamId'),
                        'shooter': self.get_player_name(shooter_id),
                        'goalie': self.get_player_name(goalie_id),
                        'shotType': play.get('details', {}).get('shotType'),
                        'emptyNetAway': False if int(situation_code[0]) == 1 else True,
                        'emptyNetHome': False if int(situation_code[3]) == 1 else True,
                        'powerplayHome': True if int(situation_code[2]) > int(situation_code[1]) else False,
                        'powerplayAway': True if int(situation_code[1]) > int(situation_code[2]) else False,
                        'coordinates': (play.get('details', {}).get('xCoord'), play.get('details', {}).get('yCoord')),
                        'result': 'goal' if event_type == 505 else 'no goal'
                    }
                    shot_events.append(shot_details)

        return shot_events


nhl_games_file_path = 'C:\\Users\\Admin\\Documents\\IFT6758-A3\\IFT-6758---A3-NHL\\nhl_data'
parser = NHLDataParser(nhl_games_file_path)
print(parser.get_season_data('2016', 'season')[10])

