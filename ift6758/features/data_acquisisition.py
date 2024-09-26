import os
import requests
import json
import pandas as pd
from pandasgui import show

class NHLDataDownloader:
    def __init__(self, start_season, final_season, data_dir=None):
        """
        Initialize the class with start and final seasons.

        Args:
            start_season (int): The first year of the season to retrieve.
            final_season (int): The first year of the last season to retrieve.
        """
        self.start_season = start_season
        self.final_season = final_season
        self.data_dir = data_dir if data_dir else os.getcwd()
        self.nhl_games_file_path = os.path.join(self.data_dir, 'nhl_game_data.json')
        self.nhl_players_file_path = os.path.join(self.data_dir, 'nhl_player_data.json')

        # Load existing player names from file if it exists
        if os.path.exists(self.nhl_players_file_path):
            with open(self.nhl_players_file_path, 'r') as players_file:
                self.player_names = json.load(players_file)
            print(f"Loaded existing player data from {self.nhl_players_file_path}")
        else:
            self.player_names = {}

    def get_nhl_game_data(self):
        """
        Retrieves all NHL game plays data from NHL's API for each game in the given seasons
        and stores them in a json file.
        """
        all_data = {}

        for season in range(self.start_season, self.final_season + 1):
            if season == 2016:
                num_regular_season_games = 1230
            elif season in [2017, 2018]:
                num_regular_season_games = 1271
            elif season == 2019:
                num_regular_season_games = 1082
            elif season == 2020:
                num_regular_season_games = 868
            else:
                num_regular_season_games = 1312

            # Regular Season Games
            for game_num in range(1, num_regular_season_games + 1):
                game_id = f'{season}02{game_num:04d}'
                url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
                print(f"Retrieving data from '{url}'...")
                response = requests.get(url)
                game_data = response.json()
                all_data[game_id] = game_data

            # Playoff Games
            rounds = 4
            matchups_per_round = [8, 4, 2, 1]
            max_games_per_series = 7
            for round_num in range(1, rounds + 1):
                for matchup_num in range(1, matchups_per_round[round_num - 1] + 1):
                    for game_num in range(1, max_games_per_series + 1):
                        game_id = f'{season}030{round_num}{matchup_num}{game_num}'
                        url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
                        print(f"Retrieving playoff data from '{url}'...")
                        response = requests.get(url)
                        if response.status_code == 200:
                            game_data = response.json()
                            all_data[game_id] = game_data
                        else:
                            print(f"No data found for playoff game {game_id}. Stopping the series.")
                            break

        # Save to JSON
        with open(self.nhl_games_file_path, 'w') as json_file:
            json.dump(all_data, json_file, indent=4)

        print(f"Data saved to {self.nhl_games_file_path}")

    def save_player_names(self):
        """
        Saves the player names dictionary to a JSON file.
        """
        with open(self.nhl_players_file_path, 'w') as players_file:
            json.dump(self.player_names, players_file, indent=4)
        print(f"Player names saved to {self.nhl_players_file_path}")

    def get_player_name(self, player_id: int):
        """
        Retrieves the name of a player given their ID.

        Args:
            player_id (int): The ID of the player to retrieve.

        Returns:
            str: The name of the player.
        """
        if player_id is None:
            return None

        if str(player_id) in self.player_names:
            return self.player_names[str(player_id)]

        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url)
        player_data = response.json()
        first_name = player_data.get('firstName', {}).get('default')
        last_name = player_data.get('lastName', {}).get('default')
        full_name = f"{first_name} {last_name}"

        self.player_names[str(player_id)] = full_name
        return full_name

    def parse_nhl_game_data(self):
        """
        Parses the NHL game data from a JSON file and filters for shot-related events.

        Returns:
            pd.DataFrame: A DataFrame containing shot-related events for all games.
        """
        all_shot_events_df = pd.DataFrame()

        with open(self.nhl_games_file_path, 'r') as json_file:
            game_data = json.load(json_file)

        # Parsing data
        for game_id, game_details in game_data.items():
            print(f"Parsing game {game_id}")
            all_plays = game_details.get('plays', [])
            shot_events = []

            for play in all_plays:
                event_type = play.get('typeCode')
                if event_type in [505, 506]:
                    shooter_id = play.get('details', {}).get('scoringPlayerId') if event_type == 505 else play.get('details', {}).get('shootingPlayerId')
                    goalie_id = play.get('details', {}).get('goalieInNetId')
                    situation_code = play.get('situationCode')
                    shot_details = {
                        'season': game_details.get('season'),
                        'gameId': game_id,
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

            if shot_events:
                shot_events_df = pd.DataFrame(shot_events)
                all_shot_events_df = pd.concat([all_shot_events_df, shot_events_df], ignore_index=True)

        return all_shot_events_df

# # Example usage:

# downloader = NHLDataDownloader(start_season=2016, final_season=2023)

# if not os.path.exists(downloader.nhl_games_file_path):
#     downloader.get_nhl_game_data()

# df = downloader.parse_nhl_game_data()
# downloader.save_player_names()

# # Display the DataFrame in a grid layout using pandasgui
# if not df.empty:
#     show(df)
# else:
#     print("No shot data available.")
