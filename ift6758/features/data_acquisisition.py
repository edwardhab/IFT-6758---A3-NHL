import pandas as pd
import requests
import json
import os
from pandasgui import show

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'nhl_game_data.json')

def get_nhl_game_data(start_season: int, final_season: int):
    """
    Retrieves all NHL game plays data from NHL's API for each game in the given seasons,
    and stores them in a json file

    Args:
        start_season (int): The first year of the season to retrieve, i.e. for the 2016-17 season you'd put in 2016.
        final_season (int): The first year of the last season to retrieve, i.e. for the 2023-2024 season you'd put in 2023.
    
    Returns: void
    """
    all_data = {}
    # Iterate through each season's games
    for season in range(start_season, final_season + 1):
        if season == 2016:
            num_games = 1230
        elif season in [2017, 2018]:
            num_games = 1271
        elif season == 2019:
            num_games = 1082
        elif season == 2020:
            num_games = 868
        else:
            num_games = 1312
        
        # Iterate through each game in the regular season
        for game_num in range(1, num_games + 1):

            
            #Query to NHL API
            game_id = f'{season}02{game_num:04d}'  # Construct the game ID
            url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play' 
            print(f"Retrieving data from '{url}'...")
            response = requests.get(url)
            game_data = response.json()
            all_data[game_id] = game_data

        with open(file_path, 'w') as json_file:
            json.dump(all_data, json_file, indent=4)

    print(f"Data saved to {file_path}")

def get_player_name(player_id: int):
    """
    Retrieves the name of a player given their ID.

    Args:
        player_id (int): The ID of the player to retrieve.
    
    Returns:
        str: The name of the player.
    """
    if player_id == None:
        return None
    
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    response = requests.get(url)
    player_data = response.json()
    first_name = player_data.get('firstName', {}).get('default')
    last_name = player_data.get('lastName', {}).get('default')
    full_name = f"{first_name} {last_name}"
    return full_name

def parse__nhl_game_data():
    """
    Parses the NHL game data from a JSON file and filters for shot-related events
    (goals, shots on goal, missed shots).

    Args:
        file_path (str): Path to the JSON file containing game data.

    Returns:
        pd.DataFrame: A DataFrame containing shot-related events for all games.
    """

    all_shot_events_df = pd.DataFrame()
    with open(file_path, 'r') as json_file:
        game_data = json.load(json_file)
    #Parsing data
    #Declare data holders
    for game_id, game_details in game_data.items():
        print(f"Parsing game {game_id}")
        all_plays = game_details.get('plays',[])
        shot_events = []
        
        for play in all_plays:
            #Filter for Goals, Shot on goal, Missed shot
            event_type = play.get('typeCode')
            if event_type in [505, 506, 507]:
                shooter_id = play.get('details', {}).get('scoringPlayerId') if event_type == 505 else play.get('details', {}).get('shootingPlayerId')
                goalie_id = play.get('details', {}).get('goalieInNetId')
                #Interpret situation code here: https://gitlab.com/dword4/nhlapi/-/issues/112
                situation_code = play.get('situationCode')
                shot_details = {
                    'season': game_details.get('season'),
                    'gameId': game_id,
                    'eventId': play.get('eventId'),
                    'period': play.get('periodDescriptor', {}).get('number'),
                    'timeInPeriod': play.get('timeInPeriod'),
                    'eventType': play.get('typeDescKey'),
                    'teamId': play.get('details', {}).get('eventOwnerTeamId'),
                    #Call player API to get player/goalie names
                    #'shooter': get_player_name(shooter_id),
                    'shooter': shooter_id,
                    #'goalie': get_player_name(goalie_id),
                    'goalie': goalie_id,
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

# Run this function once 
# get_nhl_game_data(2016, 2023)
df = parse__nhl_game_data()

# Display the DataFrame in a grid layout using pandasgui
if not df.empty:
    show(df)
else:
    print("No shot data available.")
