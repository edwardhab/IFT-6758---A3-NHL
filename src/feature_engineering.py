import pandas as pd
import os
import json
import math

class ShotEventFeatureEngineer:
    def __init__(self, data_dir=None):
        """
        Initializes the feature engineering class.
        """
        root_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.data_dir = data_dir if data_dir else os.path.join(root_directory, 'data')
        self.nhl_games_file_path = os.path.join(self.data_dir, 'nhl_game_data.json')
        self.parsed_data_path = os.path.join(self.data_dir, 'parsed_shot_events.csv')
        self.df = pd.read_csv(self.parsed_data_path)
        # Load game data
        if os.path.exists(self.nhl_games_file_path):
            with open(self.nhl_games_file_path, 'r') as json_file:
                self.game_data = json.load(json_file)
        else:
            print(f"Game data file not found at: {self.nhl_games_file_path}")
            self.game_data = {}

    def add_team_ids(self):
        """
        Adds home and away team IDs to the DataFrame by extracting them from the game data JSON file.
        """
        home_ids = []
        away_ids = []
        self.df['gameId'] = self.df['gameId'].astype(str)
        
        for _, row in self.df.iterrows():
            game_id = row['gameId']
            game_details = self.game_data.get(game_id, {})
            home_team_id = game_details.get('homeTeam', {}).get('id')
            away_team_id = game_details.get('awayTeam', {}).get('id')
            
            # Append team IDs or None if not available
            home_ids.append(int(home_team_id) if home_team_id is not None else None)
            away_ids.append(int(away_team_id) if away_team_id is not None else None)

        # Add the home and away team IDs as new columns in the DataFrame
        self.df['homeTeamId'] = home_ids
        self.df['awayTeamId'] = away_ids

        # Ensure columns are of integer type, fill NaN with a default value if necessary
        self.df['homeTeamId'] = self.df['homeTeamId'].fillna(0).astype(int)
        self.df['awayTeamId'] = self.df['awayTeamId'].fillna(0).astype(int)

    def add_empty_net_goal_column(self):
        """
        Adds a column to the DataFrame indicating if a goal was scored on an empty net.
        """
        def is_empty_net_goal(row):
            if row['eventType'] == 'goal':  # Check if the event is a goal
                if row['emptyNetHome'] and row['teamId'] != row['homeTeamId']:
                    # Goal scored against home team (home net is empty)
                    return 1
                elif row['emptyNetAway'] and row['teamId'] != row['awayTeamId']:
                    # Goal scored against away team (away net is empty)
                    return 1
            return 0

        # Apply the function to each row and create a new column 'emptyNetGoal'
        self.df['emptyNetGoal'] = self.df.apply(is_empty_net_goal, axis=1)

    def determine_offensive_side(self):
        """
        Determines the offensive side (left or right) for the home and away teams based on the first non-neutral
        shot of the first period, considering which team took the shot.
        """
        # Limit to the first 5 unique games for testing
        limited_game_ids = self.df['gameId'].unique()[:5]

        for game_id in limited_game_ids:
            game_shots = self.df[self.df['gameId'] == game_id]
            home_team_side = 'unknown'
            away_team_side = 'unknown'

            # Fetch the first non-neutral shot from the first period
            for _, shot in game_shots.iterrows():
                if shot['period'] == 1:
                    # Fetch play details from game data
                    game_details = self.game_data.get(str(game_id), {})
                    all_plays = game_details.get('plays', [])
                    play = next((p for p in all_plays if p.get('eventId') == shot['eventId']), None)

                    if play:
                        zone_code = play.get('details', {}).get('zoneCode')
                        x_coord = play.get('details', {}).get('xCoord')
                        shooting_team_id = shot['teamId']

                        if zone_code and zone_code != 'N' and x_coord is not None:
                            # Determine sides based on which team is shooting
                            if shooting_team_id == shot['homeTeamId']:
                                # Home team shot
                                if x_coord < 0 and zone_code == 'D':
                                    home_team_side = 'right'
                                    away_team_side = 'left'
                                elif x_coord < 0 and zone_code == 'O':
                                    home_team_side = 'left'
                                    away_team_side = 'right'
                                elif x_coord > 0 and zone_code == 'D':
                                    home_team_side = 'left'
                                    away_team_side = 'right'
                                elif x_coord > 0 and zone_code == 'O':
                                    home_team_side = 'right'
                                    away_team_side = 'left'
                            elif shooting_team_id == shot['awayTeamId']:
                                # Away team shot
                                if x_coord < 0 and zone_code == 'D':
                                    away_team_side = 'right'
                                    home_team_side = 'left'
                                elif x_coord < 0 and zone_code == 'O':
                                    away_team_side = 'left'
                                    home_team_side = 'right'
                                elif x_coord > 0 and zone_code == 'D':
                                    away_team_side = 'left'
                                    home_team_side = 'right'
                                elif x_coord > 0 and zone_code == 'O':
                                    away_team_side = 'right'
                                    home_team_side = 'left'
                            break  # Stop after determining the sides

            # Update the DataFrame with the determined sides based on the period
            def get_offensive_side(row):
                if row['gameId'] == game_id:
                    if row['teamId'] == row['homeTeamId']:
                        # Home team logic with modulo operation
                        return home_team_side if row['period'] % 2 == 1 else ('left' if home_team_side == 'right' else 'right')
                    elif row['teamId'] == row['awayTeamId']:
                        # Away team logic with modulo operation
                        return away_team_side if row['period'] % 2 == 1 else ('left' if away_team_side == 'right' else 'right')
                return 'unknown'
            
            # Add the column to the DataFrame
            self.df.loc[self.df['gameId'] == game_id, 'offensiveSide'] = self.df[self.df['gameId'] == game_id].apply(get_offensive_side, axis=1)

    def calculate_shot_distance_and_angle(self):
        """
        Calculates the Euclidean distance of each shot from the net based on the offensive side,
        accounting for shots originating from different zones (offensive, neutral, defensive).
        """
        def get_distance_and_angle(row):
            try:
                stripped = row['coordinates'].strip("()")
                x_str, y_str = stripped.split(",")
                x_shot, y_shot = int(x_str), int(y_str)
            #Missing values are handled by returning None
            except ValueError:
                return None, None
            
            # Determine net coordinates based on offensive side
            if row['offensiveSide'] == 'right':
                x_net, y_net = 89, 0
            elif row['offensiveSide'] == 'left':
                x_net, y_net = -89, 0
            else:
                return None, None  # Unknown side

            # Calculate Euclidean distance
            if (row['offensiveSide'] == 'right' and x_shot < 0) or (row['offensiveSide'] == 'left' and x_shot > 0):
                #If the shot is in neutral or defensive zone for example
                distance = math.sqrt((abs(x_shot) + abs(x_net))**2 + y_shot**2)

            else:
                distance = math.sqrt((x_shot - x_net)**2 + (y_shot)**2)
            
            #Neglect shots behind net
            if x_shot > 89 or x_shot < -89:
                return distance, None
            # Calculate the angle of the shot relative to y = 0 (centerline)
            angle = math.degrees(math.atan2(y_shot, abs(x_net - x_shot)))

            # Adjust the sign of the angle based on the side
            if row['offensiveSide'] == 'right':
                angle = -angle  # Reverse the angle sign for left offensive side

            return x_shot, y_shot, distance, angle

        # Apply the function to each row and create a new column 'shotDistance'
        self.df[['xCoord', 'yCoord', 'shotDistance', 'shotAngle']] = self.df.apply(lambda row: pd.Series(get_distance_and_angle(row)), axis=1)

    def add_previous_event_features(self):
        """
        Adds information about the previous event for each shot, including:
        - Last event type
        - Coordinates of the last event (x, y)
        - Time elapsed since the last event (seconds)
        - Distance from the last event
        """

        # Pre-fetch game data for all unique game IDs in the DataFrame
        unique_game_ids = self.df['gameId'].unique()
        game_data_cache = {game_id: self.game_data.get(str(game_id), {}).get('plays', []) for game_id in unique_game_ids}

        last_event_types = []
        last_event_x_coords = []
        last_event_y_coords = []
        time_elapsed_list = []
        distance_from_last_event = []

        for _, row in self.df.iterrows():
            game_id = str(row['gameId'])
            event_id = row['eventId']
            current_time = row['timeInPeriod']
            x_coord = row['xCoord']
            y_coord = row['yCoord']

            # Initialize defaults
            last_event_type = None
            last_x = None
            last_y = None
            elapsed_time = None
            distance = None

            # Fetch plays for the current game from the cache
            all_plays = game_data_cache.get(game_id, [])    

            # Find the index of the current event in all_plays, with additional criteria for unique identification
            current_index = next(
                (i for i, play in enumerate(all_plays)
                if play.get('eventId') == event_id
                and play.get('timeInPeriod') == current_time
                and (play.get('details', {}).get('xCoord'), play.get('details', {}).get('yCoord')) == (x_coord, y_coord)),
                None
            )

            # Null check for current_index
            if current_index is not None:
                if current_index > 0:
                    prev_event = all_plays[current_index - 1]
                    prev_time = prev_event.get('timeInPeriod')
                    prev_period = prev_event.get('periodDescriptor', {}).get('number')

                    #Ensure previous event is from the same period
                    if prev_period is not None and prev_period == row['period']:  
                        last_event_type = prev_event.get('typeDescKey')
                        last_x = prev_event.get('details', {}).get('xCoord')
                        last_y = prev_event.get('details', {}).get('yCoord')
                    
                        #Calculate time elapsed
                        if prev_time and current_time:
                            try:
                                curr_minutes, curr_seconds = map(int, current_time.split(":"))
                                prev_minutes, prev_seconds = map(int, prev_time.split(":"))
                                elapsed_time = (curr_minutes * 60 + curr_seconds) - (prev_minutes * 60 + prev_seconds)
                            except ValueError:
                                elapsed_time = None

                        # Calculate distance from the last event
                        try:
                            last_x = int(last_x) if last_x is not None else None
                            last_y = int(last_y) if last_y is not None else None
                            if last_x is not None and last_y is not None:
                                distance = math.sqrt((x_coord - last_x)**2 + (y_coord - last_y)**2)
                        except (ValueError, TypeError):
                            distance = None  
                    else: 
                        # If the previous event is from a different period, set default values
                        last_event_type = "Start of period"
                        last_x = None
                        last_y = None
                        elapsed_time = 0
                        distance = 0
                else:
                    # If this is the first event of the game or period, set default values
                    last_event_type = "Start of game"
                    last_x = None
                    last_y = None
                    elapsed_time = 0
                    distance = 0
            # Append results to lists
            last_event_types.append(last_event_type)
            last_event_x_coords.append(last_x)
            last_event_y_coords.append(last_y)
            time_elapsed_list.append(elapsed_time)
            distance_from_last_event.append(distance)


        # Add the new columns to the DataFrame
        self.df['lastEventType'] = last_event_types
        self.df['lastEventXCoord'] = last_event_x_coords
        self.df['lastEventYCoord'] = last_event_y_coords
        self.df['timeElapsedSinceLastEvent'] = time_elapsed_list
        self.df['distanceFromLastEvent'] = distance_from_last_event

    def previous_event_analysis(self):
        """
        Adds three new features to the DataFrame:
        - Rebound (bool): True if the last event was also a shot, otherwise False.
        - Change in shot angle: The change in shot angle relative to the previous shot, only calculated for rebounds.
        - Speed ('vitesse'): The distance from the previous event divided by the time elapsed since the previous event.
        """
        rebounds = []
        shot_angle_changes = []
        speeds = []

        for idx, row in self.df.iterrows():
            # Initialize default values
            rebound = False
            shot_angle_change = None
            speed = None
            last_angle = 0

            # Determine if the last event was a shot (rebound)
            last_event_type = row['lastEventType']
            if last_event_type == 'shot-on-goal':
                rebound = True

                # Calculate change in shot angle
                if pd.notnull(row['lastEventXCoord']) and pd.notnull(row['lastEventYCoord']) and pd.notnull(row['shotAngle']):
                    try:
                        # Calculate the angle of the previous shot relative to the net
                        if row['offensiveSide'] == 'right':
                            x_net, y_net = 89, 0
                        elif row['offensiveSide'] == 'left':
                            x_net, y_net = -89, 0
                        else:
                            last_angle = None

                        if last_angle is not None:
                            last_angle = math.degrees(math.atan2(row['lastEventYCoord'], abs(x_net - row['lastEventXCoord'])))
                            # Adjust the sign of the angle based on the side
                            if row['offensiveSide'] == 'right':
                                last_angle = -last_angle  # Reverse the angle sign for left offensive side

                            shot_angle_change = abs(row['shotAngle'] - last_angle)

                    except (ValueError, TypeError):
                        shot_angle_change = None

            # Calculate speed if time elapsed is available          
            if pd.notnull(row['timeElapsedSinceLastEvent']) and row['timeElapsedSinceLastEvent'] > 0:
                if pd.notnull(row['distanceFromLastEvent']):
                    speed = row['distanceFromLastEvent'] / row['timeElapsedSinceLastEvent']  


            # Append calculated values to lists
            rebounds.append(rebound)
            shot_angle_changes.append(shot_angle_change)
            speeds.append(speed)
        
        # Add the new columns to the DataFrame
        self.df['rebound'] = rebounds
        self.df['changeInShotAngle'] = shot_angle_changes
        self.df['speed'] = speeds    

    def save_dataframe(self, output_filename='enhanced_parsed_shot_events.csv'):
        """
        Saves the modified DataFrame to a CSV file in the data directory.
        Args:
            output_filename (str): The name of the output CSV file. Defaults to 'modified_shot_events.csv'.
        """
        output_path = os.path.join(self.data_dir, output_filename)
        self.df.to_csv(output_path, index=False)
        print(f"DataFrame saved to {output_path}")  

def main():
    feature_engineer = ShotEventFeatureEngineer()
    feature_engineer.add_team_ids()
    feature_engineer.add_empty_net_goal_column()
    feature_engineer.determine_offensive_side()
    feature_engineer.calculate_shot_distance_and_angle()
    feature_engineer.add_previous_event_features()
    feature_engineer.previous_event_analysis()

    #Save new csv
    feature_engineer.save_dataframe()

    # Get the unique game IDs
    unique_game_ids = feature_engineer.df['gameId'].unique()[:1]
    
    # Filter and print the first 3 shot events for each of the first 'unique_game_ids' games
    for game_id in unique_game_ids:
        print(f"\nFirst 4 shot events for game ID {game_id}:")
        game_shots = feature_engineer.df[feature_engineer.df['gameId'] == game_id].head(4)
        print(game_shots)

if __name__ == "__main__":
    main()
