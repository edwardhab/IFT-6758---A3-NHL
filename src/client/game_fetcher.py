import requests
import pandas as pd
import math

class NHLGameDataProcessor:
    def __init__(self, game_id: str, api_base_url: str = "https://api-web.nhle.com/v1/gamecenter"):
        """
        Initializes the NHLGameDataProcessor for a specific game.

        Args:
            game_id (str): The game ID to process.
            api_base_url (str): Base URL for the NHL API.
        """
        self.game_id = game_id
        self.api_base_url = api_base_url
        self.period = None
        self.timeRemainingInPeriod = None
        self.nhl_game_data = None
        self.nhl_shot_events = None
        self.nhl_game_events = None

    def fetch_game_data(self) -> pd.DataFrame:
        """
        Fetches play-by-play data for the specified game ID.

        Returns:
            pd.DataFrame: A DataFrame containing all events for the game.
        """
        url = f"{self.api_base_url}/{self.game_id}/play-by-play"
        response = requests.get(url)
        response.raise_for_status()  # Raises an error if the request fails
        self.nhl_game_data = response.json()
        self.nhl_game_events = response.json().get("plays", [])


    def parse_nhl_game_data(self):
        """
        Parses the NHL game data and filters for shot-related events.

        Returns:
            pd.DataFrame: A DataFrame containing shot-related events for the game.
        """
        if not self.nhl_game_events:
            print("No game events loaded.")
            return pd.DataFrame()

        shot_events = []

        for play in self.nhl_game_events:
            event_type = play.get('typeCode')
            if event_type in [505, 506]:  # 505: Goal, 506: Shot
                situation_code = play.get('situationCode', "0000")
                shot_details = {
                    'gameId': self.game_id,
                    'eventId': play.get('eventId'),
                    'homeTeam': self.nhl_game_data.get("homeTeam", {}).get("commonName", {}).get("default", "Unknown Home Team"),
                    'homeTeamId': self.nhl_game_data.get("homeTeam", {}).get("id", {}),
                    'awayTeam': self.nhl_game_data.get("awayTeam", {}).get("commonName", {}).get("default", "Unknown Away Team"),
                    'awayTeamId': self.nhl_game_data.get("awayTeam", {}).get("id", {}),
                    'homeScore': play.get('details', {}).get('homeScore'),
                    'awayScore': play.get('details', {}).get('awayScore'),
                    'period': play.get('periodDescriptor', {}).get('number'),
                    'eventType': play.get('typeDescKey'),
                    'teamId': play.get('details', {}).get('eventOwnerTeamId'),
                    'emptyNetAway': situation_code[0] != '1',
                    'emptyNetHome': situation_code[3] != '1',
                    'coordinates': (
                        play.get('details', {}).get('xCoord'),
                        play.get('details', {}).get('yCoord'),
                    ),
                    'result': 'goal' if event_type == 505 else 'no goal',
                }
                shot_events.append(shot_details)

        self.nhl_shot_events = pd.DataFrame(shot_events)
        return self.nhl_shot_events

    def add_team_ids(self):
        """
        Adds home and away team IDs to the DataFrame by extracting them from the game data JSON file.
        """
        home_ids = []
        away_ids = []

        # Iterate through each shot event in the DataFrame
        for _, row in self.nhl_shot_events.iterrows():
            # Directly use the `self.nhl_game_data` object, since `gameId` is not used as a key in `nhl_game_data`
            game_details = self.nhl_game_data

            # Access the `homeTeam` and `awayTeam` details directly
            home_team_id = game_details.get('homeTeam', {}).get('id', None)
            away_team_id = game_details.get('awayTeam', {}).get('id', None)

            # Append team IDs or None if not available
            home_ids.append(int(home_team_id) if home_team_id is not None else None)
            away_ids.append(int(away_team_id) if away_team_id is not None else None)

        # Add the home and away team IDs as new columns in the DataFrame
        self.nhl_shot_events['homeTeamId'] = home_ids
        self.nhl_shot_events['awayTeamId'] = away_ids

        # Ensure columns are of integer type, fill NaN with a default value if necessary
        self.nhl_shot_events['homeTeamId'] = self.nhl_shot_events['homeTeamId'].fillna(0).astype(int)
        self.nhl_shot_events['awayTeamId'] = self.nhl_shot_events['awayTeamId'].fillna(0).astype(int)


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
        self.nhl_shot_events['emptyNetGoal'] = self.nhl_shot_events.apply(is_empty_net_goal, axis=1)        

    def determine_offensive_side(self):
        """
        Determines the offensive side (left or right) for the home and away teams based on the first non-neutral
        shot of the first period, using the instance variables for game data.

        Returns:
            pd.DataFrame: Updated DataFrame with an 'offensiveSide' column.
        """
        if self.nhl_shot_events is None or self.nhl_shot_events.empty:
            print("Shot events data is not available.")
            return

        home_team_side = 'unknown'
        away_team_side = 'unknown'

        # Fetch the first non-neutral shot from the first period
        for _, shot in self.nhl_shot_events.iterrows():
            if shot['period'] == 1:
                # Fetch play details from game data
                all_plays = self.nhl_game_events
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

        def get_offensive_side(row):
            if row['teamId'] == row['homeTeamId']:
                return home_team_side if row['period'] % 2 == 1 else ('left' if home_team_side == 'right' else 'right')
            elif row['teamId'] == row['awayTeamId']:
                return away_team_side if row['period'] % 2 == 1 else ('left' if away_team_side == 'right' else 'right')
            return 'unknown'

        self.nhl_shot_events['offensiveSide'] = self.nhl_shot_events.apply(get_offensive_side, axis=1)

    
    def calculate_shot_distance_and_angle(self):
        """
        Calculates the Euclidean distance of each shot from the net based on the offensive side,
        accounting for shots originating from different zones (offensive, neutral, defensive).
        """
        if self.nhl_shot_events is None or self.nhl_shot_events.empty:
            print("Shot events data is not available.")
            return

        def get_distance_and_angle(row):
            try:
                x_shot, y_shot = row['coordinates']
                if x_shot is None or y_shot is None:
                    return None, None
            except (ValueError, TypeError):
                return None, None
            # Determine net coordinates based on offensive side of the team shooting
            if row['offensiveSide'] == 'right':
                x_net, y_net = 89, 0
            elif row['offensiveSide'] == 'left':
                x_net, y_net = -89, 0
            else:
                return None, None  # Unknown side

            # Calculate Euclidean distance
            if (row['offensiveSide'] == 'right' and x_shot < 0) or (row['offensiveSide'] == 'left' and x_shot > 0):
                # If the shot is in the neutral or defensive zone
                distance = math.sqrt((abs(x_shot) + abs(x_net))**2 + y_shot**2)
            else:
                distance = math.sqrt((x_shot - x_net)**2 + (y_shot)**2)

            # Neglect shots behind the net
            if x_shot > 89 or x_shot < -89:
                return distance, None

            # Calculate the angle of the shot relative to y = 0 (centerline)
            angle = math.degrees(math.atan2(y_shot, abs(x_net - x_shot)))

            # Adjust the sign of the angle based on the side
            if row['offensiveSide'] == 'right':
                angle = -angle  # Reverse the angle sign for left offensive side

            return distance, angle

        # Apply the function to calculate shot distance and angle for each row
        self.nhl_shot_events[['distance', 'angle']] = self.nhl_shot_events.apply(
            lambda row: pd.Series(get_distance_and_angle(row)), axis=1
        )

    def update_scores(self):
        """
        Updates missing homeScore and awayScore values by forward-filling
        the last known scores.

        Args:
            df (pd.DataFrame): The input DataFrame containing shot events.

        Returns:
            pd.DataFrame: The DataFrame with updated homeScore and awayScore columns.
        """
        #Initialize starting scores to 0
        if not self.nhl_shot_events.empty:
            if pd.isna(self.nhl_shot_events.loc[0, 'homeScore']):
                self.nhl_shot_events.loc[0, 'homeScore'] = 0
            if pd.isna(self.nhl_shot_events.loc[0, 'awayScore']):
                self.nhl_shot_events.loc[0, 'awayScore'] = 0

        # Forward-fill the missing scores
        self.nhl_shot_events['homeScore'] = self.nhl_shot_events['homeScore'].fillna(method='ffill')
        self.nhl_shot_events['awayScore'] = self.nhl_shot_events['awayScore'].fillna(method='ffill')

    def get_time_remaining_period(self):
        """
        Retrieves the time remaining in the period for the last play of the game.

        Returns:
            str: The time remaining in the period for the last play, or None if no plays exist.
        """
        if not self.nhl_game_events:
            print("No plays available.")
            return None
        #'timeRemainingInPeriod': play.get('timeRemaining', {}),
        # Get the last play
        last_play = self.nhl_game_events[-1]
        # Access the time remaining in the period
        self.period = last_play.get('periodDescriptor', {}).get('number')
        self.timeRemainingInPeriod = last_play.get('timeRemaining', {})