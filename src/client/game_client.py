import pandas as pd
import logging
from serving_client import ServingClient
from game_fetcher import NHLGameDataProcessor  # Import NHLGameDataProcessor

logger = logging.getLogger(__name__)

class GameClient:
    def __init__(self, api_base_url: str, serving_client: ServingClient):
        """
        Initializes the GameClient.

        Args:
            api_base_url (str): Base URL for the NHL API.
            serving_client (ServingClient): An instance of the ServingClient class to interact with the prediction service.
        """
        self.api_base_url = api_base_url
        self.serving_client = serving_client
        self.logger = logger
        self.processed_event_ids = {}  # Dictionary to track processed event IDs for each game

    def filter_new_events(self, game_id: str, events: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Filters events that have not been processed yet for a specific game.

        Args:
            game_id (str): The game ID.
            events (pd.DataFrame): All events.

        Returns:
            pd.DataFrame: New events to be processed.
        """
        # Ensure the model_name key exists
        if model_name not in self.processed_event_ids:
            self.processed_event_ids[model_name] = {}

        # Ensure the game_id key exists for the specified model_name
        if game_id not in self.processed_event_ids[model_name]:
            self.processed_event_ids[model_name][game_id] = set()

        # Filter out events that have already been processed for this model and game
        new_events = events[~events['eventId'].isin(self.processed_event_ids[model_name][game_id])]
        self.processed_event_ids[model_name][game_id].update(new_events['eventId'])
        return new_events

    def preprocess_events(self, game_id: str, model_name: str):
        """
        Fetches and preprocesses events for a specific game.

        Args:
            game_id (str): The game ID.

        Returns:
            pd.DataFrame: Preprocessed events with the required features.
        """
        # Use NHLGameDataProcessor to fetch and parse game data
        processor = NHLGameDataProcessor(game_id, self.api_base_url)
        processor.fetch_game_data()

        if processor.nhl_game_events is None  or not processor.nhl_game_events:
            logger.info("No shot events found.")
            return None
        processor.parse_nhl_game_data()
        # Add derived features
        processor.add_team_ids()
        processor.add_empty_net_goal_column()
        processor.determine_offensive_side()
        processor.calculate_shot_distance_and_angle()
        processor.update_scores()
        processor.get_time_remaining_period()

        # Filter out already processed events
        new_events = self.filter_new_events(game_id, processor.nhl_shot_events, model_name)
        if new_events.empty:
            logger.info("No new events to process.")
            return None
    
        # Return only the necessary columns
        return (
            new_events[['eventId', 'teamId', 'homeTeam', 'homeTeamId', 'awayTeam', 'awayTeamId', 
                        'homeScore', 'awayScore', 'period', 'distance', 'angle', 'result', 'emptyNetGoal']],
            processor.timeRemainingInPeriod,
            processor.period
        )

    def send_to_prediction_service(self, preprocessed_events: pd.DataFrame) -> pd.DataFrame:
        """
        Sends preprocessed events to the prediction service and retrieves probabilities.

        Args:
            preprocessed_events (pd.DataFrame): The preprocessed events.

        Returns:
            pd.DataFrame: Events with prediction probabilities added.
        """
        return self.serving_client.predict(preprocessed_events)

    def process_game(self, game_id: str, model_name: str):
        """
        Processes a game: fetches events, preprocesses them, and sends them to the prediction service.

        Args:
            game_id (str): The game ID.

        Returns:
            pd.DataFrame, bool, int, str, str, str, int, int: Processed events with predictions, game status, period, 
                                                            time left, home team, away team, home score, away score
        """
        results = self.preprocess_events(game_id, model_name)
        if results is None:
            logger.info("No new events to process.")
            return None
        preprocessed_events, remainingTimeInPeriod, period = results
        logger.info(f"Preprocessed events")
        
        preprocessed_events = preprocessed_events.fillna(0)
        
        processed_events = self.send_to_prediction_service(preprocessed_events)
                
        cumulative_home_xG = 0.0
        cumulative_away_xG = 0.0

        # Iterate over the DataFrame to calculate cumulative xG
        for index, row in preprocessed_events.iterrows():
            prediction = processed_events.loc[index, 'prediction']  # Get the prediction value
            if row['teamId'] == row['homeTeamId']:
                cumulative_home_xG += prediction
            elif row['teamId'] == row['awayTeamId']:
                cumulative_away_xG += prediction

            # Assign cumulative values to the respective columns
            preprocessed_events.at[index, 'homeExpectedGoals'] = cumulative_home_xG
            preprocessed_events.at[index, 'awayExpectedGoals'] = cumulative_away_xG

        
        final_home_score = preprocessed_events.iloc[-1]['homeScore']
        final_away_score = preprocessed_events.iloc[-1]['awayScore']
        home_team = preprocessed_events.iloc[-1]['homeTeam']
        away_team = preprocessed_events.iloc[-1]['awayTeam']

        #Return data frame with features and predictions
        if len(self.serving_client.features) == 2:
            # Extract 'distance' and 'angle'
            features_df = preprocessed_events[['distance', 'angle']].reset_index(drop=True)
        else:
            # Extract only 'distance'
            features_df = preprocessed_events[['distance']].reset_index(drop=True)
        
        predictions_df = processed_events[['prediction']].reset_index(drop=True)
        
        results_df = pd.concat([features_df, predictions_df], axis=1)
        results_df = results_df.rename(columns={'prediction': 'Model output'})
        results_df.index = [f"Event {i}" for i in range(len(results_df))]
        
        return home_team, away_team, period, remainingTimeInPeriod, final_home_score, final_away_score, cumulative_home_xG, cumulative_away_xG, results_df