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
        self.processed_event_ids = {}  # Dictionary to track processed event IDs for each game

    def filter_new_events(self, game_id: str, events: pd.DataFrame) -> pd.DataFrame:
        """
        Filters events that have not been processed yet for a specific game.

        Args:
            game_id (str): The game ID.
            events (pd.DataFrame): All events.

        Returns:
            pd.DataFrame: New events to be processed.
        """
        if game_id not in self.processed_event_ids:
            self.processed_event_ids[game_id] = set()
        # Start by filtering out events that have already been processed
        new_events = events[~events['eventId'].isin(self.processed_event_ids[game_id])]
        self.processed_event_ids[game_id].update(new_events['eventId'])
        return new_events

    def preprocess_events(self, game_id: str) -> pd.DataFrame:
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
        processor.parse_nhl_game_data()

        # Add derived features
        processor.add_team_ids()
        processor.add_empty_net_goal_column()
        processor.determine_offensive_side()
        processor.calculate_shot_distance_and_angle()

        # Filter out already processed events
        new_events = self.filter_new_events(game_id, processor.nhl_shot_events)
        if new_events.empty:
            logger.info("No new events to process.")
            return pd.DataFrame()

        # Return only the necessary columns
        return new_events[['eventId', 'distance', 'angle', 'result', 'emptyNetGoal']]

    def send_to_prediction_service(self, preprocessed_events: pd.DataFrame) -> pd.DataFrame:
        """
        Sends preprocessed events to the prediction service and retrieves probabilities.

        Args:
            preprocessed_events (pd.DataFrame): The preprocessed events.

        Returns:
            pd.DataFrame: Events with prediction probabilities added.
        """
        return self.serving_client.predict(preprocessed_events)

    def process_game(self, game_id: str) -> pd.DataFrame:
        """
        Processes a game: fetches events, preprocesses them, and sends them to the prediction service.

        Args:
            game_id (str): The game ID.

        Returns:
            pd.DataFrame: Processed events with predictions.
        """
        preprocessed_events = self.preprocess_events(game_id)
        if preprocessed_events.empty:
            logger.info("No new events to process.")
            return pd.DataFrame()
        
        preprocessed_events = preprocessed_events.fillna(0)
        
        processed_events = self.send_to_prediction_service(preprocessed_events)
        if processed_events.empty:
            logger.info("No more events left to process.")

        return processed_events
