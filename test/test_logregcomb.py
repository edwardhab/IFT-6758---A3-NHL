import logging
from game_client import GameClient
from serving_client import ServingClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_game_client():
    """
    Test the GameClient with real data from the NHL API.
    """
    # Initialize ServingClient (ensure your server is running)
    serving_client = ServingClient(ip="127.0.0.1", port=5000)

    # Initialize GameClient with the NHL API base URL
    api_base_url = "https://api-web.nhle.com/v1/gamecenter"
    game_client = GameClient(api_base_url=api_base_url, serving_client=serving_client)

    # Specify the game ID dynamically (update this as needed)
    
    game_id = input("Enter the game ID to fetch data for: ")

    try:

        # Process the game and get predictions
        logger.info(f"Processing game ID: {game_id}")
        predictions = game_client.process_game(game_id)

        # Print the results
        if not predictions.empty:
            print("\nPredictions for game ID:", game_id)
            print(predictions)
        else:
            print(f"No new events to process for game ID: {game_id}.")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_real_game_client()
