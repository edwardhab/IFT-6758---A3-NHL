import logging
from ..src.client.game_client import GameClient
from ..src.client.serving_client import ServingClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_game_client():
    """
    Test the GameClient with real data from the NHL API, while switching models.
    """
    # Initialize ServingClient (ensure your server is running)
    serving_client = ServingClient(ip="127.0.0.1", port=5000)

    # Switch to the "logreg_dist:latest" model
    try:
        logger.info("Switching to the model")
        # CHANGE MODEL NAME HERE TO "logreg_dist" TO SWITCH TO THE DISTRIBUTED MODEL OR "logreg_comb" TO SWITCH TO THE COMBINED MODEL
        model_switch_response = serving_client.download_registry_model(model="logreg_dist", version="latest")
        logger.info(f"Model switched successfully: {model_switch_response}")
    except Exception as e:
        logger.error(f"Error during model switch: {e}")
        print(f"An error occurred while switching models: {e}")
        return

    # Initialize GameClient with the NHL API base URL
    api_base_url = "https://api-web.nhle.com/v1/gamecenter"
    game_client = GameClient(api_base_url=api_base_url, serving_client=serving_client)

    # Prompt for the first game ID
    game_id = input("Enter the first game ID to fetch data for: ")

    try:
        # Process the first game and get predictions
        logger.info(f"Processing first game ID: {game_id}")
        print(serving_client.logs())
        predictions = game_client.process_game(game_id)

        # Print the results for the first game
        if not predictions.empty:
            print(f"\nPredictions for the first game ID {game_id}:")
            print(predictions)
        else:
            print(f"No new events to process for game ID: {game_id}.")
    except Exception as e:
        logger.error(f"Error during processing of first game ID: {e}")
        print(f"An error occurred: {e}")

    # Prompt for the second game ID
    second_game_id = input("\nEnter the second game ID to fetch data for: ")

    try:
        # Process the second game and get predictions
        logger.info(f"Processing second game ID: {second_game_id}")
        second_predictions = game_client.process_game(second_game_id)

        # Print the results for the second game
        if not second_predictions.empty:
            print(f"\nPredictions for the second game ID {second_game_id}:")
            print(second_predictions)
        else:
            print(f"No new events to process for game ID: {second_game_id}.")
    except Exception as e:
        logger.error(f"Error during processing of second game ID: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_real_game_client()
