import streamlit as st
import pandas as pd
import numpy as np
import json
from src.client.game_client import GameClient  # Corrected import
from src.client.serving_client import ServingClient

st.title("Hockey Visualization App")

# Initialize ServingClient in session_state if it doesn't already exist
if 'serving_client' not in st.session_state:
    st.session_state.serving_client = ServingClient(ip='serving', port=5000)

# Initialize GameClient in session_state if it doesn't already exist
if 'game_client' not in st.session_state:
    st.session_state.game_client = GameClient(
        api_base_url="https://api-web.nhle.com/v1/gamecenter",
        serving_client=st.session_state.serving_client
    )

# Use the persistent instances
sc = st.session_state.serving_client
gc = st.session_state.game_client

with st.sidebar:
    # Select model and download functionality
    workspace = st.selectbox(
        "Work space",
        ("IFT6758.2024-A03", "other")
    )
    model = st.selectbox(
        "Model",
        ("logreg distance + angle", "logreg distance")
    )
    version = st.selectbox(
        "Version",
        ("latest", "other")
    )
    
    if st.button('Get Model'):
        # Reset tracker.json on model change
        # with open('tracker.json', 'w') as outfile:
        #     json.dump({}, outfile)  # Reset tracker.json
        model_name = "logreg_comb" if model == "logreg distance + angle" else "logreg_dist"
        sc.download_registry_model(workspace=str(workspace), model=str(model_name), version=str(version))
        st.write('Model Downloaded')

def ping_game_id(game_id):
    try:
        # Load tracker data
        with open('tracker.json') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Use the current model name as the top-level key
    if len(sc.features) == 2:
        model = "logreg_comb"
    elif len(sc.features) == 1:
        model = "logreg_dist"    

    if model not in data:
        data[model] = {}  # Initialize model-specific section

    # Initialize game-specific data if not present
    if game_id not in data[model]:
        data[model][game_id] = {
            'cumulative_home_xG': 0.0,
            'cumulative_away_xG': 0.0,
            'home_score': 0,
            'away_score': 0,
            'home_team': "Unknown",
            'away_team': "Unknown",
            'period': 1,
            'remainingTimeInPeriod': '20:00',
            'results_df': []
        }

    # Process game using GameClient
    result = gc.process_game(game_id, model)
    if result is None:
        # If no new events, display current data
        with st.container():
            total_home_xG = data[model][game_id]['cumulative_home_xG']
            total_away_xG = data[model][game_id]['cumulative_away_xG']
            home_score = data[model][game_id]['home_score']
            away_score = data[model][game_id]['away_score']
            home_team = data[model][game_id]['home_team']
            away_team = data[model][game_id]['away_team']
            period = data[model][game_id]['period']
            remainingTimeInPeriod = data[model][game_id]['remainingTimeInPeriod']

            with st.container():
                st.subheader(f"Game {game_id}: {home_team} vs {away_team}")
                if period == 3 and remainingTimeInPeriod == '00:00':
                    st.write('Game already ended')  
                st.write(f'Period {period} - {remainingTimeInPeriod} left')

                cols = st.columns(2)
                cols[0].metric(label=f'{home_team} xG (actual)', value=f"{total_home_xG:.1f} ({int(home_score)})", delta=round(home_score - total_home_xG, 1))
                cols[1].metric(label=f'{away_team} xG (actual)', value=f"{total_away_xG:.1f} ({int(away_score)})", delta=round(away_score - total_away_xG, 1))
                st.write('No new events to process.')

                # Display previously stored results_df if available
                old_results = data[model][game_id].get('results_df', [])

                old_results_df = pd.DataFrame(old_results)
                st.subheader("Data used for predictions (and predictions))")
                st.table(old_results_df)

        return

    # Unpack result and update tracker
    home_team, away_team, period, remainingTimeInPeriod, final_home_score, final_away_score, cumulative_home_xG, cumulative_away_xG, results_df = result

    # Convert any NumPy numeric types to Python natives
    final_home_score = int(final_home_score)
    final_away_score = int(final_away_score)
    period = int(period)
    cumulative_home_xG = float(cumulative_home_xG)
    cumulative_away_xG = float(cumulative_away_xG)

    data[model][game_id]['cumulative_home_xG'] += cumulative_home_xG
    data[model][game_id]['cumulative_away_xG'] += cumulative_away_xG
    data[model][game_id]['home_score'] = final_home_score
    data[model][game_id]['away_score'] = final_away_score
    data[model][game_id]['home_team'] = home_team
    data[model][game_id]['away_team'] = away_team
    data[model][game_id]['period'] = period
    data[model][game_id]['remainingTimeInPeriod'] = remainingTimeInPeriod

    old_results = data[model][game_id].get('results_df', [])
    if old_results:
        old_results_df = pd.DataFrame(old_results)
        combined_results_df = pd.concat([old_results_df, results_df], ignore_index=True)
    else:
        combined_results_df = results_df

    # Convert all integer columns from int64 to int (if any)
    for col in combined_results_df.select_dtypes(include=[np.int64]).columns:
        combined_results_df[col] = combined_results_df[col].astype(int)

    data[model][game_id]['results_df'] = combined_results_df.to_dict(orient='records')
    # Save updated tracker data
    with open('tracker.json', 'w') as outfile:
        json.dump(data, outfile)

    # Display updated metrics
    with st.container():
        st.subheader(f"Game {game_id}: {home_team} vs {away_team}")
        if period == 3 and remainingTimeInPeriod == '00:00':
            st.write('Game already ended')
        if period > 3:
            st.write('Game has gone into overtime')  
        st.write(f'Period {period} - {remainingTimeInPeriod} left')

        total_home_xG = data[model][game_id]['cumulative_home_xG']
        total_away_xG = data[model][game_id]['cumulative_away_xG']

        cols = st.columns(2)
        cols[0].metric(label=f'{home_team} xG (actual)', value=f"{total_home_xG:.1f} ({int(final_home_score)})", delta=round(final_home_score - total_home_xG, 1))
        cols[1].metric(label=f'{away_team} xG (actual)', value=f"{total_away_xG:.1f} ({int(final_away_score)})", delta=round(final_away_score - total_away_xG, 1))

        # Display results_df
        st.subheader("Data used for predictions (and predictions)")
        if len(combined_results_df) != 0:
            st.table(combined_results_df)
        else:
            st.write(f"We have seen all the events for game {game_id}")

# Input for game ID and button to trigger API call
with st.container():
    game_id = st.text_input("Specify the game ID to ping, e.g. 2021020329", '')
    if st.button('Ping game'):
        ping_game_id(game_id)
