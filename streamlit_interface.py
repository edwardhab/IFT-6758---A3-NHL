import streamlit as st
import pandas as pd
import numpy as np
import json
from src.client.game_client import GameClient  # Corrected import
from src.client.serving_client import ServingClient

st.title("Hockey Visualization App")

# Corrected instantiation of ServingClient
sc = ServingClient(ip='serving', port=5000)

# Corrected instantiation of GameClient
gc = GameClient(api_base_url="https://api-web.nhle.com/v1/gamecenter", serving_client=sc)

with st.sidebar:
    # Sélection des paramètres pour télécharger le modèle
    workspace = st.selectbox(
        "Work space",
        ("ift6758-24-team3", "only one workspace")
    )
    model = st.selectbox(
        "Model",
        ("logreg_comb:latest", "logreg_dist:latest")
    )
    version = st.selectbox(
        "Version",
        ("1.0.0", "only one version")
    )
    
    if st.button('Get Model'):
        # Téléchargement du modèle depuis Wandb
        with open('tracker.json', 'w') as outfile:
            json.dump({}, outfile)  # Réinitialisation du fichier tracker.json à chaque changement de modèle
        sc.download_registry_model(workspace=workspace, model=model, version=version)
        st.write('Model Downloaded')

# Fonction pour récupérer et afficher les informations sur le jeu
def ping_game_id(game_id):
    # Récupérer les données du jeu en utilisant le client game
    df, live, period, timeLeft, home, away, home_score, away_score = gc.process_game(game_id)
    
    with st.container():
        home_xG = 0
        away_xG = 0
        
        st.subheader(f"Game {game_id}: {home} vs {away}")
        
        # Afficher le statut du jeu (en direct ou déjà terminé)
        if live:
            st.write(f'Period {period} - {timeLeft} left')
        else:
            st.write(f'Game already ended, total number of periods: {period}')
        
        # Traitement des prédictions de xG si des données sont disponibles
        if len(df) != 0:
            y = sc.predict(df)  # Prédiction des buts attendus (xG) avec le modèle
            df_y = pd.DataFrame(y.items())
            df['xG'] = df_y.iloc[:, 1]
            
            # Calcul des buts attendus pour chaque équipe
            home_xG = df.loc[df['home'] == df['team'], ['xG']].sum()['xG']
            away_xG = df.loc[df['away'] == df['team'], ['xG']].sum()['xG']
        
        # Lire et mettre à jour le tracker.json
        try:
            with open('tracker.json') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        
        if str(game_id) not in data:
            data[str(game_id)] = {'home_xG': 0, 'away_xG': 0}
        
        data[str(game_id)]['home_xG'] += home_xG
        data[str(game_id)]['away_xG'] += away_xG

        home_xG = data[str(game_id)]['home_xG']
        away_xG = data[str(game_id)]['away_xG']
        
        # Affichage des informations sur les buts attendus et les scores actuels
        cols = st.columns(2)
        cols[0].metric(label=f'{home} xG (actual)', value=f"{home_xG} ({home_score})", delta=home_score - home_xG)
        cols[1].metric(label=f'{away} xG (actual)', value=f"{away_xG} ({away_score})", delta=away_score - away_xG)
        
        # Sauvegarde des données dans tracker.json
        with open('tracker.json', 'w') as outfile:
            json.dump(data, outfile)
        
        # Affichage des données utilisées pour les prédictions
        df = df.reset_index(drop=True)
        st.subheader("Data used for predictions (and predictions)")

        if len(df) != 0:
            st.table(df)
        else:
            st.write(f"We have seen all the events for game {game_id}")


# Entrée de l'ID du jeu et bouton pour déclencher l'appel à l'API
with st.container():
    game_id = st.text_input("Specify the game ID to ping, e.g. 2021020329", '')
    if st.button('Ping game'):
        ping_game_id(game_id)
