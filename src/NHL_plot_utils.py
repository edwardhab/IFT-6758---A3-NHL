import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import ast
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def league_shot_avg(df: pd.DataFrame, year: int) -> np.array:
    """
    Computes the average number of shots per hour across the league for 1 season.
    :param df: tidy frame with xy coordinates projected on a half-rink
    :param year: int representation of a year. ex: 2018
    :return: np.array with shape 100 X 85 (i.e. the XY half rink plane)
    """
    season = int(str(year) + str(year + 1))

    df_copy = df[df["season"] == season].copy()
    df_copy["coord_tuple"] = df_copy[["x_coordinate", "y_coordinate"]].apply(tuple, axis=1)

    data_league = np.zeros((100, 85))

    for i, j in df_copy["coord_tuple"]:
        if np.isnan(i) or np.isnan(j):
            pass
        else:
            data_league[int(i), int(j)] += 1

    total_games = df_copy.drop_duplicates(subset=['gameId'], keep='last')['gameId'].count()
    data_league = data_league / (total_games * 2)

    return data_league


def team_shot_avg(df: pd.DataFrame, year: int, team: str) -> np.array:
    """
    Computes the average number of shots per hour for 1 team for 1 season.
    :param df: tidy frame with xy coordinates projected on a half-rink
    :param year: int representation of a year. ex: 2016
    :return: np.array with shape 100 X 85 (i.e. the XY half rink plane)
    """
    season = int(str(year) + str(year + 1))

    # Filter for the specified season and team
    df_copy = df[df["season"] == season].copy()
    df_copy2 = df_copy[df_copy["teamId"] == team].copy()
    df_copy2["coord_tuple"] = df_copy2[["x_coordinate", "y_coordinate"]].apply(tuple, axis=1)

    data_team = np.zeros((100, 85))

    for i, j in df_copy2["coord_tuple"]:
        if np.isnan(i) or np.isnan(j):
            pass
        else:
            data_team[int(i), int(j)] += 1


    total_team_games = df_copy2.drop_duplicates(subset=["gameId"], keep="last")['gameId'].count()

    if total_team_games > 0:
        data_team = data_team / total_team_games

    return data_team

def all_team_avg(df: pd.DataFrame, start_year: int = 2016, end_year: int = 2020, sigma: int = 4,
                        threshold: float = 0.001) -> dict:
    """
    Computes the average number of shots per hour across the league for all season in-between start_year & end_year.
    :param df: tidy frame with xy coordinates projected on a half-rink
    :param start_year: int representation of the year of the first season of interest. ex: 2016
    :param end_year: int representation of the year of the last season of interest. ex: 2020
    :param sigma: Gaussian kernel hyper-parameter. Recommended range: [2,4]
    :param threshold: All gaussian differences within float threshold of 0 are ignored and replaced by None
    :return: dict of years of dict of teams
    """
    teams_per_season = {}
    shot_freq_per_team_per_season = {}

    # Build dict of all uniques teams per season
    for year in range(start_year, end_year + 1):
        # Index season as 20162017 for example
        season = int(str(year) + str(year + 1))
        # Get all unique teams for a season
        teams_per_season[str(year)] = np.array(df[df['season'] == season]['teamId'].unique())

    
    for year in range(start_year, end_year + 1):
        # Create a dict object for each year. Each such dict object is a dict of teams containing their individual xy
        # shot frequencies
        shot_freq_per_team_per_season[str(year)] = {}
        league_avg = league_shot_avg(df, year)
        for team in teams_per_season[str(year)]:
            team_avg = team_shot_avg(df, year, team)
            avg_difference = team_avg - league_avg
            # smoothing results
            test_total_filter = gaussian_filter(avg_difference, sigma=sigma)

            # Filter out values that are very close to zero for plotting purposes
            test_total_filter[np.abs(test_total_filter - 0) <= threshold] = None

            # Store result
            shot_freq_per_team_per_season[str(year)][team] = test_total_filter

    return shot_freq_per_team_per_season


def transform_coord() -> pd.DataFrame:
    """
    Loads the "normal" tidy dataframe and returns a projection of the XY shot coordinates on a half-rink.
    :return: half-rink projected dataframe. All columns identical to tidy_data.csv except x/y-coordinates.
    """
    rink_Tohalf_df = pd.read_csv(os.path.join(DATA_DIR, "parsed_shot_events.csv"))

    # Remove NaN coordinates. Ensure 'coordinates' are tuples and check for NaNs
    rink_Tohalf_df['coordinates'] = rink_Tohalf_df['coordinates'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )


    # Filter out rows with None coordinates
    rink_Tohalf_df = rink_Tohalf_df[rink_Tohalf_df['coordinates'].apply(lambda coord: coord is not None)]
    

    rink_Tohalf_df["x_coordinate"] = rink_Tohalf_df["coordinates"].apply(
        lambda coord: coord[0] if isinstance(coord, tuple) else np.nan
    )

    rink_Tohalf_df["y_coordinate"] = rink_Tohalf_df["coordinates"].apply(
        lambda coord: coord[1] if isinstance(coord, tuple) else np.nan
    )

    # Transform coordinates to half-rink
    rink_Tohalf_df["x_coordinate"] = np.where(
        rink_Tohalf_df["x_coordinate"] < 0,
        -rink_Tohalf_df["x_coordinate"],
        rink_Tohalf_df["x_coordinate"],
    )
    rink_Tohalf_df["y_coordinate"] = np.where(
        rink_Tohalf_df["x_coordinate"] < 0,
        -rink_Tohalf_df["y_coordinate"],
        rink_Tohalf_df["y_coordinate"],
    )
    
    return rink_Tohalf_df
