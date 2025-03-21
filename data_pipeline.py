import json
import requests
import pandas as pd
import sqlite3

''' 
# Fetch data from FPL website API if needed
def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url) # HTTP GET request
    data = response.json() # convert JSON into dictionary
    players_dataframe = pandas.DataFrame(data['elements']) # use Pandas dataframe for

    return players_dataframe
'''

def load_season_data(season):
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"

    # Load the CSV file with error handling
    try:
        df = pd.read_csv(url, on_bad_lines='skip')  # Skip problematic rows
        return df
    except Exception as e:
        print(f"Error loading data for season {season}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error


def load_all_data():
    # Load data for the previous season 2023-24 and the current season (2024-25)

    # Load data for the previous season
    df_2023_24 = load_season_data("2023-24")
    df_2023_24['season'] = '2023-24'  # Add a column to identify the season

    # Load data for the current season
    df_2024_25 = load_season_data("2024-25")
    df_2024_25['season'] = '2024-25'  # Add a column to identify the season

    # Combine both seasons into a single DataFrame
    combined_df = pd.concat([df_2023_24, df_2024_25], ignore_index=True)

    # Debug: Print column names and sample data
    print("Columns in the dataset:", combined_df.columns.tolist())
    print("Sample data:\n", combined_df.head())

    # Return combined df for both seasons
    return combined_df


if __name__ == '__main__':
    # Load and print the combined data
    data = load_all_data()
    print(data.head())
