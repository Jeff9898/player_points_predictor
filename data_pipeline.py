import pandas as pd

# Load data for specific FPL seasons
def load_season_data(season):
    # Construct URL to access the CSV file for specific season
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
    # Load csv file into pandas dataframe
    df = pd.read_csv(url, on_bad_lines='skip')

    return df

# Load and combine data for 2023-24 and 2024-25 seasons
def load_all_data():
    # Load data for 23/24 season and add a column to identify the season
    df_2023_24 = load_season_data("2023-24")
    df_2023_24['season'] = '2023-24'

    # Load data for 24/25 season and add a column to identify the season
    df_2024_25 = load_season_data("2024-25")
    df_2024_25['season'] = '2024-25'

    # combine both seasons data into a single dataframe
    combined_df = pd.concat([df_2023_24, df_2024_25], ignore_index=True)

    return combined_df


if __name__ == '__main__':
    data = load_all_data()
    print(data.head())

