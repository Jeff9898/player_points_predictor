import json
import requests
import pandas
import sqlite3

def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url) # HTTP GET request
    data = response.json() # convert JSON into dictionary
    players_dataframe = pandas.DataFrame(data['elements']) # use Pandas dataframe for

    return players_dataframe

def save_fpl_data(players_dataframe, fpl_database="fpl_new_data.db", table_name="fpl_players"):
    sqlConn = sqlite3.connect(fpl_database) # create connection to SQLite database
    players_dataframe.to_sql(table_name, sqlConn, if_exists='replace', index=False) # convert dataframe to SQL table, replace table if already exists
    sqlConn.close()

if __name__ == '__main__':
    players_dataframe = fetch_fpl_data()

    print(players_dataframe.head())

    save_fpl_data(players_dataframe)
    print("Data Saved Successfully")
