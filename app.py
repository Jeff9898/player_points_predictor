import json
import requests
import pandas
import sqlite3

def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url) # HTTP GET request
    data = response.json() # convert JSON into dictionary
    players_dataframe = pandas.DataFrame(data['elements'])

    return players_dataframe




