import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_pipeline import load_all_data
'''
def load_data(database="fpl_new_data.db", table="fpl_players"):
    sqlConn = sqlite3.connect(database) # create connection to SQLite database
    players_dataframe = pandas.read_sql_query(f"SELECT * FROM {table}", sqlConn)
    sqlConn.close()
    return players_dataframe
'''

def main():
    # Load the data
    data = load_all_data()

    # Filter data for the 2023-24 season
    season_data = data[data['season'] == '2023-24']




if __name__ == '__main__':
    main()