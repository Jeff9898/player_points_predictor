import sqlite3
import matplotlib
import matplotlib.pyplot as matplot
import seaborn
import pandas

def load_data(database="fpl_new_data.db", table="fpl_players"):
    sqlConn = sqlite3.connect(database) # create connection to SQLite database
    players_dataframe = pandas.read_sql_query(f"SELECT * FROM {table}", sqlConn)
    sqlConn.close()
    return players_dataframe

if __name__ == '__main__':
    players_dataframe = load_data() # load the data from the database

    matplot.figure(figsize=(8,6))
    seaborn.histplot(data=players_dataframe['total_points'].dropna(), bins=20, kde=True) # histogram for player points, dropna() in case of missing values
    matplot.xlabel("Total Points")
    matplot.ylabel("Frequency")
    matplot.title("Total Points Distribution")
    matplot.show()

    # print statements for understanding the data structure and debugging
    print("Total players:", players_dataframe.shape[0])
    print("Dataset Shape:", players_dataframe.shape)
    print("Columns:", players_dataframe.columns.tolist())
    print("\nFirst 5 Rows:")
    print(players_dataframe.head())