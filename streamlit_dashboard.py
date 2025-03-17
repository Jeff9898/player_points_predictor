import streamlit
import sqlite3
import pandas
import matplotlib.pyplot as matplot
import seaborn

def load_data(database="fpl_new_data.db", table="fpl_players"):
    sqlConn = sqlite3.connect(database)
    players_dataframe = pandas.read_sql_query(f"SELECT * FROM {table}", sqlConn)
    sqlConn.close()
    return players_dataframe


def main():
    streamlit.title("FPL Dashboard") # Main title of dashboard

    players_dataframe = load_data() # Load FPL data from the database
    streamlit.subheader("Top Points Scorers")
    sorted_preview = players_dataframe.sort_values(by='total_points', ascending=False)[['web_name', 'total_points', 'goals_scored', 'assists']]
    streamlit.dataframe(sorted_preview.head(80))


    # Visualization 1 - Histogram of Total Points
    streamlit.subheader("Distribution of Total Points") # label the section
    fig1, axis1 = matplot.subplots(figsize=(8,6)) # create new figure
    seaborn.histplot(players_dataframe['total_points'].dropna(), bins=20, kde=True, ax=axis1) # create Histogram of total_points column
    axis1.set_xlabel("Total Points")
    axis1.set_ylabel("Frequency")
    axis1.set_title("Total Points Distribution")
    streamlit.pyplot(fig1) # Create the figure in the Streamlit app


    # Visualization 2 - Scatter Plot - Goals Scored Vs. Total Points
    streamlit.subheader("Goals Scored vs Total Points")
    fig2, axis2 = matplot.subplots(figsize=(8,6))
    seaborn.scatterplot(data=players_dataframe, x='goals_scored', y='total_points', ax=axis2)
    axis2.set_xlabel("Goals Scored")
    axis2.set_ylabel("Total Points")
    axis2.set_title("Scatter Plot: Goals Scored vs Total Points")
    streamlit.pyplot(fig2)

    # Convert xGI(float) and total points to numeric
    players_dataframe['expected_goal_involvements'] = pandas.to_numeric(
        players_dataframe['expected_goal_involvements'], errors="coerce")
    players_dataframe['total_points'] = pandas.to_numeric(
        players_dataframe['total_points'], errors="coerce")

    # Visualization 3 - Regression Plot (Regplot) - Expected Goal Involvements vs Total Points
    streamlit.subheader("Correlation: Expected Goal Involvements vs Total Points")
    fig3, axis3 = matplot.subplots(figsize=(8,6))
    seaborn.regplot(data=players_dataframe, x='expected_goal_involvements', y='total_points', ax=axis3)
    axis3.set_xlabel("Expected Goal Involvements(xGI)")
    axis3.set_ylabel("Total Points")
    axis3.set_title("Expected Goal Involvements(xGI) vs Total Points")
    streamlit.pyplot(fig3)


if __name__ == '__main__':
    main()





























