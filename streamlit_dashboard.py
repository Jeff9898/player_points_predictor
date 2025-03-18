import numpy
import streamlit
import sqlite3
import pandas
import matplotlib.pyplot as matplot
import seaborn
from sklearn.metrics import mean_absolute_error

from predictive_model import load_data, prepare_data_attackers, train_and_evaluate_model


def main():
    streamlit.title("FPL Dashboard") # Main title of dashboard
    streamlit.write("The following data is specifically for Fantasy Premier League Forwards and Midfielders")

    players_dataframe = load_data(database="fpl_new_data.db", table="fpl_players") # Load FPL data from the database

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


    #
    # Predictive Model Integration
    #
    streamlit.divider()

    streamlit.subheader("Predictive Model: Actual vs Predicted Total Points")
    streamlit.write("The following section contains predictive data from a Random Forest Regression machine learning model")

    # Prepare data for attackers (forwards and midfielders) using imported function
    X, y, web_names = prepare_data_attackers(players_dataframe)

    # Train the model and evaluate its performance using imported function
    model, X_test, y_test, predictions, mae, rmse, r2 = train_and_evaluate_model(X, y)

    mae = round(mae, 3)
    rmse = round(rmse, 3)
    r2 = round(r2, 3)


    streamlit.subheader("Model Evaluation Metrics")
    col1, col2, col3 = streamlit.columns(3)
    col1.metric(label="MAE", value=mae)
    col2.metric(label="RMSE", value=rmse)
    col3.metric(label="R² Score", value=r2)


    # Create a dataframe to compare actual vs predicted total points from the test set
    results_dataframe = pandas.DataFrame({
        'web_name': web_names.loc[X_test.index],
        'Actual Total Points': y_test,
        'Predicted Total Points': predictions
    })

    results_dataframe = results_dataframe.sort_values(by='Actual Total Points', ascending=False)

    streamlit.subheader("Actual vs Predicted Total Points (Test Set)")
    streamlit.dataframe(results_dataframe.head(50))

    # Visualization 4: Scatter Plot for Actual vs Predicted Total Points
    streamlit.subheader("Scatter Plot: Actual vs Predicted Total Points")
    fig4, axis4 = matplot.subplots(figsize=(8,6))
    seaborn.scatterplot(data=results_dataframe, x='Actual Total Points', y='Predicted Total Points', ax=axis4)
    axis4.set_xlabel("Actual Total Points")
    axis4.set_ylabel("Predicted Total Points")
    axis4.set_title("Actual vs Predicted Total Points")
    streamlit.pyplot(fig4)




if __name__ == '__main__':
    main()





