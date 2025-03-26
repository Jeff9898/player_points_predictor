import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from predictive_model import train_and_evaluate_model, predict_next_gameweek_points
from data_pipeline import load_all_data

import datetime
import uuid  # For anonymous session IDs

# Simple logging
with open("security.log", "a") as f:
    session_id = str(uuid.uuid4())[:8]  # Short random ID
    f.write(f"{datetime.datetime.now()}: Session {session_id} started\n")


def main():
    st.title("Fantasy Premier League Dashboard")
    st.write("This prediction tool uses up-to-date player performance data to generate accurate point predictions and help FPL managers make smarter decisions each gameweek of the 2024-25 season")

    # Load the data
    data = load_all_data()

    # Filter data for the 23/24 season (training set) and 24/25 season (predictions)
    training_data = data[data['season'] == '2023-24']
    prediction_data = data[data['season'] == '2024-25']

    # Train the model and get feature names, teams, and player names
    model, X_test, y_test, predictions, mae, rmse, r2, feature_names, teams, player_names = train_and_evaluate_model(training_data)

    # Points Predictor section
    st.divider()
    st.subheader("Upcoming Gameweek Player Points Predictor")

    # Only show players from the current season (2024-25) in the dropdown
    available_players = prediction_data['name'].unique()
    selected_player = st.selectbox("Select a Player", available_players)

    # Button to trigger the prediction
    if st.button("Predict Next Gameweek Points"):
        # Filter the players historical data from the 24-25 season
        player_data = prediction_data[prediction_data['name'] == selected_player]

        if not player_data.empty:
            # Predict next gameweek points for the selected player
            predicted_points = predict_next_gameweek_points(model, player_data, feature_names, teams)

            if predicted_points is not None:
                st.subheader(f"Predicted Next Gameweek Points for {selected_player}")
                st.write(f"Predicted Points for Next Gameweek: **{predicted_points:.2f}**")
            else:
                st.error("Insufficient data to make a prediction for this player")
        else:
            st.error("Selected player not available - This player is no longer in the Premier League")

    # Table of Players with most predicted points for next gw
    st.divider()
    st.subheader("Predicted Top Picks for Next Gameweek")

    if st.button("Generate Predictions (Top 20)"):
        # Create a list to store the predictions
        predictions_list = []

        # Loop through all players in the 2024-25 season data
        for player in player_names.unique():
            player_data = prediction_data[prediction_data['name'] == player]
            if not player_data.empty:
                predicted_points = predict_next_gameweek_points(model, player_data, feature_names, teams)
                if predicted_points is not None:
                    predictions_list.append({
                        'name': player,
                        'predicted_points': predicted_points
                    })

        # Convert the list to dataframe
        predictions_df = pd.DataFrame(predictions_list)

        # Sort the dataframe by predicted points in descending order
        predictions_df = predictions_df.sort_values(by='predicted_points', ascending=False)

        # display the table
        st.dataframe(predictions_df.head(20))  # Show top 20 players

    #  ------- Visualizations --------

    # Visualizations Section
    st.divider()
    st.divider()
    st.subheader("Visualizations (2023-24 Season)")
    st.write("The following visuals are based on historical data from the 2023-24 Premier League season")


    # Group by player and sum season totals for certain metrics
    season_totals = training_data.groupby(['name', 'position', 'team'], as_index=False).agg({
        'total_points': 'sum',
        'goals_scored': 'sum',
        'expected_goals': 'sum',
        'minutes': 'sum',
        'expected_assists': 'sum'
    })


    # Visualization 1: Expected Goals vs Actual Goals Scored (Season Totals)
    st.subheader("Expected Goals vs Actual Goals Scored")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=season_totals, x='expected_goals', y='goals_scored', hue='position', ax=ax1)
    ax1.set_title('Expected Goals vs Actual Goals Scored (2023-24 Season Totals)')
    ax1.set_xlabel('Expected Goals (Season)')
    ax1.set_ylabel('Actual Goals Scored (Season)')
    ax1.legend(title='Position')
    st.pyplot(fig1)

    # Visualization 2: Top 10 Players by Total Points (Season Totals)
    st.subheader("Top 10 Players by Total Points")
    top_players = season_totals.sort_values(by='total_points', ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_players['name'], x=top_players['total_points'], palette='viridis', ax=ax2)
    ax2.set_title('Top 10 Players by Total Points (2023-24 Season Totals)')
    ax2.set_xlabel('Total Points (Season)')
    ax2.set_ylabel('Player')
    st.pyplot(fig2)

    # Visualization 3: Points Distribution by Position
    st.subheader("Points Distribution by Position")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=season_totals, x='position', y='total_points', palette='viridis')
    ax3.set_title('FPL Points Distribution by Position (2023-24 Season)')
    ax3.set_xlabel('Player Position')
    ax3.set_ylabel('Total Points (Season)')
    st.pyplot(fig3)


if __name__ == '__main__':
    main()
