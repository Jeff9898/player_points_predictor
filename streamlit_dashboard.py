import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predictive_model import train_and_evaluate_model, predict_next_gameweek_points
from data_pipeline import load_all_data

def main():
    st.title("FPL Dashboard")
    st.write("The following data is for FPL Managers looking to gain an edge")

    # Load the data
    data = load_all_data()

    # Filter data for the 23/24 season (training set) and 24/25 season (predictions)
    training_data = data[data['season'] == '2023-24']
    prediction_data = data[data['season'] == '2024-25']

    # Train the model and get feature names, teams, and player names
    model, X_test, y_test, predictions, mae, rmse, r2, feature_names, teams, player_names = train_and_evaluate_model(training_data)

    # Points Predictor section
    st.divider()
    st.subheader("Predict Next Gameweek Points for a Player")

    # Dropdown menu to select a player
    selected_player = st.selectbox("Select a Player", player_names.unique())

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
                st.error("Insufficient data to make a prediction for this player.")
        else:
            st.error("Player not found in the database.")

    # Table of Players with most predicted points for next gw
    st.divider()
    st.subheader("Players with Most Predicted Points for Next Gameweek")

    if st.button("Generate Predictions for Top 20"):
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

    # Visualizations Section
    st.divider()
    st.subheader("Visualizations (2023-24 Season)")

    # Display top points scorers
    st.subheader("Top Points Scorers (2023-24 Season)")
    top_scorers = training_data.groupby('name')['total_points'].sum().reset_index().sort_values(by='total_points', ascending=False)
    st.dataframe(top_scorers.head(20))



#  ------- Visualizations --------

# Aggregate 2023-24 season data
# Group each player by 'name', and sum the values for each parameter

    grouped_df = (
        training_data
        .groupby('name', as_index=False)
        .agg({
            'goals_scored': 'sum',
            'expected_goals' : 'sum',
            'expected_goal_involvements': 'sum',
            'expected_assists' : 'sum',
            'total_points': 'sum'
        }))

    # Figure 1: Regression Plot for Expected Goal Involvements vs Total Points
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.regplot(data=grouped_df, x='expected_goal_involvements', y='total_points', ax=ax1)
    ax1.set_xlabel("Expected Goal Involvements (Season Total)")
    ax1.set_ylabel("Total Points (Season Total)")
    ax1.set_title("Expected Goal Involvements vs Total Points (Season Totals 2023-24)")
    st.pyplot(fig1)

    # Figure 3: Scatter Plot for Expected Goals vs Total Points
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=grouped_df, x='expected_goals', y='total_points', ax=ax2)
    ax2.set_xlabel("Expected Goals (Season Total)")
    ax2.set_ylabel("Total Points (Season Total)")
    ax2.set_title("Expected Goals vs Total Points (Season Totals 2023-24)")
    st.pyplot(fig2)

    # Figure 2: Histogram for Expected Assists vs Total Points
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.histplot(data=grouped_df, x='expected_assists', y='total_points', ax=ax3, bins=20)
    ax3.set_xlabel("Expected Assists (Season Total)")
    ax3.set_ylabel("Total Points (Season Total)")
    ax3.set_title("Expected Assists vs. Total Points (Season Totals 2023-24)")
    st.pyplot(fig3)


if __name__ == '__main__':
    main()


