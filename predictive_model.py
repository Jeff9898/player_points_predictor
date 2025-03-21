import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate_model(training_data):
    # Prepare the data
    features = ['minutes', 'goals_scored', 'assists', 'expected_goals',
                'expected_assists', 'expected_goal_involvements', 'bonus', 'bps',
                'clean_sheets', 'saves', 'penalties_saved', 'yellow_cards', 'red_cards',
                'opponent_team', 'was_home']
    target = 'total_points'

    # Filter the dataframe to include only selected columns and drop rows with missing values
    filtered_df = training_data[features + [target, 'name', 'season']].dropna(subset=[target])

    # Convert categorical features (opponent_team and was_home) to numerical
    filtered_df['was_home'] = filtered_df['was_home'].astype(int)
    filtered_df = pd.get_dummies(filtered_df, columns=['opponent_team'], drop_first=True)

    # Separate features, target, and player names
    X = filtered_df.drop(columns=[target, 'name', 'season'])
    y = filtered_df[target]
    player_names = filtered_df['name']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Create and train the random forest regression model
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)

    # Save the feature names and the full list of teams
    feature_names = X_train.columns.tolist()
    teams = [col for col in feature_names if col.startswith('opponent_team_')]

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Print the evaluation metrics
    print("Model Evaluation:")
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R² Score:", r2)

    return model, X_test, y_test, predictions, mae, rmse, r2, feature_names, teams, player_names


def predict_next_gameweek_points(model, player_data, feature_names, teams):
    # Predict next gameweek points for a specific player using their historical data

    # Features used in the model
    features = ['minutes', 'goals_scored', 'assists', 'expected_goals',
                'expected_assists', 'expected_goal_involvements', 'bonus', 'bps',
                'clean_sheets', 'saves', 'penalties_saved', 'yellow_cards', 'red_cards',
                'opponent_team', 'was_home']

    # Filter the player data to include only the relevant features
    player_data_filtered = player_data[features].dropna()

    # Convert any categorical features (opponent_team and was_home) to numerical
    player_data_filtered['was_home'] = player_data_filtered['was_home'].astype(int)
    player_data_filtered = pd.get_dummies(player_data_filtered, columns=['opponent_team'], drop_first=True)

    # Ensure the prediction data has the same columns as the training data
    for col in feature_names:
        if col not in player_data_filtered.columns:
            player_data_filtered[col] = 0  # Add missing columns with a value of 0

    # Reorder the columns to match the training data
    player_data_filtered = player_data_filtered[feature_names]

    # Predict the points
    if not player_data_filtered.empty:
        predicted_points = model.predict(player_data_filtered)
        return np.mean(predicted_points)  # Return the average prediction
    else:
        return None  # Return None if no data is available
