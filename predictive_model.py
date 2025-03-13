import sqlite3
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(database="fpl_new_data.db", table="fpl_players"):
    sqlConn = sqlite3.connect(database)
    players_dataframe = pandas.read_sql_query(f"SELECT * FROM {table}", sqlConn)
    sqlConn.close()
    return players_dataframe

def prepare_data_attackers(players_dataframe): # forwards and midfielders together as they both earn their points through goals and assists
    attackers_dataframe = players_dataframe[players_dataframe['element_type'].isin([3,4])] # mids = 3, fwds = 4

    # features relevant for attacker points
    features = ['minutes', 'goals_scored', 'assists','points_per_game', 'yellow_cards','bonus', 'bps',
                'expected_goals', 'expected_assists','expected_goal_involvements'
                ]

    target = 'total_points' # value we are predicting

    # filter forwards dataframe to only include selected columns, drop rows w missing values
    filtered_dataframe = attackers_dataframe[features + [target]].dropna()

    X = filtered_dataframe[features]
    y = filtered_dataframe[target]

    return X, y



def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # split data in training and testing sets (80% train, 20% test)

    # Create and train random forest regressor with 100 trees
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test) # Make predictions on test set

    mae = mean_absolute_error(y_test, predictions) # Average absolute difference between predictions and actual values

    rmse = numpy.sqrt(mean_squared_error(y_test, predictions)) # Square root of the average squared differences (RMSE)

    r2 = r2_score(y_test, predictions) # Proportion of the variance in the target that's explained by the features


    # Print evaluation metrics
    print("Model Evaluation:")
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R² Score:", r2)

    return model

if __name__ == '__main__':

    # load the data from SQLite
    players_dataframe = load_data()
    print("Data loaded, shape:", players_dataframe.shape)

    # Prepare the data for attackers (forwards and midfielders)
    X, y = prepare_data_attackers(players_dataframe)
    print("Prepared data: features shape =", X.shape, "target shape =", y.shape)

    # Train the model and evaluate its performance
    model = train_and_evaluate_model(X, y)
