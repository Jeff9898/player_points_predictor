import seaborn as sns
import matplotlib.pyplot as plt
from data_pipeline import load_all_data

# Load combined dataset
data = load_all_data()

# Filter for the 23-24 season
season_data = data[data['season'] == '2023-24']

# Aggregate data by player to get the full-season totals
season_totals = season_data.groupby(['name', 'position', 'team'], as_index=False).agg({
    'total_points': 'sum',
    'goals_scored': 'sum',
    'expected_goals': 'sum',
    'minutes': 'sum',
    'expected_assists': 'sum'
})


# Visualization 1: Total Points distribution (Season Totals)
plt.figure(figsize=(8, 6))
sns.histplot(season_totals['total_points'], bins=30, kde=True)
plt.title('Distribution of Total Points (2023-24 Season Totals)')
plt.xlabel('Total Points (Season)')
plt.ylabel('Number of Players')
plt.show()


# Visualization 2: Goals Scored by Position (Season Totals)
plt.figure(figsize=(8, 6))
sns.boxplot(data=season_totals, x='position', y='goals_scored')
plt.title('Goals Scored by Player Position (Season Totals)')
plt.xlabel('Position')
plt.ylabel('Goals Scored (Season)')
plt.show()


# Visualization 3: Expected Goals vs Actual Goals Scored (Season Totals)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=season_totals, x='expected_goals', y='goals_scored', hue='position')
plt.title('Expected Goals vs Actual Goals Scored (Season Totals)')
plt.xlabel('Expected Goals (Season)')
plt.ylabel('Actual Goals Scored (Season)')
plt.legend(title='Position')
plt.show()


# Visualization 4: Top 10 Players by Total Points (Season Totals)
top_players = season_totals.sort_values(by='total_points', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_players['name'], x=top_players['total_points'], palette='viridis')
plt.title('Top 10 Players by Total Points (2023-24 Season Totals)')
plt.xlabel('Total Points (Season)')
plt.ylabel('Player')
plt.show()


# Visualization 5: Minutes Played vs Total Points (Season Totals)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=season_totals, x='minutes', y='total_points', hue='position')
plt.title('Minutes Played vs Total Points (Season Totals)')
plt.xlabel('Minutes Played (Season)')
plt.ylabel('Total Points (Season)')
plt.legend(title='Position')
plt.show()



