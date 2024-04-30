import pandas as pd
import numpy as np

# Load the datasets
results = pd.read_excel("Data/Raw/nfl_schedule.xlsx")
cap = pd.read_excel("Data/Raw/team_cap.xlsx")
status_p = pd.read_excel("Data/Raw/nfl_player_status.xlsx")

# Create a binary 'out' column based on the 'Active_Inactive' status
status_p['out'] = np.where(status_p['Active_Inactive'] == "Out", 1, 0)

# Filter for only regular season
status_p = status_p[status_p['Week'].between(1, 17)]

# Creating a checkpoint
status = status_p.copy()

# Calculate the average snap rate for each player when not 'out'
avg_snaps = status_p[status_p['out'] == 0].groupby('gsis_id')['Total_Snaps']\
    .mean().reset_index(name='avg_snaps')

# Merge avg_snaps back to the original DataFrame on gsis_id
status = pd.merge(status_p, avg_snaps, on='gsis_id', how='left')

# Filter the DataFrame for 'out' players only for position dummies
out_players = status[status['out'] == 1]

# Mapping of positions to buckets
position_to_bucket = {
    'RB': 'RB', 'SS': 'DB', 'LB': 'LB', 'DB': 'DB', 'DT': 'DL',
    'MLB': 'LB', 'FB': 'RB', 'DE': 'DL', 'TE': 'TE', 'CB': 'DB',
    'WR': 'WR', 'P': 'P', 'OLB': 'LB', 'OT': 'OL', 'OG': 'OL',
    'FS': 'DB', 'LS': 'LS', 'QB': 'QB', 'C': 'OL', 'ILB': 'LB',
    'NT': 'DL', 'K': 'K', 'SAF': 'DB', 'G': 'OL',
    'T': 'OL', 'DL': 'DL', 'PR': 'K'
}

# Apply the mapping to categorize the positions
out_players['position'] = out_players['position'].map(position_to_bucket)

# Create dummy variables for different player positions
# This will create a new column for each position with 1s and 0s
position_dummies = pd.get_dummies(out_players['position'])
out_players = pd.concat([out_players, position_dummies], axis=1)

# Create dummy variables for different player positions
positions = ['RB', 'DB', 'LB', 'DL', 'TE', 'WR', 'P', 'OL', 'LS', 'QB', 'K']

# Perform aggregation on 'out_players' DataFrame without considering 'out' status
# For each position, we use the max aggregation to indicate presence (1) or absence (0) of that position
aggregated_data = out_players.groupby(['Team', 'Season', 'Week']).agg({
    **{pos: 'max' for pos in positions},  # Use max to find if at least one player is present
    'avg_snaps': 'mean',  # Calculate the average snap rate
    'out':'sum'
}).reset_index()

# Convert boolean columns to integers
position_columns = ['RB', 'DB', 'LB', 'DL', 'TE', 'WR', 'P', 'OL', 'LS', 'QB', 'K']
aggregated_data[position_columns] = aggregated_data[position_columns].astype(int)

# Merge with the 'cap' DataFrame to include the age variable
status = pd.merge(aggregated_data, cap[['Team', 'Season', 'AVG AGE']], 
                  on=['Team', 'Season'], how='left').rename(columns={'AVG AGE': 'age'})

# Create Win-Loss Data
winners = results[['Week', 'Season', 'Winner/tie']].rename(
    columns={'Winner/tie': 'Team'})
winners['win'] = 1

losers = results[['Week', 'Season', 'Loser/tie']].rename(
    columns={'Loser/tie': 'Team'})
losers['win'] = 0

win_loss = pd.concat([winners, losers]).sort_values(by=['Team', 'Season', 'Week'])

# Merge injuries & average age to win_loss data
finaldata = pd.merge(win_loss, status, on=['Team', 'Season', 'Week'], how='left')

# Account for relocated teams
relocations = {
    "Washington Redskins": "Washington Football Team",
    "St. Louis Rams": "Los Angeles Rams",
    "San Diego Chargers": "Los Angeles Chargers"
}
finaldata['Team'] = finaldata['Team'].replace(relocations)

# Create win percentage by team-season & merge it to finaldata
team_season_wins = finaldata.groupby(['Team', 'Season'])['win'].mean()\
    .reset_index(name='win_perc')
team_season_wins['lag_win_perc'] = team_season_wins.groupby(['Team'])\
    ['win_perc'].shift(1)

finaldata = pd.merge(finaldata, team_season_wins, on=['Team', 'Season'], how='left')

# Lowercase variable names
finaldata.columns = [col.lower() for col in finaldata.columns]

# Filter out 2015
finaldata = finaldata[finaldata['season'] != 2012]

#Drop Nas
finaldata = finaldata.dropna(subset=['out'])

# Export to CSV
finaldata.to_csv("Data/Clean/final_data.csv", index=False)