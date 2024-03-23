import pandas as pd
import numpy as np

# Load the datasets
results = pd.read_excel("Data/Raw/NFL Schedule.xlsx")
cap = pd.read_excel("Data/Raw/Team Cap.xlsx")
status_p = pd.read_excel("Data/Raw/NFL Player Status.xlsx")

# Create a binary 'out' column based on the 'Active_Inactive' status
status_p['out'] = np.where(status_p['Active_Inactive'] == "Out", 1, 0)

# Creating a checkpoint
status = status_p.copy()

# Calculate team-wide snap totals and max snaps by type
status['Team_Snaps'] = status.groupby(['Team', 'Season', 'Week'])\
    ['Total_Snaps'].transform('sum')
status['max_o_snaps'] = status.groupby(['Team', 'Season', 'Week'])\
    ['Offense_Snaps'].transform('max')
status['max_d_snaps'] = status.groupby(['Team', 'Season', 'Week'])\
    ['Defense_Snaps'].transform('max')
status['max_s_snaps'] = status.groupby(['Team', 'Season', 'Week'])\
    ['Special_Teams_Snaps'].transform('max')

# Calculate total game snaps by summing the max offense, defense, and special teams snaps
status['game_snaps'] = status['max_o_snaps'] + status['max_d_snaps'] + \
    status['max_s_snaps']

# Calculate snap rate as the proportion of total snaps to game snaps
status['snap_rate'] = status['Total_Snaps'] / status['game_snaps']

# Calculate the average snap rate for each player when not 'out'
status['avg_snap_rate'] = status[status['out'] == 0].groupby(['gsis_id'])\
    ['snap_rate'].transform('mean')

# Map the calculated average snap rates back to the original DataFrame
avg_snap_rate_mapping = status.groupby('gsis_id')['avg_snap_rate']\
    .first().to_dict()
status['avg_snap_rate'] = status['gsis_id'].map(avg_snap_rate_mapping)

# Remove duplicate records
status = status.drop_duplicates()

# Filter the DataFrame for 'out' players only for position dummies
out_players = status[status['out'] == 1]

# Mapping of positions to buckets
position_to_bucket = {
    'RB': 'RB', 'SS': 'DB', 'LB': 'LB', 'DB': 'DB', 'DT': 'DL',
    'MLB': 'LB', 'FB': 'RB', 'DE': 'DL', 'TE': 'TE', 'CB': 'DB',
    'WR': 'WR', 'P': 'P', 'OLB': 'LB', 'OT': 'OL', 'OG': 'OL',
    'FS': 'DB', 'LS': 'LS', 'QB': 'QB', 'C': 'OL', 'ILB': 'LB',
    'NT': 'DL', 'K': 'K', 'OL': 'OL', 'SAF': 'DB', 'G': 'OL',
    'T': 'OL', 'DL': 'DL', 'PR': 'PR'
}

# Apply the mapping to bucket the positions
out_players['position'] = out_players['position'].map(position_to_bucket)

# Create dummy variables for different player positions
positions = ['RB', 'DB', 'LB', 'DL', 'TE', 'WR', 'P', 'OL',
             'LS', 'QB', 'K', 'PR']
for position in positions:
    out_players[position] = np.where(out_players['position'] == position, 1, 0)

# Perform aggregation on 'out_players' DataFrame
aggregated_data = out_players.groupby(['Team', 'Season', 'Week']).agg({
    **{pos: 'sum' for pos in positions},  # Aggregate sum of position dummies for 'out' players
    'out': 'sum',  # Count the total 'out' players
    'avg_snap_rate': 'mean'  # Calculate the average snap rate for 'out' players
}).reset_index()

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
finaldata = finaldata[finaldata['season'] != 2015]

# Export to CSV
finaldata.to_csv("Data/Clean/final_data.csv", index=False)