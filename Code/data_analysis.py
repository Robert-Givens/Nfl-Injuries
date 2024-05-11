# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Navigate up one directory from 'code' to the main project folder
project_dir = os.path.dirname(script_dir)

# Build paths to the data files
final_data_path = os.path.join(project_dir, 'data', 'clean', 'final_data.csv')

# Load the datasets
finaldata = pd.read_csv(final_data_path)

# Create columns for the positions
position_columns = ['rb', 'db', 'ol', 'lb', 'dl', 'qb','k']
# Create empty index to hold data
position_win_percs = []

# Loop that averages the win% when an injury occurs by position
for position in position_columns:
    # Compute average win percentage for each injury status (0 or 1) in the position
    for status in [0, 1]:
        avg_win_perc = finaldata[finaldata[position] == status]['win'].mean()
        position_win_percs.append([position, status, avg_win_perc])

# Convert the aggregated data into a DataFrame
position_win_percs_df = pd.DataFrame(position_win_percs, columns=
                                     ['Position', 'Injury Status', 'Average Win Percentage'])

# Plotting win% by position
plt.figure(figsize=(14, 8))
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
sns.barplot(x='Position', y='Average Win Percentage', hue='Injury Status', data=position_win_percs_df)
plt.savefig("poster/images/position_win.png")



# Sum weeks where position group is injured
positions = ['rb', 'db', 'ol', 'lb', 'dl', 'qb','k']
injuries_by_week = finaldata.groupby('week')[positions].sum()

# Reset index to make 'week' a column again for easier plotting
injuries_by_week.reset_index(inplace=True)

injuries_by_week.head()
# Set up the figure and axes for a large plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each position as a separate line
for position in positions:
    ax.plot(injuries_by_week['week'], injuries_by_week[position], label=position)

# Add some plot aesthetics
ax.set_xlabel('Week', fontsize=14)
ax.set_ylabel('Number of Injuries', fontsize=14)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(injuries_by_week['week'])
plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig("./poster/images/injuries_over_season.png")


# Assuming 'Win' is a binary variable that needs to be aggregated by 'Team'
# Sum wins by team
win_totals = finaldata.groupby('team')['win'].sum().reset_index()

# Assuming other columns are something like 'QB_Injuries', 'RB_Injuries', etc., which need to be summed and then ranked
# We need to sum these up per team
# This assumes your file has these exact columns for injuries, replace as necessary with actual columns
injury_columns = [col for col in finaldata.columns if 'Rank' in col and col != 'team']
finaldata['Total_Injuries'] = finaldata[injury_columns].sum(axis=1)
injury_totals = finaldata.groupby('team')['Total_Injuries'].sum().reset_index()

# Merge the win totals and injury totals
merged_data = pd.merge(win_totals, injury_totals, on='team')

# Sort data by total wins ascending and then by total injuries descending
sorted_data = merged_data.sort_values(by=['win', 'Total_Injuries'], ascending=[True, False])

# Set up plot for visualization
fig, ax = plt.subplots(figsize=(14, 8))  # Adjust size as needed
ax.axis('off')  # Hide axes

the_table = ax.table(cellText=sorted_data.values, colLabels=sorted_data.columns, loc='center', cellLoc='center')

# Adjust table styling
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust font size for better readability
the_table.scale(1.2, 1.2)  # Scale table to fit
# Optionally display the plot
plt.savefig("./poster/images/team_position_summary_final_ranked_updated.jpg")


position_aliases = {
    'rb': 'Running Back',
    'db': 'Defensive Back',
    'ol': 'Offensive Lineman',
    'lb': 'Linebacker',
    'dl': 'Defensive Lineman',
    'te': 'Tight End',
    'wr': 'Wide Receiver',
    'p': 'Punter',
    'ls': 'Long Snapper',
    'qb': 'Quarterback',
    'k': 'Kicker'
}

# Logistic models
# Assuming your data and model fitting code from previous steps
model_formula = 'win ~ ' + ' + '.join(position_aliases.keys())
glm_model = smf.glm(formula=model_formula, data=finaldata, family=sm.families.Binomial()).fit()

# Extracting and rounding coefficients and other stats
coefficients = glm_model.params.round(3)
p_values = glm_model.pvalues.round(3)
conf_int = glm_model.conf_int().round(3)
conf_int.columns = ['95% CI Lower', '95% CI Upper']

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Position': coefficients.index,
    'Coefficient': coefficients.values,
    'P-value': p_values.values,
    '95% CI Lower': conf_int['95% CI Lower'],
    '95% CI Upper': conf_int['95% CI Upper']
})

# Replace abbreviations with full names
summary_df['Position'] = summary_df['Position'].replace(position_aliases)

# If 'Intercept' is included and you want to remove it
summary_df = summary_df[summary_df['Position'] != 'Intercept']

# Print and visualize
print(summary_df)

# Save to CSV
summary_df.to_csv('./output/rounded_coefficients_full_names.csv')

# Visualize the table
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
ax.axis('off')  # Hide axes
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)  # Adjust font size for better readability
table.scale(1.2, 1.2)  # Scale table to fit
plt.savefig("./poster/images/regression_results_table.jpg")