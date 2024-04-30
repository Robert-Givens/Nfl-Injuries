import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

finaldata = pd.read_csv("data/clean/final_data.csv")

# Aggregate the win percentages by position and injury status directly within the plotting command
position_columns = ['rb', 'db', 'ol', 'lb', 'dl', 'qb','k']
position_win_percs = []

for position in position_columns:
    # Compute average win percentage for each injury status (0 or 1) in the position
    for status in [0, 1]:
        avg_win_perc = finaldata[finaldata[position] == status]['win'].mean()
        position_win_percs.append([position, status, avg_win_perc])

# Convert the aggregated data into a DataFrame
position_win_percs_df = pd.DataFrame(position_win_percs, columns=['Position', 'Injury Status', 'Average Win Percentage'])

# Plotting the corrected data
plt.figure(figsize=(14, 8))
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
sns.barplot(x='Position', y='Average Win Percentage', hue='Injury Status', data=position_win_percs_df)
plt.savefig("poster/images/position_win.png")
plt.show()


# Sum the injuries for the specified positions per week
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
plt.show()

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

# Logistic models
glm_model1 = smf.glm('win ~ out', data=finaldata, family=sm.families.Binomial()).fit()
glm_model2 = smf.glm('win ~ out + age', data=finaldata, family=sm.families.Binomial()).fit()
glm_model3 = smf.glm('win ~ out + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4 = smf.glm('win ~ out + C(team) + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5 = smf.glm('win ~ out + C(team) + age + win_perc + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
