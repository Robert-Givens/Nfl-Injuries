import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML

finaldata = pd.read_csv("Data/Clean/final_data.csv")

# Group by 'win' and calculate mean for 'out'
tab = finaldata.groupby('ol')['win'].mean().reset_index()
sns.barplot(x='ol', y='win', data=tab)
plt.show()

finaldata['dl'].value_counts()

finaldata["out"].describe()

# Adjust bin size,outline on bars, match color theme for other plots
# Label more clearly
sns.distplot(finaldata["out"])
plt.title('Distribution of "out"')
plt.xlabel('"out" values')
plt.ylabel('Frequency')
plt.savefig("Output/position_win.png")
plt.show()

# Attempting a different approach to correctly aggregate and visualize the data without causing column mismatch errors

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
sns.barplot(x='Position', y='Average Win Percentage', hue='Injury Status', data=position_win_percs_df)

plt.title('Impact of Injuries on Win Percentage by Position')
plt.xlabel('Position')
plt.ylabel('Average Win Percentage')
plt.legend(title='Injury Status', labels=['No Injury', 'Injury Present'])
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("poster/images/position_win.png")
plt.show()


# List of positions for which we want to generate the heatmap
positions_list = ['rb', 'db', 'lb', 'dl', 'te', 'wr', 'p', 'ol',
                  'ls', 'qb', 'k']

# Initialize a matrix for average win rates
win_rates = np.zeros((len(positions_list), int(finaldata[positions_list].max().max()) + 1))
# Initialize a matrix for frequencies
frequencies = np.zeros_like(win_rates)

# Populate the matrices
for i, position in enumerate(positions_list):
    # Group by the number of injuries and calculate the average win rate
    win_rate_group = finaldata.groupby(position)['win'].mean()
    frequency_group = finaldata[position].value_counts()
    
    for num_injuries in win_rate_group.index:
        win_rates[i, int(num_injuries)] = win_rate_group[num_injuries]
        frequencies[i, int(num_injuries)] = frequency_group[num_injuries]

# Define the annotations for the heatmap cells: "average win rate (frequency)"
annotations = np.empty_like(win_rates, dtype=object)
for i in range(win_rates.shape[0]):
    for j in range(win_rates.shape[1]):
        annotations[i, j] = f"{win_rates[i, j]:.2f} ({int(frequencies[i, j])})" if frequencies[i, j] > 0 else ""

# Create the heatmap
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(win_rates, annot=annotations, fmt='', cmap='coolwarm', ax=ax,
            cbar_kws={'label': 'Average Win Rate'}, xticklabels=range(win_rates.shape[1]),
            yticklabels=positions_list)

# Aesthetic elements
# Creating the heatmap with combined annotations
plt.figure(figsize=(16, 12))
sns.heatmap(win_rates, annot=annotations, fmt='', cmap='coolwarm', cbar_kws={'label': 'Average Win Rate'})
plt.title('Average Win Percentage / Frequency by Position and Number of Injuries', fontsize=16)
plt.xlabel('Number of Injuries', fontsize=14)
plt.ylabel('Position', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Output/Heat_Map.png")
plt.show()
# Save Heat Map plot


# Linear models
lm_model1 = smf.ols('win ~ out', data=finaldata).fit()
lm_model2 = smf.ols('win ~ out + age', data=finaldata).fit()
lm_model3 = smf.ols('win ~ out + age + lag_win_perc', data=finaldata).fit()
lm_model4 = smf.ols('win ~ out + C(team) + age + lag_win_perc', data=finaldata).fit()
lm_model5 = smf.ols('win ~ out + C(team) + age + win_perc + lag_win_perc', data=finaldata).fit()

# Logistic models
glm_model1 = smf.glm('win ~ out', data=finaldata, family=sm.families.Binomial()).fit()
glm_model2 = smf.glm('win ~ out + age', data=finaldata, family=sm.families.Binomial()).fit()
glm_model3 = smf.glm('win ~ out + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4 = smf.glm('win ~ out + C(team) + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5 = smf.glm('win ~ out + C(team) + age + win_perc + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()

smf.glm('win ~ wr * qb', data=finaldata, family=sm.families.Binomial()).fit().summary()
# Loop through each position and fit the model
for position in positions_list:
    formula = f'win ~ {position}'
    model = smf.glm(formula, data=finaldata, family=sm.families.Binomial()).fit()
    print(f"Position: {position}")
    print(model.summary())
    print("\n")
# View models
dfoutput = summary_col([glm_model1, glm_model2, glm_model3, glm_model4, glm_model5], 
                       stars=True, 
                       model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'],
                       info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                  'Pseudo R2': lambda x: "{:.2f}".format(x.prsquared)},
                       regressor_order=['Intercept', 'out', 'age', 'lag_win_perc', 'win_perc'])

print(dfoutput)

