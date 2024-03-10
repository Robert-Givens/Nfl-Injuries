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

# Summary statistics - Python doesn't have a direct equivalent to `stargazer` for pretty tables in console output,
# but you can use `describe()` for basic summaries or the `summary_col` function from `statsmodels` for regression summaries.

# Relationship between wins & injuries
tab = finaldata.groupby('win')['out'].mean().reset_index()
sns.barplot(x='win', y='out', data=tab)
plt.show()

# Correlation table
pd.set_option('display.max_rows', None)
print(finaldata[['win', 'out', 'age', 'lag_win_perc', 'offense_snap_rate', 'defense_snap_rate', 'special_teams_snap_rate']].corr())

# Linear Models
lm_model1 = smf.ols('win ~ out', data=finaldata).fit()
lm_model2 = smf.ols('win ~ out + age', data=finaldata).fit()
lm_model3 = smf.ols('win ~ out + age + lag_win_perc', data=finaldata).fit()
lm_model4a = smf.ols('win ~ out + C(team)', data=finaldata).fit()
lm_model5a = smf.ols('win ~ out + C(team) + age + lag_win_perc', data=finaldata).fit()
lm_model4b = smf.ols('win ~ out + C(team) + C(season)', data=finaldata).fit()
lm_model5b = smf.ols('win ~ out + C(team) + C(season) + age + lag_win_perc + offense_snap_rate + defense_snap_rate + special_teams_snap_rate' , data=finaldata).fit()

# Generalized Linear Models for logistic regression (assuming 'win' is a binary outcome)
glm_model1 = smf.glm('win ~ out', data=finaldata, family=sm.families.Binomial()).fit()
glm_model2 = smf.glm('win ~ out + age', data=finaldata, family=sm.families.Binomial()).fit()
glm_model3 = smf.glm('win ~ out + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4a = smf.glm('win ~ out + C(team)', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5a = smf.glm('win ~ out + C(team) + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4b = smf.glm('win ~ out + C(team) + C(season)', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5b = smf.glm('win ~ out + C(team) + C(season) + age + lag_win_perc + offense_snap_rate + defense_snap_rate + special_teams_snap_rate', data=finaldata, family=sm.families.Binomial()).fit()

# Assuming glm_model1, glm_model2, glm_model3, glm_model4a, glm_model5b are your model objects
dfoutput = summary_col([glm_model1,glm_model2, glm_model3, glm_model4a, glm_model5b], 
                       stars=True, 
                       model_names=['Model 1','Model 2','Model 3','Model 4a','Model 5b'],
                       info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'Pseudo R2':lambda x: "{:.2f}".format(x.prsquared)},
                       regressor_order=['Intercept', 'out', 'age', 'lag_win_perc', 'offense_snap_rate', 'defense_snap_rate', 'special_teams_snap_rate'])

print(dfoutput)