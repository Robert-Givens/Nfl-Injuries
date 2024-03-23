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
tab = finaldata.groupby('win')['out'].mean().reset_index()
sns.barplot(x='win', y='out', data=tab)
plt.show()

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


# View models
dfoutput = summary_col([glm_model1, glm_model2, glm_model3, glm_model4, glm_model5], 
                       stars=True, 
                       model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'],
                       info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                  'Pseudo R2': lambda x: "{:.2f}".format(x.prsquared)},
                       regressor_order=['Intercept', 'out', 'age', 'lag_win_perc', 'win_perc'])

print(dfoutput)