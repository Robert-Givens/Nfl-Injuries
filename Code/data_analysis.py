import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

finaldata = pd.read_csv("final_data.csv")

# Summary statistics - Python doesn't have a direct equivalent to `stargazer` for pretty tables in console output,
# but you can use `describe()` for basic summaries or the `summary_col` function from `statsmodels` for regression summaries.

# Relationship between wins & injuries
tab = finaldata.groupby('win')['out'].mean().reset_index()
sns.barplot(x='win', y='out', data=tab)
plt.show()

# Correlation table
print(finaldata[['win', 'out', 'age', 'lag_win_perc']].corr())

# Linear Models
lm_model1 = smf.ols('win ~ out', data=finaldata).fit()
lm_model2 = smf.ols('win ~ out + age', data=finaldata).fit()
lm_model3 = smf.ols('win ~ out + age + lag_win_perc', data=finaldata).fit()
lm_model4a = smf.ols('win ~ out + C(team)', data=finaldata).fit() # C() for categorical variables
lm_model5a = smf.ols('win ~ out + C(team) + age + lag_win_perc', data=finaldata).fit()
lm_model4b = smf.ols('win ~ out + C(team) + C(season)', data=finaldata).fit()
lm_model5b = smf.ols('win ~ out + C(team) + C(season) + age + lag_win_perc + offense.snap.rate + defense.snap.rate + special.teams.snap.rate' , data=finaldata).fit()

# Generalized Linear Models for logistic regression (assuming 'win' is a binary outcome)
glm_model1 = smf.glm('win ~ out', data=finaldata, family=sm.families.Binomial()).fit()
glm_model2 = smf.glm('win ~ out + age', data=finaldata, family=sm.families.Binomial()).fit()
glm_model3 = smf.glm('win ~ out + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4a = smf.glm('win ~ out + C(team)', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5a = smf.glm('win ~ out + C(team) + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()
glm_model4b = smf.glm('win ~ out + C(team) + C(season)', data=finaldata, family=sm.families.Binomial()).fit()
glm_model5b = smf.glm('win ~ out + C(team) + C(season) + age + lag_win_perc', data=finaldata, family=sm.families.Binomial()).fit()

# Assuming lm_model1, lm_model2, ..., glm_model5b are already fitted
models = {
    'LM1': lm_model1,
    'LM2': lm_model2,
    'LM3': lm_model3,
    'LM4a': lm_model4a,
    'LM5a': lm_model5a,
    'LM4b': lm_model4b,
    'LM5b': lm_model5b,
    'GLM1': glm_model1,
    'GLM2': glm_model2,
    'GLM3': glm_model3,
    'GLM4a': glm_model4a,
    'GLM5a': glm_model5a,
    'GLM4b': glm_model4b,
    'GLM5b': glm_model5b
}

def extract_model_results(models):
    results = {}
    for name, model in models.items():
        summary_df = model.summary2().tables[1]  # Extract the summary table as a DataFrame
        
        # Dynamically get column names for coefficients, standard errors, and p-values
        coef_col = 'Coef.' if 'Coef.' in summary_df.columns else 'coef'
        stderr_col = 'Std.Err.' if 'Std.Err.' in summary_df.columns else 'std err'
        pvalue_col = [col for col in summary_df.columns if 'P>' in col or 'p-value' in col][0]  # Adjust based on common patterns
        
        # Extract the relevant columns
        extracted_data = summary_df[[coef_col, stderr_col, pvalue_col]]
        results[name] = extracted_data.rename(columns={coef_col: 'Coef.', stderr_col: 'Std.Err.', pvalue_col: 'P>|t|'})
    
    return pd.concat(results, axis=1)

# Assuming your models are stored in the `models` dictionary
results_df = extract_model_results(models)
print(results_df)
