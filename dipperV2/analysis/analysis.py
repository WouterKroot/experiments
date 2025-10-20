# ToDo:
# per id create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)

# take the mean of each condition across ids and plot group average with standard error shading

# show that facilitation and inhibition effects can be modelled as multiplicative gain modulation multiplied by input contrast

#%%
from pathlib import Path
import sys
import scripts.functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from pathlib import Path
import os
# dynamic import
this_file = Path(__file__).resolve()
utils_path = this_file.parent.parent / 'utils'  # go up 2 levels to dipperV2 then into utils
sys.path.append(str(utils_path))
import utils

#%% Dynamic paths for data loading
output_path = this_file.parent.parent / 'Output'

exp_path = output_path / 'Exp'
baseline_path = exp_path / 'Baseline'
main_path = output_path / 'analysis'
# main_path = exp_path / 'Main'
eyelink_path = output_path / 'Eyelink'

#%%
baseline_df = utils.load_data(baseline_path)
main_df = utils.load_data(main_path)

#%% 
ids = main_df['id'].unique()
print(f"Found {len(ids)} participant(s): {ids}")

labels = main_df['label'].unique()
print(f"Found {len(labels)}") # conditions: {labels}")

#%% seperate dataframes per participant
participant_dfs = {participant_id: main_df[main_df['id'] == participant_id] for participant_id in ids}

#%% Clean data
fit_results = {}
for participant_id, df in participant_dfs.items():
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    participant_dfs[participant_id]['num_false_positives'] = num_false_positives
    participant_dfs[participant_id] = cleaned_df

    fit_result = smf.glm(
        formula='response ~ TC * FC * condition',
        data=cleaned_df,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        var_weights=cleaned_df['weight']
        ).fit()
    fit_results[participant_id] = fit_result


#%%
# for each id, create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)
# store results in a dataframe
# plot group average with standard error shading
