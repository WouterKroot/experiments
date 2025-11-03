# ToDo:
# per id create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)

# take the mean of each condition across ids and plot group average with standard error shading

# show that facilitation and inhibition effects can be modelled as multiplicative gain modulation multiplied by input contrast

#%%
from psychopy import data
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
import pylab
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

#%% Clean data from null trials and check n false positives
fit_results = {}

threshVal = 0.8
expectedMin = 0.0

for participant_id, df in participant_dfs.items():
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    #participant_dfs[participant_id]['num_false_positives'] = num_false_positives
    participant_dfs[participant_id] = cleaned_df

    interaction_result = smf.glm(
        formula='response ~ TC * FC * C(condition, Treatment(reference="target"))',
        data=cleaned_df,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        # var_weights=cleaned_df['weight']
    ).fit()

    main_result = smf.glm(
        formula='response ~ TC',
        data=cleaned_df,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        # var_weights=cleaned_df['weight']
    ).fit()

    fit_results[participant_id] = [interaction_result, main_result]
    
    # manual fitting using psychopy and GLM for main conditions 
    for label in cleaned_df['condition'].unique():
        df_label = cleaned_df[cleaned_df['condition'] == label]  # filter rows
        intensities = df_label['TC'].values
        responses = df_label['response'].values
    
        combinedInten, combinedResp, combinedN = data.functionFromStaircase(
        intensities, responses, bins=10)

        fit_psycho = data.FitLogistic(
        combinedInten, combinedResp,
        expectedMin=expectedMin,
        sems=1.0 / np.array(combinedN))
        
        smoothInt = np.linspace(intensities.min(), intensities.max(), 500)
        smoothResp_psycho = fit_psycho.eval(smoothInt)
        thresh_psycho = fit_psycho.inverse(threshVal)
        
        glm_pred = main_result.predict(pd.DataFrame({"TC": smoothInt}))
        
        pylab.figure(figsize=(6,4))
        pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
        pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
        pylab.plot(combinedInten, combinedResp, 'o', label='Binned data')
        pylab.axvline(thresh_psycho, color='k', linestyle=':', label=f'Logistic threshold={thresh_psycho:.3f}')
        # pylab.ylim([0,1])
        pylab.xlabel('Stimulus Intensity (TC)')
        pylab.ylabel('P(Response=1)')
        pylab.legend()
        pylab.title(f'{label}: Psychometric Function Comparison')
        pylab.show()
    
    # manual fitting using psychopy and GLM for individual flanker conditions
    for label in cleaned_df['label'].unique():
        df_label = cleaned_df[cleaned_df['label'] == label]  # filter rows
        intensities = df_label['TC'].values
        responses = df_label['response'].values
    
        combinedInten, combinedResp, combinedN = data.functionFromStaircase(
        intensities, responses, bins=10)

        fit_psycho = data.FitLogistic(
        combinedInten, combinedResp,
        expectedMin=expectedMin,
        sems=1.0 / np.array(combinedN))
        
        smoothInt = np.linspace(intensities.min(), intensities.max(), 500)
        smoothResp_psycho = fit_psycho.eval(smoothInt)
        thresh_psycho = fit_psycho.inverse(threshVal)
        
        glm_pred = main_result.predict(pd.DataFrame({"TC": smoothInt}))
        
        pylab.figure(figsize=(6,4))
        pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
        pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
        pylab.plot(combinedInten, combinedResp, 'o', label='Binned data')
        pylab.axvline(thresh_psycho, color='k', linestyle=':', label=f'Logistic threshold={thresh_psycho:.3f}')
        # pylab.ylim([0,1])
        pylab.xlabel('Stimulus Intensity (TC)')
        pylab.ylabel('P(Response=1)')
        pylab.legend()
        pylab.title(f'{label}: Psychometric Function Comparison')
        pylab.show()
    

    


#%%
for participant_id, fit_result in fit_results.items():
    print(f"Participant: {participant_id}, coefficients:")
    print(fit_result[0].summary())
    print(fit_result[1].summary())
    
    


