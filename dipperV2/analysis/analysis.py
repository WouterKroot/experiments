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
from scipy.optimize import curve_fit

# dynamic import
this_file = Path(__file__).resolve()
utils_path = this_file.parent.parent / 'utils'  # go up 2 levels to dipperV2 then into utils
sys.path.append(str(utils_path))
import utils

#%% Dynamic paths for data loading
output_path = this_file.parent.parent.parent.parent / 'Data'

exp_path = output_path / 'Exp'
#baseline_path = exp_path / 'Baseline'
#main_path = output_path / 'analysis'
main_path = exp_path / 'Main'
summary_path = exp_path / 'Summary'
#eyelink_path = output_path / 'Eyelink'

#%%
#baseline_df = utils.load_data(baseline_path)
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
summary = []
summary_glm = []

threshVal = 0.8
expectedMin = 0.0

for participant_id, df in participant_dfs.items():
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    cleaned_df['label'] = cleaned_df['label'].astype(str).str.replace(r'^target.*', 'target', regex=True)
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
        intensities = df_label[df_label['TC']<= 0.06]['TC'].values
        responses = df_label[df_label['TC'] <= 0.06]['response'].values
    
        #combinedInten, combinedResp, combinedN = data.functionFromStaircase(
        #intensities, responses, bins=10)

        fit_psycho = data.FitWeibull(
        intensities, responses,
        expectedMin=expectedMin)
        
        smoothInt = np.linspace(-0.05, 0.1, 500)
        smoothResp_psycho = fit_psycho.eval(smoothInt)
        thresh_psycho = fit_psycho.inverse(threshVal)
        
        cond_result = smf.glm(
            formula='response ~ TC',
            data=df_label[df_label['TC']<= 0.06],
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            # var_weights=cleaned_df['weight']
            ).fit()
        glm_pred = cond_result.predict(pd.DataFrame({"TC": smoothInt}))
        
        pylab.figure(figsize=(6,4))
        pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
        pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
        pylab.plot(intensities, responses, 'o', label='Binned data')
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

        intensities_fit = df_label[df_label["TC"] <= 0.06]['TC'].values
        responses_fit = df_label[df_label["TC"] <= 0.06]['response'].values
        #print(intensities_fit)
        #print(responses_fit)

        flanker = df_label['flanker_multiplier'].unique()[0]
        condition = df_label['condition'].unique()[0]
    
        #combinedInten, combinedResp, combinedN = data.functionFromStaircase(
        #intensities, responses, bins=10)

        # -- psychopy FitLogistic -- 
        fit_psycho = data.FitWeibull(
        intensities_fit, responses_fit, expectedMin=0.0)
        
        smoothInt = np.linspace(-0.05, 0.1, 500)
        smoothResp_psycho = fit_psycho.eval(smoothInt)

        thresh_psycho = fit_psycho.inverse(threshVal)
        if label == 'target':
            for FC in cleaned_df['flanker_multiplier'].unique():
                summary.append({
                    "Participant": participant_id,
                    "condition": condition,
                    "flanker": FC,
                    "value": float(thresh_psycho)
                    })
        else:
            summary.append({
                    "Participant": participant_id,
                    "condition": condition,
                    "flanker": flanker,
                    "value": float(thresh_psycho)
                    })
        
        # -- glm CLogLog --
        fit_glm = smf.glm(
            formula='response ~ TC',
            data=df_label[df_label['TC'] <= 0.06],
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            # var_weights=cleaned_df['weight']
            ).fit()
        glm_pred = fit_glm.predict(pd.DataFrame({"TC": smoothInt}))
        eta = fit_glm.family.link(threshVal)
        thresh_glm = (eta - fit_glm.params['Intercept']) / fit_glm.params['TC']

        if label == 'target':
            for FC in cleaned_df['flanker_multiplier'].unique():
                summary_glm.append({
                    "Participant": participant_id,
                    "condition": condition,
                    "flanker": FC,
                    "value": thresh_glm
                    })
        else:
            summary_glm.append({
                "Participant": participant_id,
                "condition": condition,
                "flanker": flanker,
                "value": thresh_glm
                })
            
        pylab.figure(figsize=(6,4))
        pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
        pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
        pylab.plot(intensities_fit, responses_fit, 'o', label='Binned data')
        pylab.axvline(thresh_glm, color='k', linestyle=':', label=f'Logistic threshold={thresh_glm:.3f}')
        # pylab.ylim([0,1])
        pylab.xlabel('Stimulus Intensity (TC)')
        pylab.ylabel('P(Response=1)')
        pylab.legend()
        pylab.title(f'{label}: Psychometric Function Comparison')
        pylab.show()
    
    summary_data = pd.DataFrame(summary)
    summary_glm_data = pd.DataFrame(summary_glm)

    summary_data.to_csv(f"{summary_path}/psychopy/{participant_id}/summary_data.csv")
    summary_glm_data.to_csv(f"{summary_path}/glm/{participant_id}/summary_glm_data.csv")

    # -- summary plots per participant -- 
    plt.figure()
    plt.title(f'Summary psychopy: participant {participant_id}')
    plt.xlabel('Flanker contrast %Baseline')
    plt.ylabel('Target contrast')
    for cond in summary_data['condition'].unique():
        df_cond = summary_data[summary_data['condition'] == cond].sort_values('flanker')
        plt.plot(df_cond['flanker'], df_cond['value'], label = f'{cond}', marker = 'o')
    plt.legend()

    plt.figure()
    plt.title(f'Summary glm: participant {participant_id}')
    plt.xlabel('Flanker contrast %Baseline')
    plt.ylabel('Target contrast')
    for cond in summary_glm_data['condition'].unique():
        df_cond = summary_glm_data[summary_glm_data['condition'] == cond].sort_values('flanker')
        plt.plot(df_cond['flanker'], df_cond['value'], label = f'{cond}', marker = 'o')
    plt.legend()

#%%
# -- average summary plot over participants -- 

# - Psychopy version -
summary_df = utils.load_data(summary_path / 'psychopy')

mean = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].mean()
std_error = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].sem()

plt.figure()
plt.title(f'Summary glm')
plt.xlabel('Flanker contrast %Baseline')
plt.ylabel('Target contrast')

def fit_func(x, A,B,D,b,d):
    x = np.asarray(x, dtype=float)
    return A + B*x*np.exp(-b*x) - D*np.exp(-d*x) 

lower_bounds = [-np.inf, 0, 0, 0, 0]
upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
x_fit = np.linspace(min(mean.index), max(mean.index), 200)

for cond in mean['condition'].unique():
    cond_df = mean[mean['condition']==cond]
    data = plt.errorbar(cond_df['flanker'], cond_df['value'], yerr= std_error[std_error['condition']==cond]['value'], linestyle = '', marker ='o')
    y = cond_df['value'].to_list()
    popt, _ = curve_fit(fit_func, cond_df['flanker'], y, p0 = [min(y), (max(y)-min(y))/2, (max(y)-min(y))/2, 0.01, 0.01],maxfev =5000, bounds=(lower_bounds,upper_bounds))
    plt.plot(x_fit, fit_func(x_fit, *popt), color = data[0].get_color(), label = f'fit: {cond}') 

plt.legend()

# - GLM version -
summary_df = utils.load_data(summary_path / 'glm')

mean = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].mean()
std_error = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].sem()

plt.figure()
plt.title(f'Summary glm')
plt.xlabel('Flanker contrast %Baseline')
plt.ylabel('Target contrast')

def fit_func(x, A,B,D,b,d):
    x = np.asarray(x, dtype=float)
    return A + B*x*np.exp(-b*x) - D*np.exp(-d*x) 

lower_bounds = [-np.inf, 0, 0, 0, 0]
upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
x_fit = np.linspace(min(mean.index), max(mean.index), 200)

for cond in mean['condition'].unique():
    cond_df = mean[mean['condition']==cond]
    data = plt.errorbar(cond_df['flanker'], cond_df['value'], yerr= std_error[std_error['condition']==cond]['value'], linestyle = '', marker ='o')
    y = cond_df['value'].to_list()
    popt, _ = curve_fit(fit_func, cond_df['flanker'], y, p0 = [min(y), (max(y)-min(y))/2, (max(y)-min(y))/2, 0.01, 0.01],maxfev =5000, bounds=(lower_bounds,upper_bounds))
    plt.plot(x_fit, fit_func(x_fit, *popt), color = data[0].get_color(), label = f'fit: {cond}') 
plt.legend() 
    

#%%
def response_distribution_all_labels(df):
    """
    Shows the distribution of response=1 vs response=0 per intensity (TC) for every unique label.
    
    Parameters:
        df : pandas.DataFrame
            The cleaned dataframe with columns 'label', 'TC', and 'response'.
    """
    unique_labels = df['label'].unique()
    
    for label_name in unique_labels:
        print(f"\nLabel: {label_name}")
        # Filter for the label
        label_df = df[df['label'] == label_name]
        
        # Group by intensity (TC) and response, count occurrences
        distribution = label_df.groupby(['TC', 'response']).size().reset_index(name='count')
        
        # Pivot for easier reading
        distribution_pivot = distribution.pivot(index='TC', columns='response', values='count').fillna(0)
        distribution_pivot.columns = ['Response_0', 'Response_1']
        distribution_pivot = distribution_pivot.astype(int)
        
        print(distribution_pivot)
        

# Example usage:
response_distribution_all_labels(cleaned_df)