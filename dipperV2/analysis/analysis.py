# ToDo:
# per id create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)
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

test = True
#%%
#Dynamic paths for data loading
#output_path = this_file.parent.parent.parent.parent / 'Data'
data_path = this_file.parent.parent / 'Output'
output_path = this_file.parent / 'Output' / 'dipperV2'

if test == True:
    exp_path = data_path / 'Test'
else:
    exp_path = data_path / 'Exp'
    
baseline_path = exp_path / 'Baseline'
main_path = exp_path / 'Main'
eyelink_path = data_path / 'Eyelink'

#%%
baseline_df = utils.load_data(baseline_path)
main_df = utils.load_data(main_path)
#%% 
ids = main_df['id'].unique()
print(f"Found {len(ids)} participant(s): {ids}")

labels = main_df['label'].unique()
print(f"Found {len(labels)}") # conditions: {labels}")

#%% seperate dataframes per participant

participant_dfs = {}
for pid in ids:
    # Slice baseline and main data
    base = baseline_df[baseline_df['id'] == pid].copy()
    main = main_df[main_df['id'] == pid].copy()

    # Add session labels for clarity
    base['session'] = 'baseline'
    main['session'] = 'main'

    # Combine both
    combined = pd.concat([base, main], ignore_index=True)

    # Store in dict
    participant_dfs[pid] = {
        'baseline': base,
        'main': main,
        'combined': combined}
#%%
# Investigate the raw number of responses and calculate the proportion
for participant_id, dfs in participant_dfs.items():
    df = dfs['combined'].copy()
    cleaned_df, false_positives = utils.clean_df(df)
    all_distributions, combined_df = utils.response_distribution(cleaned_df, false_positives, max_val=0.05, n_bins=10) # bins are to be samller than the smalles log stepsize, is 0.005
   
    participant_dfs[participant_id]['cleaned_df'] = cleaned_df
    participant_dfs[participant_id]['false_positives'] = false_positives
    participant_dfs[participant_id]['response_summary'] = all_distributions
    
#%%
threshVal = 0.7  # probability threshold
fit_results = {}
thresholds = {}  # store threshold values

for participant_id, dfs in participant_dfs.items():
    print(f"\n=== Participant {participant_id} ===")

    response_summary = dfs['response_summary']
    fit_results[participant_id] = {}
    thresholds[participant_id] = {}

    participant_output_path = os.path.join(output_path, str(participant_id))
    os.makedirs(participant_output_path, exist_ok=True)

    for label_name, df_label in response_summary.items():
        if df_label.empty:
            continue

        df_label = df_label.copy()
        df_label['TC_center'] = df_label['TC_bin'].apply(
            lambda x: x.mid if hasattr(x, 'mid') else np.nan
        )
        df_label = df_label.dropna(subset=['TC_center'])
        df_label['Total'] = df_label['Response_0'] + df_label['Response_1']

        glm_data = df_label[['TC_center', 'Adjusted_yes', 'Total']].copy()
        glm_data = glm_data.dropna()
        glm_data = glm_data[glm_data['Total'] > 0]
        if glm_data.empty:
            continue

        glm_model = smf.glm(
            formula='Adjusted_yes ~ TC_center',
            data=glm_data,
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            freq_weights=glm_data['Total']
        ).fit()

        fit_results[participant_id][label_name] = glm_model

        eta = glm_model.family.link(threshVal)
        thresh_glm = (eta - glm_model.params['Intercept']) / glm_model.params['TC_center']
        thresholds[participant_id][label_name] = thresh_glm 

        smoothInt = np.linspace(glm_data['TC_center'].min(),
                                glm_data['TC_center'].max(), 200)
        glm_pred = glm_model.predict(pd.DataFrame({'TC_center': smoothInt}))

        plt.figure(figsize=(6, 4))
        plt.plot(smoothInt, glm_pred, label='GLM (CLogLog) Fit', lw=2)
        plt.plot(df_label['TC_center'], df_label['Adjusted_yes'], 'o',
                 label='Adjusted Data')
        plt.axvline(thresh_glm, color='k', linestyle=':',
                    label=f'GLM Threshold (P={threshVal}) = {thresh_glm:.3f}')
        plt.xlabel('Stimulus Intensity (TC)')
        plt.ylabel('Adjusted P(Response=1)')
        plt.title(f'{participant_id} | {label_name}')
        plt.legend()
        plt.tight_layout()

        safe_label = label_name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(participant_output_path, f"{safe_label}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved GLM plot for {label_name} → {save_path}")
        print(f"{label_name} GLM summary:")
        print(glm_model.summary())
        print(f"{label_name} GLM threshold (P={threshVal}) = {thresh_glm:.3f}")

#%%    
plot_data = []

for participant_id, dfs in participant_dfs.items():
    cleaned_df = dfs['cleaned_df']
    
    # Get unique labels and their corresponding flanker contrasts (FC)
    unique_labels = cleaned_df['label'].unique()
    
    for label in unique_labels:
        # Get flanker contrasts for this label
        label_info = cleaned_df[cleaned_df['label'] == label][['condition', 'FC']].drop_duplicates()
        
        for _, row in label_info.iterrows():
            condition_name = row['condition']
            flanker_contrast = row['FC']
            
            # Get the threshold value for this condition
            if condition_name in thresholds[participant_id]:
                thresh_val = thresholds[participant_id][condition_name]
                plot_data.append({
                    'participant': participant_id,
                    'label': label,
                    'flanker': flanker_contrast,
                    'thresh_glm': thresh_val
                })

# Convert to DataFrame
plot_df = pd.DataFrame(plot_data)

# Optional: compute mean ± SEM across participants for each label × flanker
summary_df = plot_df.groupby(['label', 'flanker'])['thresh_glm'].agg(['mean', 'sem']).reset_index()

# Plot each label
plt.figure(figsize=(8,5))
labels = summary_df['label'].unique()
for lbl in labels:
    df_lbl = summary_df[summary_df['label'] == lbl].sort_values('flanker')
    plt.errorbar(df_lbl['flanker'], df_lbl['mean'], yerr=df_lbl['sem'],
                 fmt='o-', lw=2, capsize=5, label=lbl)

plt.xlabel('Flanker Contrast (FC)')
plt.ylabel('GLM Threshold (TC at P=0.7)')
plt.title('GLM Thresholds Across Flanker Contrasts')
plt.xticks(sorted(cleaned_df['FC'].unique()))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# fit_results = {}

# for participant_id, dfs in participant_dfs.items():
#     print(f"\nParticipant {participant_id}")

#     response_summary = dfs['response_summary']

#     fit_results[participant_id] = {}

#     participant_output_path = os.path.join(output_path, str(participant_id))
#     os.makedirs(participant_output_path, exist_ok=True)
    
#     for label_name, df_label in response_summary.items():
#         if df_label.empty:
#             continue

#         # Convert TC_bin to a numeric midpoint
#         df_label = df_label.copy()
#         df_label['TC_center'] = df_label['TC_bin'].apply(
#             lambda x: x.mid if hasattr(x, 'mid') else np.nan
#         )

#         # Drop bins without counts
#         df_label = df_label.dropna(subset=['TC_center'])

#         # Compute total trials per bin
#         df_label['Total'] = df_label['Response_0'] + df_label['Response_1']

#         # but also provide weights = total number of trials
#         glm_data = df_label[['TC_center', 'Adjusted_yes', 'Total']].copy()
#         glm_data = glm_data.dropna()

#         # Avoid bins with no variance or total=0
#         glm_data = glm_data[glm_data['Total'] > 0]
#         if glm_data.empty:
#             continue

#         # ---------------------------------
#         # Fit GLM to adjusted proportions
#         # ---------------------------------
#         glm_model = smf.glm(
#             formula='Adjusted_yes ~ TC_center',
#             data=glm_data,
#             family=sm.families.Binomial(link=sm.families.links.CLogLog()),
#             freq_weights=glm_data['Total']
#         ).fit()

#         # Store results
#         fit_results[participant_id][label_name] = glm_model

#         # ---------------------------------
#         # Plot comparison
#         # ---------------------------------
#         smoothInt = np.linspace(glm_data['TC_center'].min(),
#                                 glm_data['TC_center'].max(), 200)
#         glm_pred = glm_model.predict(pd.DataFrame({'TC_center': smoothInt}))

#         pylab.figure(figsize=(6, 4))
#         pylab.plot(smoothInt, glm_pred, label='GLM (CLogLog) Fit', lw=2)
#         pylab.plot(df_label['TC_center'], df_label['Adjusted_yes'], 'o',
#                    label='Adjusted Data')
#         pylab.xlabel('Stimulus Intensity (TC)')
#         pylab.ylabel('Adjusted P(Response=1)')
#         pylab.title(f'{participant_id} | {label_name}')
#         pylab.legend()
#         pylab.tight_layout()
#         pylab.show()

#         print(f"{label_name} GLM summary:")
#         print(glm_model.summary())
        
#         safe_label = label_name.replace("/", "_").replace("\\", "_")
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         save_path = os.path.join(participant_output_path, f"{safe_label}.png")
#         plt.savefig(save_path, dpi=300)
#         plt.close()

#         print(f"Saved GLM plot for {label_name} → {save_path}")

# Calculate per participant the proportions yes/no per bin per  and adjust it with kapadia
 
# Plot the adjusted proportions per bin

# Fit the glm on the adjust proportion per bin, show differences Weilbull and log

# Take out the 0.75 point

# Aggregate over participants

# Plot per condition, threshold per flanker contrast

 

#!!! Fix first adjusted p prime voor fit en plot de fit voor Cloglog op de adjusted p prime using aggegqrated data glm

#%% old code 
# #%% Clean data from null trials and check n false positives
# fit_results = {}
# summary = []
# summary_glm = []

# threshVal = 0.7
# expectedMin = 0.0

# for participant_id, dfs in participant_dfs.items():
#     df = dfs['combined'].copy()
    
#     cleaned_df, sum_false_positive, false_positive_ratio = scripts.functions.clean_df(df)
#     cleaned_df['label'] = cleaned_df['label'].astype(str).str.replace(r'^target.*', 'target', regex=True)
#     #participant_dfs[participant_id]['num_false_positives'] = num_false_positives
#     # participant_dfs[participant_id] = cleaned_df

#     interaction_result = smf.glm(
#         formula='response ~ TC * FC * C(condition, Treatment(reference="target"))',
#         data=cleaned_df,
#         family=sm.families.Binomial(link=sm.families.links.CLogLog()),
#         # var_weights=cleaned_df['weight']
#     ).fit()

#     main_result = smf.glm(
#         formula='response ~ TC',
#         data=cleaned_df,
#         family=sm.families.Binomial(link=sm.families.links.CLogLog()),
#         # var_weights=cleaned_df['weight']
#     ).fit()

#     fit_results[participant_id] = [interaction_result, main_result]
    
#     # manual fitting using psychopy and GLM for main conditions 
#     for label in cleaned_df['condition'].unique():
#         df_label = cleaned_df[cleaned_df['condition'] == label]  # filter rows
#         intensities = df_label['TC'].values
#         responses = df_label['response'].values
    
#         #combinedInten, combinedResp, combinedN = data.functionFromStaircase(
#         #intensities, responses, bins=10)

#         fit_psycho = data.FitWeibull(
#         intensities, responses,
#         expectedMin=expectedMin)
        
#         smoothInt = np.linspace(-0.05, 0.1, 500)
#         smoothResp_psycho = fit_psycho.eval(smoothInt)
#         thresh_psycho = fit_psycho.inverse(threshVal)
        
#         cond_result = smf.glm(
#             formula='response ~ TC',
#             data=df_label,
#             family=sm.families.Binomial(link=sm.families.links.CLogLog()),
#             # var_weights=cleaned_df['weight']
#             ).fit()
#         glm_pred = cond_result.predict(pd.DataFrame({"TC": smoothInt}))
        
#         pylab.figure(figsize=(6,4))
#         pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
#         pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
#         pylab.plot(intensities, responses, 'o', label='Binned data')
#         pylab.axvline(thresh_psycho, color='k', linestyle=':', label=f'Logistic threshold={thresh_psycho:.3f}')
#         # pylab.ylim([0,1])
#         pylab.xlabel('Stimulus Intensity (TC)')
#         pylab.ylabel('P(Response=1)')
#         pylab.legend()
#         pylab.title(f'{label}: Psychometric Function Comparison')
#         pylab.show()
    
#     # manual fitting using psychopy and GLM for individual flanker conditions
#     for label in cleaned_df['label'].unique():
#         df_label = cleaned_df[cleaned_df['label'] == label]  # filter rows
#         intensities = df_label['TC'].values
#         responses = df_label['response'].values

#         # intensities_fit = df_label[df_label["TC"] <= 0.06]['TC'].values
#         # responses_fit = df_label[df_label["TC"] <= 0.06]['response'].values
        
#         intensities_fit = df_label["TC"].values
#         responses_fit = df_label['response'].values
        
#         #print(intensities_fit)
#         #print(responses_fit)

#         flanker = df_label['flanker_multiplier'].unique()[0]
#         condition = df_label['condition'].unique()[0]
    
#         #combinedInten, combinedResp, combinedN = data.functionFromStaircase(
#         #intensities, responses, bins=10)

#         # -- psychopy FitLogistic -- 
#         fit_psycho = data.FitWeibull(
#         intensities_fit, responses_fit, expectedMin=0.0)
        
#         smoothInt = np.linspace(-0.05, 0.1, 500)
#         smoothResp_psycho = fit_psycho.eval(smoothInt)

#         thresh_psycho = fit_psycho.inverse(threshVal)
#         if label == 'target':
#             for FC in cleaned_df['flanker_multiplier'].unique():
#                 summary.append({
#                     "Participant": participant_id,
#                     "condition": condition,
#                     "flanker": FC,
#                     "value": float(thresh_psycho)
#                     })
#         else:
#             summary.append({
#                     "Participant": participant_id,
#                     "condition": condition,
#                     "flanker": flanker,
#                     "value": float(thresh_psycho)
#                     })
        
#         # -- glm CLogLog --
#         fit_glm = smf.glm(
#             formula='response ~ TC',
#             data=df_label,
#             family=sm.families.Binomial(link=sm.families.links.CLogLog()),
#             # var_weights=cleaned_df['weight']
#             ).fit()
#         glm_pred = fit_glm.predict(pd.DataFrame({"TC": smoothInt}))
#         eta = fit_glm.family.link(threshVal)
#         thresh_glm = (eta - fit_glm.params['Intercept']) / fit_glm.params['TC']

#         if label == 'target':
#             for FC in cleaned_df['flanker_multiplier'].unique():
#                 summary_glm.append({
#                     "Participant": participant_id,
#                     "condition": condition,
#                     "flanker": FC,
#                     "value": thresh_glm
#                     })
#         else:
#             summary_glm.append({
#                 "Participant": participant_id,
#                 "condition": condition,
#                 "flanker": flanker,
#                 "value": thresh_glm
#                 })
            
#         pylab.figure(figsize=(6,4))
#         pylab.plot(smoothInt, smoothResp_psycho, label='PsychoPy Logistic Fit', lw=2)
#         pylab.plot(smoothInt, glm_pred, label='GLM CLogLog Fit', lw=2, linestyle='--')
#         pylab.plot(intensities_fit, responses_fit, 'o', label='Binned data')
#         pylab.axvline(thresh_glm, color='k', linestyle=':', label=f'Logistic threshold={thresh_glm:.3f}')
#         # pylab.ylim([0,1])
#         pylab.xlabel('Stimulus Intensity (TC)')
#         pylab.ylabel('P(Response=1)')
#         pylab.legend()
#         pylab.title(f'{label}: Psychometric Function Comparison')
#         pylab.show()
    
#     summary_data = pd.DataFrame(summary)
#     summary_glm_data = pd.DataFrame(summary_glm)

#     summary_data.to_csv(f"{summary_path}/psychopy/{participant_id}/summary_data.csv")
#     summary_glm_data.to_csv(f"{summary_path}/glm/{participant_id}/summary_glm_data.csv")

#     # -- summary plots per participant -- 
#     plt.figure()
#     plt.title(f'Summary psychopy: participant {participant_id}')
#     plt.xlabel('Flanker contrast %Baseline')
#     plt.ylabel('Target contrast')
#     for cond in summary_data['condition'].unique():
#         df_cond = summary_data[summary_data['condition'] == cond].sort_values('flanker')
#         plt.plot(df_cond['flanker'], df_cond['value'], label = f'{cond}', marker = 'o')
#     plt.legend()

#     plt.figure()
#     plt.title(f'Summary glm: participant {participant_id}')
#     plt.xlabel('Flanker contrast %Baseline')
#     plt.ylabel('Target contrast')
#     for cond in summary_glm_data['condition'].unique():
#         df_cond = summary_glm_data[summary_glm_data['condition'] == cond].sort_values('flanker')
#         plt.plot(df_cond['flanker'], df_cond['value'], label = f'{cond}', marker = 'o')
#     plt.legend()

#%%
# # -- average summary plot over participants -- 

# # - Psychopy version -
# summary_df = utils.load_data(summary_path / 'psychopy')

# mean = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].mean()
# std_error = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].sem()

# plt.figure()
# plt.title(f'Summary glm')
# plt.xlabel('Flanker contrast %Baseline')
# plt.ylabel('Target contrast')

# def fit_func(x, A,B,D,b,d):
#     x = np.asarray(x, dtype=float)
#     return A + B*x*np.exp(-b*x) - D*np.exp(-d*x) 

# lower_bounds = [-np.inf, 0, 0, 0, 0]
# upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
# x_fit = np.linspace(min(mean.index), max(mean.index), 200)

# for cond in mean['condition'].unique():
#     cond_df = mean[mean['condition']==cond]
#     data = plt.errorbar(cond_df['flanker'], cond_df['value'], yerr= std_error[std_error['condition']==cond]['value'], linestyle = '', marker ='o')
#     y = cond_df['value'].to_list()
#     popt, _ = curve_fit(fit_func, cond_df['flanker'], y, p0 = [min(y), (max(y)-min(y))/2, (max(y)-min(y))/2, 0.01, 0.01],maxfev =5000, bounds=(lower_bounds,upper_bounds))
#     plt.plot(x_fit, fit_func(x_fit, *popt), color = data[0].get_color(), label = f'fit: {cond}') 

# plt.legend()

# # - GLM version -
# summary_df = utils.load_data(summary_path / 'glm')

# mean = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].mean()
# std_error = summary_df.groupby(['condition', 'flanker'], as_index=False)['value'].sem()

# plt.figure()
# plt.title(f'Summary glm')
# plt.xlabel('Flanker contrast %Baseline')
# plt.ylabel('Target contrast')

# def fit_func(x, A,B,D,b,d):
#     x = np.asarray(x, dtype=float)
#     return A + B*x*np.exp(-b*x) - D*np.exp(-d*x) 

# lower_bounds = [-np.inf, 0, 0, 0, 0]
# upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
# x_fit = np.linspace(min(mean.index), max(mean.index), 200)

# for cond in mean['condition'].unique():
#     cond_df = mean[mean['condition']==cond]
#     data = plt.errorbar(cond_df['flanker'], cond_df['value'], yerr= std_error[std_error['condition']==cond]['value'], linestyle = '', marker ='o')
#     y = cond_df['value'].to_list()
#     popt, _ = curve_fit(fit_func, cond_df['flanker'], y, p0 = [min(y), (max(y)-min(y))/2, (max(y)-min(y))/2, 0.01, 0.01],maxfev =5000, bounds=(lower_bounds,upper_bounds))
#     plt.plot(x_fit, fit_func(x_fit, *popt), color = data[0].get_color(), label = f'fit: {cond}') 
# plt.legend() 
    

#%%
# def response_distribution_all_labels(df):
#     """
#     Shows the distribution of response=1 vs response=0 per intensity (TC) for every unique label.
    
#     Parameters:
#         df : pandas.DataFrame
#             The cleaned dataframe with columns 'label', 'TC', and 'response'.
#     """
#     unique_labels = df['label'].unique()
    
#     for label_name in unique_labels:
#         print(f"\nLabel: {label_name}")
#         # Filter for the label
#         label_df = df[df['label'] == label_name]
        
#         # Group by intensity (TC) and response, count occurrences
#         distribution = label_df.groupby(['TC', 'response']).size().reset_index(name='count')
        
#         # Pivot for easier reading
#         distribution_pivot = distribution.pivot(index='TC', columns='response', values='count').fillna(0)
#         distribution_pivot.columns = ['Response_0', 'Response_1']
#         distribution_pivot = distribution_pivot.astype(int)
        
#         print(distribution_pivot)
        
# import pandas as pd
# import numpy as np

# def response_distribution_all_labels(df, n_bins=20):
#     """
#     Shows the distribution of response=1 vs response=0 per binned intensity (TC) for every unique label.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Must contain columns 'label', 'TC', and 'response'.
#     n_bins : int
#         Number of bins to divide the TC range into.
#     """
#     unique_labels = df['label'].unique()
    
#     # Define bins across TC range
#     bins = np.linspace(df['TC'].min(), df['TC'].max(), n_bins + 1)
#     df = df.copy()
#     df['TC_bin'] = pd.cut(df['TC'], bins=bins, include_lowest=True)

#     for label_name in unique_labels:
#         print(f"\nLabel: {label_name}")
        
#         # Filter for the label
#         label_df = df[df['label'] == label_name]
        
#         # Group by binned TC and response, count occurrences
#         distribution = (
#             label_df.groupby(['TC_bin', 'response'])
#             .size()
#             .reset_index(name='count')
#         )
        
#         # Pivot for easier reading
#         distribution_pivot = (
#             distribution.pivot(index='TC_bin', columns='response', values='count')
#             .fillna(0)
#             .rename(columns={0: 'Response_0', 1: 'Response_1'})
#             .astype(int)
#         )
        
#         print(distribution_pivot)
# # Example usage:
# response_distribution_all_labels(cleaned_df)
# %%
