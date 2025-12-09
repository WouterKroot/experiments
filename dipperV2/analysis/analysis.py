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

test = False
#%%
#Dynamic paths for data loading
#output_path = this_file.parent.parent.parent.parent / 'Data'
data_path = this_file.parent.parent / 'Output'
output_path = this_file.parent / 'Output' / 'dipperV2' / 'test'

if test == True:
    exp_path = data_path / 'Test2'
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
    all_distributions, combined_df = utils.response_distribution(cleaned_df, false_positives, max_val=0.1, n_bins=40) # Size of the smallest log stepsize, is 0.0025
   
    participant_dfs[participant_id]['cleaned_df'] = cleaned_df
    participant_dfs[participant_id]['false_positives'] = false_positives
    participant_dfs[participant_id]['response_summary'] = all_distributions
    
#%%
fit_results = {}
thresholds = {}  # store threshold values

for participant_id, dfs in participant_dfs.items():
    print(f"\n Participant {participant_id}")

    response_summary = dfs['response_summary']
    fit_results[participant_id] = {}
    thresholds[participant_id] = {}

    participant_output_path = os.path.join(output_path, str(participant_id))
    os.makedirs(participant_output_path, exist_ok=True)

    for label_name, df_label in response_summary.items():
        if label_name not in thresholds[participant_id]:
            thresholds[participant_id][label_name] = {}
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
        
        total_sum = glm_data['Total'].sum()
        glm_data['prop_weight'] = glm_data['Total'] / total_sum
     
        glm_model = smf.glm(
            formula='Adjusted_yes ~ TC_center',
            data=glm_data,
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            freq_weights=glm_data['prop_weight']  # use prop_weight here
        ).fit() # or cov_type='HC3' # changed freq_weights to var_weights
         
        fit_results[participant_id][label_name] = glm_model
        
        if label_name == "target":
            use_thresh_vals = [0.5, 0.7]
        else:
            use_thresh_vals = [0.7]
            
        for threshVal in use_thresh_vals:
            eta = glm_model.family.link(threshVal)
            thresh_glm = (eta - glm_model.params['Intercept']) / glm_model.params['TC_center']

            thresholds[participant_id][label_name][threshVal] = thresh_glm 

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
        plt.show()
        plt.close()

        print(f"Saved GLM plot for {label_name} → {save_path}")
        print(f"{label_name} GLM summary:")
        print(glm_model.summary())
        print(f"{label_name} GLM threshold (P={threshVal}) = {thresh_glm:.3f}")

#%% Self: make sure that this runs for multiple participants as well
# The binning needs to be done based on the individual participant data
agg_plot_df = pd.DataFrame()
for participant_id, dfs in participant_dfs.items():
    plot_data = []
    cleaned_df = dfs['cleaned_df']

    # baseline target 0.5
    baseline = thresholds[participant_id]['target'][0.5]
    target = thresholds[participant_id]['target'][0.7]

    conditions = cleaned_df['condition'].unique()
    
    multipliers = np.sort(cleaned_df['flanker_multiplier'].unique())
    FC = cleaned_df['FC'].unique()
    FC = np.sort(FC)

    for cond in conditions:
        for mult in multipliers:
            print(mult)
            key = f"{cond}_{mult}"
            if key in thresholds[participant_id] and 0.7 in thresholds[participant_id][key]:
                t07 = thresholds[participant_id][key][0.7]
                if mult > 900:
                    fc_x = 1.0
                else:
                    fc_x = baseline * (mult/100)
                
                plot_data.append({
                    'participant': participant_id,
                    'condition': cond,
                    'flanker': mult,
                    'FC': fc_x,
                    'threshold07': t07,
                    'target07': target
                })
    plot_df = pd.DataFrame(plot_data)
    agg_plot_df = pd.concat([agg_plot_df, plot_df], ignore_index=True)
    
    conditions = plot_df['condition'].unique()
    plt.figure(figsize=(8,6))

    for cond in conditions:
        sub = plot_df[plot_df['condition'] == cond]
        plt.plot(sub['FC'], sub['threshold07'], marker='o', label=cond)

    plt.axhline(y=target, color='k', linestyle='--', label='Target (0.7)')
    plt.xlabel("FC")
    #plt.xlim(0, 0.2)
    plt.ylabel("Adjusted Threshold (0.7)")
    plt.title(f"Participant: {participant_id} Thresholds by Condition")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(participant_output_path, f"participant_{participant_id}_thresholds_by_condition.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    
    plt.close()
# %%
df_mean = (
    agg_plot_df.groupby(['condition', 'flanker'])
          .agg(mean_threshold=('threshold07', 'mean'),
               std_threshold=('threshold07', 'std'),
               n=('threshold07', 'count'),
               mean_FC=('FC', 'mean'),
               mean_target=('target07', 'mean'))
          .reset_index()
)
df_mean['sem'] = df_mean['std_threshold'] / np.sqrt(df_mean['n'])

plt.figure(figsize=(8,6))

for cond in df_mean['condition'].unique():
    sub = df_mean[df_mean['condition'] == cond]

    plt.errorbar( 
        sub['mean_FC'],  # assuming baseline target 0.5
        sub['mean_threshold'],
        yerr=sub['sem'],      # optional: remove if no error bars
        marker='o',
        capsize=3,
        label=cond
    )
plt.axhline(y=df_mean['mean_target'].mean(), color='k', linestyle='--', label='Mean Target (0.7)')
plt.xlabel("FC")
plt.ylabel("Mean Adjusted Threshold (0.7)")
#plt.xlim(0, 0.2)
plt.title("Mean Thresholds Across Participants by Condition")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_path, "mean_thresholds_by_condition.png"), dpi=300)
plt.show()
# %%
