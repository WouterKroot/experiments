# ToDo:
# per id create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)
# show that facilitation and inhibition effects can be modelled as multiplicative gain modulation multiplied by input contrast
#%%
from psychopy import data
from pathlib import Path
import sys
import seaborn as sns
import scripts.functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from pathlib import Path
import os
from datetime import datetime

# dynamic import
this_file = Path(__file__).resolve()
utils_path = this_file.parent.parent / 'utils'  # go up 2 levels to dipperV2 then into utils
sys.path.append(str(utils_path))
import utils

test = False 
#%%
#Dynamic paths for data loading
#results_path = this_file.parent.parent.parent.parent / 'Data'
daystamp = datetime.now().strftime("%Y%m%d")
timestamp = datetime.now().strftime("%H%M%S")   
data_path = this_file.parent.parent / 'data' / 'dark_background_300'
results_path = this_file.parent / 'results' / 'dark_background_300' / daystamp / timestamp
os.makedirs(results_path, exist_ok=True)

if test == True:
    exp_path = data_path / 'Test'
else:
    exp_path = data_path
    
baseline_path = exp_path / 'Baseline'
main_path = exp_path / 'Main'
eyelink_path = data_path / 'Eyelink'

#%%
baseline_df = utils.load_data(baseline_path)
main_df = utils.load_data(main_path)
#%% 
ids = main_df['id'].unique()
#ids = [1126]
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
    
    # For 150 ms, bins of max val -0.8, and 10 bins works
    # for 300 ms, bins of max val -0.82 and 
    all_distributions, combined_df = utils.response_distribution(cleaned_df, false_positives, max_val=-0.87, n_bins=15) # Size of the smallest log stepsize, is 0.0025


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

    participant_results_path = os.path.join(results_path, str(participant_id))
    os.makedirs(participant_results_path, exist_ok=True)

    for label_name, df_label in response_summary.items():
        if label_name not in thresholds[participant_id]:
            thresholds[participant_id][label_name] = {}
        if df_label.empty:
            continue

        df_label = df_label.copy()
        # CHANGED THE LAMBDA
        df_label['TC_center'] = df_label['TC_bin'].apply(
             lambda x: x.mid if hasattr(x, 'mid') else np.nan)
        df_label = df_label.dropna(subset=['TC_center'])
        
        df_label['Total'] = df_label['Response_0'] + df_label['Response_1']

        glm_data = df_label[['TC_center', 'Adjusted_yes', 'Total']].copy()
        glm_data = glm_data.dropna()
        glm_data = glm_data[glm_data['Total'] > 0]
        
        if glm_data.empty:
            continue
        
        total_sum = glm_data['Total'].sum()
        glm_data['prop_weight'] = glm_data['Total'] / total_sum
        #glm_data.loc[glm_data['prop_weight'] > 0.02, 'prop_weight'] = 0
     
        glm_model = smf.glm(
            formula='Adjusted_yes ~ TC_center',
            data=glm_data,
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            freq_weights=glm_data['prop_weight']  # use prop_weight here
        ).fit() # or cov_type='HC3' # changed freq_weights to var_weights
         
        fit_results[participant_id][label_name] = glm_model
       
       #ToDO: differentiate between target and target baseline
        # if label_name == "target":
        #     use_thresh_vals = [0.5, 0.7]
        # else:
        use_thresh_vals = [0.5, 0.99, 0.7]
            
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
        save_path = os.path.join(participant_results_path, f"{safe_label}.png")
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

    #ToDO: get baseline_target and main_target thresholds
    # baseline target 0.5
    # baseline = thresholds[participant_id]['target'][0.5]
    target = thresholds[participant_id]['target'][0.7]
    conditions = cleaned_df['condition'].unique()
    
    # build mapping: flanker_condition → FC
    # fc_map = (
    #     cleaned_df
    #     .dropna(subset=['flanker_condition', 'FC'])
    #     .groupby('flanker_condition')['FC']
    #     .nunique()
    # )
    # #assert (fc_map == 1).all(), "Some flanker conditions map to multiple FC values!"
    
    # fc_map = (
    # cleaned_df
    # .dropna(subset=['flanker_condition', 'FC'])
    # .groupby('flanker_condition')['FC']
    # .first()
    # .to_dict()
    # )
    fc_raw = (
    cleaned_df
    .dropna(subset=['flanker_condition', 'FC'])
    .groupby('flanker_condition')['FC']
    .first()
    .to_dict()
    )

    # sort flanker conditions (keys)
    flanker_sorted = sorted(fc_raw.keys())

    # sort FC values (lowest → highest)
    fc_sorted = sorted(fc_raw.values())

    # safety check
    assert len(flanker_sorted) == len(fc_sorted), "Mismatch between flanker conditions and FC values"

    # rebuild mapping: smallest flanker → lowest FC
    fc_map = dict(zip(flanker_sorted, fc_sorted))

    flanker_conditions = sorted(fc_map.keys())
    #flanker_conditions = [125, 150, 300, 900, 1000]
    
    for cond in conditions:
        for mult in flanker_conditions:
            #Todo: check how to change the mult to be a string
            mult_str = str(int(mult))
            key = f"{cond}_{mult_str}"

            if key not in thresholds[participant_id]:
                continue
            if 0.7 not in thresholds[participant_id][key]:
                continue

            if mult not in fc_map:
                continue  # safety

            fc_x = fc_map[mult]
            t07 = thresholds[participant_id][key][0.7]

            plot_data.append({
                'participant': participant_id,
                'condition': cond,
                'flanker': mult,
                'FC': fc_x,
                'threshold07': t07,
                'target07': target
            })
    #ToDo: Find the correct FC values per condition
    # flanker_conditions = np.sort(cleaned_df['flanker_condition'].dropna().unique())
    # FC = cleaned_df['FC'].dropna().unique()
    # FC = np.sort(FC)

    # for cond in conditions:
    #     for mult in flanker_conditions:
    #         print(mult)
    #         key = f"{cond}_{mult}"
    #         if key in thresholds[participant_id] and 0.7 in thresholds[participant_id][key]:
    #             t07 = thresholds[participant_id][key][0.7]
    #             if mult > 4:
    #                 fc_x = 1.0
    #             else: #ToDo: FC should not be calculated, but taken from data
    #                 fc_x = baseline * (mult/100)
                
    #             plot_data.append({
    #                 'participant': participant_id,
    #                 'condition': cond,
    #                 'flanker': mult,
    #                 'FC': fc_x,
    #                 'threshold07': t07,
    #                 'target07': target
    #             })
    plot_df = pd.DataFrame(plot_data)
    agg_plot_df = pd.concat([agg_plot_df, plot_df], ignore_index=True)
    
    plot_conditions = plot_df['condition'].unique()
    plt.figure(figsize=(8,6))

    for cond in plot_conditions:
        sub = plot_df[plot_df['condition'] == cond]
        plt.plot(sub['FC'], sub['threshold07'], marker='o', label=cond)

    plt.axhline(y=target, color='k', linestyle='--', label='Target (0.7)')
    plt.xlabel("FC")
    plt.xlim(-1.0, 1)
    plt.ylabel("Adjusted Threshold (0.7)")
    plt.title(f"Participant: {participant_id} Thresholds by Condition")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(participant_results_path, f"participant_{participant_id}_thresholds_by_condition.png")
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
#plt.xlim(-0.9, -0.7)
plt.title("Mean Thresholds Across Participants by Condition")
plt.legend()
plt.grid(True)
#plt.savefig(os.path.join(results_path, "mean_thresholds_by_condition.png"), dpi=300)
plt.show()
##%
#%% LLM version for relative contrast over threshold:
allowed_flankers = df_mean['flanker'].unique()[2:]  # Exclude first two flankers

df_mean = (
    agg_plot_df[agg_plot_df['flanker'].isin(allowed_flankers)]
    .groupby(['condition', 'flanker'])
    .agg(
        mean_threshold=('threshold07', 'mean'),
        std_threshold=('threshold07', 'std'),
        n=('threshold07', 'count'),
        mean_FC=('FC', 'mean'),
        mean_target=('target07', 'mean')
    )
    .reset_index()
)

# SEM
df_mean['sem'] = df_mean['std_threshold'] / np.sqrt(df_mean['n'])

# Background contrast
background = -0.9

# Scale by background → target contrast
df_mean['delta_threshold_bg_scaled'] = (
    (df_mean['mean_threshold'] - df_mean['mean_target']) /
    (df_mean['mean_target'] - background)
) * 100

df_mean['sem_bg_scaled'] = (
    df_mean['sem'] /
    (df_mean['mean_target'] - background)
) * 100

# Normalize FC for x-axis (unchanged)
maximum = 1.0
df_mean['mean_FC_pct'] = (
    (df_mean['mean_FC'] - background) / (maximum - background)
) * 100

# Plot
plt.figure(figsize=(8, 6))

for cond in df_mean['condition'].unique():
    sub = df_mean[df_mean['condition'] == cond]
    plt.errorbar(
        sub['mean_FC_pct'],
        sub['delta_threshold_bg_scaled'],
        yerr=sub['sem_bg_scaled'],
        marker='o',
        capsize=3,
        label=cond
    )

plt.axhline(0, color='k', linestyle='--', label='Target (Δ = 0)')
plt.xlabel("FC (% normalized from −0.9)")
plt.ylabel("%ΔDetection with respect to background-target contrast)")
plt.xscale('log')
plt.xlim(0.1, 150)
plt.title("Contour Detection is Modulated by Surround Contrast")
plt.grid(False)
# plt.legend()

plt.savefig(
    os.path.join(results_path, "delta_thresholds_bg_scaled.png"),
    dpi=300
)
plt.show()

# allowed_flankers = df_mean['flanker'].unique()[2:]  # Exclude first two flankers

# df_mean = (
#     agg_plot_df[agg_plot_df['flanker'].isin(allowed_flankers)]
#     .groupby(['condition', 'flanker'])
#     .agg(
#         mean_threshold=('threshold07', 'mean'),
#         std_threshold=('threshold07', 'std'),
#         n=('threshold07', 'count'),
#         mean_FC=('FC', 'mean'),
#         mean_target=('target07', 'mean')
#     )
#     .reset_index()
# )

# # SEM
# df_mean['sem'] = df_mean['std_threshold'] / np.sqrt(df_mean['n'])

# # Normalization parameters
# baseline = -0.9
# maximum = 1.0

# # Normalize FC
# df_mean['mean_FC_pct'] = (
#     (df_mean['mean_FC'] - baseline) / (maximum - baseline)
# ) * 100

# # Normalize thresholds and target
# df_mean['mean_threshold_pct'] = (
#     (df_mean['mean_threshold'] - baseline) / (maximum - baseline)
# ) * 100

# df_mean['mean_target_pct'] = (
#     (df_mean['mean_target'] - baseline) / (maximum - baseline)
# ) * 100

# # Difference relative to target
# df_mean['delta_threshold_pct'] = (
#     df_mean['mean_threshold_pct'] - df_mean['mean_target_pct']
# )

# # SEM in normalized space
# df_mean['sem_pct'] = df_mean['sem'] / (maximum - baseline) * 100

# # Plot
# plt.figure(figsize=(8, 6))

# for cond in df_mean['condition'].unique():
#     sub = df_mean[df_mean['condition'] == cond]
#     plt.errorbar(
#         sub['mean_FC_pct'],
#         sub['delta_threshold_pct'],
#         yerr=sub['sem_pct'],
#         marker='o',
#         capsize=3,
#         label=cond
#     )

# plt.axhline(0, color='k', linestyle='--', label='Target (Δ = 0)')
# plt.xlabel("FC (% normalized from −0.9)")
# plt.ylabel("Δ Threshold (% contrast relative to target)")
# plt.xscale('log')
# plt.xlim(0.1, 150)
# plt.title("Normalized Threshold Difference Relative to Target")
# plt.grid(False)
# # plt.legend()

# plt.savefig(os.path.join(results_path, "delta_thresholds_by_condition.png"), dpi=300)
# plt.show()
# %%
# Normalize FC to 0-100% based on min and max values
# df_mean = (
#     agg_plot_df.groupby(['condition', 'flanker'])
#           .agg(mean_threshold=('threshold07', 'mean'),
#                std_threshold=('threshold07', 'std'),
#                n=('threshold07', 'count'),
#                mean_FC=('FC', 'mean'),
#                mean_target=('target07', 'mean'))
#           .reset_index()
# )
allowed_flankers = df_mean['flanker'].unique()[2:]  # Exclude first two flankers
df_mean = (
    agg_plot_df[agg_plot_df['flanker'].isin(allowed_flankers)].groupby(['condition', 'flanker'])
          .agg(mean_threshold=('threshold07', 'mean'),
               std_threshold=('threshold07', 'std'),
               n=('threshold07', 'count'),
               mean_FC=('FC', 'mean'),
               mean_target=('target07', 'mean'))
          .reset_index()
)


df_mean['sem'] = df_mean['std_threshold'] / np.sqrt(df_mean['n'])

# Normalize with -0.9 as baseline (0%) and 1.0 as max (100%)
baseline = -0.9
maximum = 1.0
df_mean['mean_FC_pct'] = ((df_mean['mean_FC'] - baseline) / (maximum - baseline)) * 100

plt.figure(figsize=(8,6))
for cond in df_mean['condition'].unique():
    sub = df_mean[df_mean['condition'] == cond]
    plt.errorbar( 
        sub['mean_FC_pct'],
        sub['mean_threshold'],
        yerr=sub['sem'],
        marker='o',
        capsize=3,
        label=cond
    )

plt.axhline(y=df_mean['mean_target'].mean(), color='k', linestyle='--', label='Mean Target (0.7)')
plt.xlabel("FC (%, normalized from -0.9)")
plt.ylabel("Mean Adjusted Threshold (0.7)")
plt.xscale('log')
plt.xlim(0.1, 150)  # Start slightly above 0 for log scale
plt.title("Normalized Mean Thresholds Across Participants by Condition")
#plt.legend()
plt.grid(False)
plt.savefig(os.path.join(results_path, "mean_thresholds_by_condition.png"), dpi=300)
plt.show()


# %% Bar plot (needs simplification)
# self bar plot, express everything as percentage change from target threshold at 0.7
def change_function(target, value):
    return ((value - target) / abs(target)) * 100

agg_plot_df['norm_TC'] = ((agg_plot_df['threshold07'] - agg_plot_df['target07']) / agg_plot_df['target07']) * 100

conditions = agg_plot_df['condition'].unique()
flankers = agg_plot_df['flanker'].unique()
allowed_flankers = flankers[2:]
participants = agg_plot_df['participant'].unique()

for part in participants:
    for cond in conditions:
        # Filter data for this participant and condition
        subset = agg_plot_df[
            (agg_plot_df['participant'] == part) &
            (agg_plot_df['condition'] == cond) &
            (agg_plot_df['flanker'].isin(allowed_flankers))]
        
        if subset.empty:
            continue
            
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.barplot(data=subset,
                    x='flanker',
                    y='norm_TC',
                    ax=ax,
                    palette='Set2')
        
        ax.set_xlabel('Flanker Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized TC (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Participant {part} | {cond}', fontsize=14)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        plt.close()


#%%
# grand_avg_df = agg_plot_df.groupby(['condition', 'flanker'])['norm_TC'].mean().reset_index()
# grand_avg_df = grand_avg_df[(grand_avg_df['flanker'].isin(allowed_flankers))]
grand_avg_df = (
    agg_plot_df[agg_plot_df['flanker'].isin(allowed_flankers)]
    .groupby(['condition', 'flanker'])
    .agg(
        mean_norm_TC=('norm_TC', 'mean'),
        sem_norm_TC=('norm_TC', lambda x: x.std(ddof=1) / np.sqrt(x.count()))
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=grand_avg_df,
            x='condition',
            y='mean_norm_TC',
            hue='flanker',
            ax=ax,
         palette='Set2')


ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized TC (%) - Avg over Participants', fontsize=12, fontweight='bold')
ax.set_title('Average across All Participants', fontsize=14)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(title='Flanker Condition')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()



# Per FC, calculate the percentage difference from target threshold [0.7], per condition. This will give us FC_25 = -0.89.. 
# Then we might see a shift in dipper function
# all_data = []

# for participant_id, dfs in participant_dfs.items():
#     participant_thresholds = thresholds[participant_id]
#     cleaned_df = dfs['cleaned_df']
    
#     # Get the target threshold at 0.7
#     if 'target' not in participant_thresholds or 0.7 not in participant_thresholds['target']:
#         print(f"Skipping participant {participant_id}: No target threshold at 0.7")
#         continue
    
#     target_threshold_07 = participant_thresholds['target'][0.7]
    
#     # Build FC mapping from data
#     fc_raw = (
#         cleaned_df
#         .dropna(subset=['flanker_condition', 'FC'])
#         .groupby('flanker_condition')['FC']
#         .first()
#         .to_dict()
#     )
    
#     if not fc_raw:
#         print(f"Skipping participant {participant_id}: No FC mapping found")
#         continue
    
#     flanker_sorted = sorted(fc_raw.keys())
#     fc_sorted = sorted(fc_raw.values())
#     fc_map = dict(zip(flanker_sorted, fc_sorted))
    
#     # Get unique conditions from the cleaned_df (excluding 'target')
#     bar_conditions = [cond for cond in cleaned_df['condition'].unique() if cond != 'target']
    
#     if not conditions:
#         print(f"Skipping participant {participant_id}: No conditions found")
#         continue
    
#     # Process each condition separately
#     for cond in bar_conditions:
#         condition_data = []
        
#         for flanker in flanker_sorted:
#             flanker_str = str(int(flanker))
#             fc_value = fc_map[flanker]
            
#             condition_key = f"{cond}_{flanker_str}"
            
#             if condition_key not in participant_thresholds:
#                 continue
#             if 0.7 not in participant_thresholds[condition_key]:
#                 continue
            
#             condition_threshold_07 = participant_thresholds[condition_key][0.7]
            
#             # Calculate percentage change from target
#             pct_change = ((condition_threshold_07 - target_threshold_07) / target_threshold_07) * 100
            
#             condition_data.append({
#                 'flanker': flanker,
#                 'FC': fc_value,
#                 'condition_threshold': condition_threshold_07,
#                 'pct_change': pct_change
#             })
            
#             all_data.append({
#                 'participant': participant_id,
#                 'condition': cond,
#                 'flanker': flanker,
#                 'FC': fc_value,
#                 'target_threshold': target_threshold_07,
#                 'condition_threshold': condition_threshold_07,
#                 'pct_change': pct_change
#             })
        
#         if not condition_data:
#             continue
        
#         # Create DataFrame for this condition
#         df_cond = pd.DataFrame(condition_data)
#         df_cond = df_cond.sort_values('FC')
        
#         # Create bar plot for this participant and condition
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         x_positions = range(len(df_cond))
#         colors = ['red' if x < 0 else 'green' for x in df_cond['pct_change']]
        
#         bars = ax.bar(x_positions, df_cond['pct_change'], 
#                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        
#         # Add horizontal line at 0
#         ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        
#         # Customize plot
#         ax.set_xlabel('FC (Flanker Contrast)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('% Change from Target Threshold (0.7)', fontsize=12, fontweight='bold')
#         ax.set_title(f'Participant {participant_id} - Condition: {cond}\n(Target = {target_threshold_07:.4f})', 
#                      fontsize=14, fontweight='bold')
#         ax.set_xticks(x_positions)
#         ax.set_xticklabels([f'{fc:.3f}' for fc in df_cond['FC']], 
#                           rotation=45, ha='right', fontsize=9)
#         ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
#         # Add value labels on bars
#         for i, (idx, row) in enumerate(df_cond.iterrows()):
#             height = row['pct_change']
#             ax.text(i, height, f"{height:.1f}%", 
#                     ha='center', va='bottom' if height > 0 else 'top',
#                     fontsize=9, fontweight='bold')
        
#         # Add text box with statistics
#         stats_text = f"Mean: {df_cond['pct_change'].mean():.1f}%\n"
#         stats_text += f"Range: [{df_cond['pct_change'].min():.1f}%, {df_cond['pct_change'].max():.1f}%]"
#         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#                 fontsize=10, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         plt.tight_layout()
        
#         # Save figure
#         save_path = os.path.join(participant_results_path, 
#                                 f"participant_{participant_id}_condition_{cond}_pct_change.png")
#         #plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()

#        print(f"Saved plot for participant {participant_id}, condition {cond}")
# %%
# def change_function(target, value):
#     return ((value - target) / abs(target)) * 100

