#%%
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scripts.functions  # your cleaning utilities

#%% Dynamic imports
this_file = Path(__file__).resolve()
utils_path = this_file.parent.parent / 'utils'
sys.path.append(str(utils_path))
import utils

#%%
def fitfunction(TC, a, b):
    eta = a + b * TC
    return 1 - np.exp(-np.exp(eta))

def fitfunctioninverse(p, a, b):
    eta = np.log(-np.log(1 - p))
    return (eta - a) / b

detection_prob = 0.75

#%% Dynamic paths for data loading
output_path = this_file.parent.parent / 'Output'
exp_path = output_path / 'Exp'
baseline_path = exp_path / 'Baseline'
main_path = output_path / 'analysis'
eyelink_path = output_path / 'Eyelink'

#%% Load data
baseline_df = utils.load_data(baseline_path)
main_df = utils.load_data(main_path)

#%% Extract participant IDs
ids = main_df['id'].unique()
print(f"Found {len(ids)} participant(s): {ids}")

#%% Separate dataframes per participant
participant_dfs = {
    pid: main_df[main_df['id'] == pid].copy() for pid in ids
}

#%% Clean and fit GLM per participant
fit_results = {}
for pid, df in participant_dfs.items():
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    cleaned_df['num_false_positives'] = num_false_positives
    participant_dfs[pid] = cleaned_df

    fit_result = smf.glm(
        formula='response ~ TC * FC * C(condition, Treatment(reference="target"))',
        data=cleaned_df,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        var_weights=cleaned_df['weight']
    ).fit()
    fit_results[pid] = fit_result

    print(f"\nParticipant {pid} model fit summary:")
    print(fit_result.summary())

#############################################################
#####



# #%% To check, 
# for pid, fit_result in fit_results.items():
#     df = participant_dfs[pid]
#     df = df[df['TC'] < 0.03]
#     print(f"\n--- Plotting GLM fits for participant {pid} ---")

#     # Get all unique conditions present for this participant
#     conditions = df['condition'].unique()

#     plt.figure(figsize=(7,5))
#     plt.title(f"Participant {pid} — GLM fit per condition")
#     plt.xlabel("Target Contrast (TC)")
#     plt.ylabel("Detection Probability")

#     # Generate prediction range for TC
#     tc_range = np.linspace(df['TC'].min(), df['TC'].max(), 200)

#     for cond in conditions:
#         # Create a DataFrame for prediction
#         pred_df = pd.DataFrame({
#             "TC": tc_range,
#             "FC": [df['FC'].median()] * len(tc_range),  # or set FC=0 if you want fixed flanker contrast
#             "condition": [cond] * len(tc_range)
#         })

#         # Predict using GLM
#         try:
#             pred = fit_result.get_prediction(pred_df).summary_frame()
#         except Exception as e:
#             print(f" Skipping {cond}: {e}")
#             continue

#         # Plot predicted line and confidence interval
#         plt.plot(tc_range, pred['mean'], label=f"{cond}")
#         plt.fill_between(tc_range, pred['mean_ci_lower'], pred['mean_ci_upper'], alpha=0.2)

#         # Overlay actual data points for this condition
#         subset = df[df['condition'] == cond]
#         plt.scatter(subset['TC'], subset['response'], alpha=0.3, s=20)

#         plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.show()

# %%
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

TC_UPPER = 0.05  # <-- adjust range as needed
fit_results = {}

for pid, df in participant_dfs.items():
    print(f"\n=== Participant {pid} ===")

    # --- Clean participant data ---
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    cleaned_df['num_false_positives'] = num_false_positives
    participant_dfs[pid] = cleaned_df

    # --- Restrict to TC range of interest ---
    df_fit = cleaned_df[cleaned_df['TC'] < TC_UPPER].copy()
    if df_fit.empty:
        print(f"⚠️ No data below TC={TC_UPPER} for participant {pid}. Skipping.")
        continue

    print(f"Participant {pid}: fitting on {len(df_fit)} raw trials (TC < {TC_UPPER})")
    print(df_fit.groupby('condition')['TC'].count())

    # --- Fit GLM directly on all raw trials ---
    fit_result = smf.glm(
        formula='response ~ TC * FC * C(condition, Treatment(reference="target"))',
        data=df_fit,
        family=sm.families.Binomial(link=sm.families.links.CLogLog())
    ).fit()
    
    fit_results[pid] = fit_result
    print(fit_result.summary())

    # --- Prediction grid ---
    tc_range = np.linspace(df_fit['TC'].min(), df_fit['TC'].max(), 200)
    conditions = df_fit['condition'].unique()

    # =========================================================
    # (1) Combined plot — all conditions together
    # =========================================================
    plt.figure(figsize=(7,5))
    plt.title(f"Participant {pid} — GLM fits (TC < {TC_UPPER})")
    plt.xlabel("Target Contrast (TC)")
    plt.ylabel("Detection Probability")

    for cond in conditions:
        fc_val = df_fit.loc[df_fit['condition'] == cond, 'FC'].median()
        pred_df = pd.DataFrame({
            'TC': tc_range,
            'FC': [fc_val] * len(tc_range),
            'condition': [cond] * len(tc_range)
        })
        pred = fit_result.get_prediction(pred_df).summary_frame()

        plt.plot(tc_range, pred['mean'], label=f"{cond}")
        plt.fill_between(tc_range, pred['mean_ci_lower'], pred['mean_ci_upper'], alpha=0.2)

        subset = df_fit[df_fit['condition'] == cond]
        plt.scatter(subset['TC'], subset['response'], alpha=0.5, s=20, edgecolor='k')

    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # (2) Separate plots per condition
    # =========================================================
    n_cond = len(conditions)
    ncols = 2
    nrows = int(np.ceil(n_cond / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, cond in enumerate(conditions):
        ax = axes[i]
        fc_val = df_fit.loc[df_fit['condition'] == cond, 'FC'].median()
        pred_df = pd.DataFrame({
            'TC': tc_range,
            'FC': [fc_val] * len(tc_range),
            'condition': [cond] * len(tc_range)
        })
        pred = fit_result.get_prediction(pred_df).summary_frame()

        subset = df_fit[df_fit['condition'] == cond]
        ax.plot(tc_range, pred['mean'], color='C0', lw=2)
        ax.fill_between(tc_range, pred['mean_ci_lower'], pred['mean_ci_upper'], alpha=0.2, color='C0')
        ax.scatter(subset['TC'], subset['response'], color='k', alpha=0.7, s=20)

        ax.set_title(f"{cond}")
        ax.set_xlabel("Target Contrast (TC)")
        ax.set_ylabel("Detection Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Participant {pid} — Separate fits per condition (TC < {TC_UPPER})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    
    
# %% each condition itself
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

TC_UPPER = 0.1  # adjust as needed
fit_results = {}

for pid, df in participant_dfs.items():
    print(f"\n=== Participant {pid} ===")

    # --- Clean participant data ---
    cleaned_df, num_false_positives = scripts.functions.clean_df(df)
    cleaned_df['num_false_positives'] = num_false_positives
    participant_dfs[pid] = cleaned_df

    # --- Restrict to TC range of interest ---
    df_fit = cleaned_df[cleaned_df['TC'] < TC_UPPER].copy()
    if df_fit.empty:
        print(f"⚠️ No data below TC={TC_UPPER} for participant {pid}. Skipping.")
        continue

    print(f"Participant {pid}: fitting on {len(df_fit)} raw trials (TC < {TC_UPPER})")
    print(df_fit.groupby('label')['TC'].count())

    # --- Fit GLM directly on all raw trials, using label as factor ---
    fit_result = smf.glm(
        formula='response ~ TC * FC * C(label, Treatment(reference="target"))',
        data=df_fit,
        family=sm.families.Binomial(link=sm.families.links.CLogLog())
    ).fit()
    
    fit_results[pid] = fit_result
    print(fit_result.summary())

    # --- Prediction grid ---
    tc_range = np.linspace(df_fit['TC'].min(), df_fit['TC'].max(), 200)
    labels = df_fit['label'].unique()

    # =========================================================
    # (1) Combined plot — all labels together
    # =========================================================
    plt.figure(figsize=(7,5))
    plt.title(f"Participant {pid} — GLM fits per label (TC < {TC_UPPER})")
    plt.xlabel("Target Contrast (TC)")
    plt.ylabel("Detection Probability")

    for lab in labels:
        fc_val = df_fit.loc[df_fit['label'] == lab, 'FC'].median()
        pred_df = pd.DataFrame({
            'TC': tc_range,
            'FC': [fc_val] * len(tc_range),
            'label': [lab] * len(tc_range)
        })
        pred = fit_result.get_prediction(pred_df).summary_frame()

        plt.plot(tc_range, pred['mean'], label=f"{lab}")
        plt.fill_between(tc_range, pred['mean_ci_lower'], pred['mean_ci_upper'], alpha=0.2)

        subset = df_fit[df_fit['label'] == lab]
        plt.scatter(subset['TC'], subset['response'], alpha=0.5, s=20, edgecolor='k')

    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # (2) Separate plots per label
    # =========================================================
    n_label = len(labels)
    ncols = 2
    nrows = int(np.ceil(n_label / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lab in enumerate(labels):
        ax = axes[i]
        fc_val = df_fit.loc[df_fit['label'] == lab, 'FC'].median()
        pred_df = pd.DataFrame({
            'TC': tc_range,
            'FC': [fc_val] * len(tc_range),
            'label': [lab] * len(tc_range)
        })
        pred = fit_result.get_prediction(pred_df).summary_frame()

        subset = df_fit[df_fit['label'] == lab]
        ax.plot(tc_range, pred['mean'], color='C0', lw=2)
        ax.fill_between(tc_range, pred['mean_ci_lower'], pred['mean_ci_upper'], alpha=0.2, color='C0')
        ax.scatter(subset['TC'], subset['response'], color='k', alpha=0.7, s=20)

        ax.set_title(f"{lab}")
        ax.set_xlabel("Target Contrast (TC)")
        ax.set_ylabel("Detection Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Participant {pid} — Separate fits per label (TC < {TC_UPPER})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# %%
