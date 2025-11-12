#%%
import os
import sys
from pathlib import Path
import yaml
import math
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils import utils
from analysis.scripts import functions
#from dipperV2.utils import utils
from psychopy import core, visual, data, event, monitors, logging
import src.eyelink as eyelink
from src.stimulus import Stimulus
from src.window import Window
from src.experiment import Experiment

#%%
#baseline_df = utils.load_data(baseline_path)
main_path = '/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/Output/Test/Main/990_main/990_main.csv'
main_df = pd.read_csv(main_path)
main_df, _ = functions.clean_df(main_df)
#%% 
ids = main_df['id'].unique()
print(f"Found {len(ids)} participant(s): {ids}")

labels = main_df['label'].unique()
print(f"Found {len(labels)}") # conditions: {labels}")

#%% seperate dataframes per participant
participant_dfs = {participant_id: main_df[main_df['id'] == participant_id] for participant_id in ids}

def analyze_staircases(df):
    """
    For each label:
        - Sort by trial
        - Detect reversals
        - Compute step sizes between trials
        - Print summary diagnostics
    """

    results = {}

    for label in df['label'].unique():
        print("\n" + "="*60)
        print(f"Staircase analysis for label: {label}")
        print("="*60)

        # Filter & sort
        d = df[df['label'] == label].sort_values("trial").reset_index(drop=True)

        intensities = d['TC'].values
        responses = d['response'].values

        # -------------------------------------------
        # Detect reversals
        # -------------------------------------------
        # direction: +1 if intensity increased, -1 if decreased, 0 if no change
        delta = np.diff(intensities)
        direction = np.sign(delta)

        reversals = []
        for i in range(1, len(direction)):
            if direction[i] != 0 and direction[i] != direction[i-1]:
                # reversal occurs at trial i+1 (because diff shifts by 1)
                reversals.append(i+1)

        # -------------------------------------------
        # Step size per trial
        # -------------------------------------------
        step_sizes = np.abs(delta)

        # -------------------------------------------
        # Print diagnostic info
        # -------------------------------------------
        print(f"Total trials: {len(d)}")
        print(f"Intensities: {intensities}")
        print(f"Responses:   {responses}")
        print(f"Step sizes:  {step_sizes}")
        print(f"Reversals at trial indices: {reversals}")

        # Summary
        print(f"Number of reversals: {len(reversals)}")
        if len(step_sizes) > 0:
            print(f"Mean step size: {np.mean(step_sizes):.4f}")
            print(f"Min/Max step: {np.min(step_sizes):.4f} / {np.max(step_sizes):.4f}")

        # Save in result structure
        results[label] = {
            "df": d,
            "intensities": intensities,
            "responses": responses,
            "step_sizes": step_sizes,
            "reversals": reversals
        }

    return results



results = analyze_staircases(main_df)
print(results)

#%%
import pandas as pd
import pylab
file_path = '/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/Output/Test/Baseline/990_baseline/990_baseline.csv'
main_df = pd.read_csv(file_path)
df, _ = functions.clean_df(main_df)
df = df[df['TC'] < 0.08]
allIntensities = df['TC']
allResponses = df['response']
#%%

combinedInten, combinedResp, combinedN = \
             data.functionFromStaircase(allIntensities, allResponses, 'unique')
#fit curve - in this case using a Weibull function
fit = data.FitWeibull(combinedInten, combinedResp, guess=[0.02, 0.1])
smoothInt = pylab.arange(min(combinedInten), max(combinedInten), 0.001)
smoothResp = fit.eval(smoothInt)
thresh = fit.inverse(0.8)
print(thresh)

#plot curve
pylab.subplot(122)
pylab.plot(smoothInt, smoothResp, '-')
pylab.plot([thresh, thresh],[0,0.8],'--'); pylab.plot([0, thresh],\
[0.8,0.8],'--')
pylab.title('threshold = %0.3f' %(thresh))
#plot points
pylab.plot(combinedInten, combinedResp, 'o')
pylab.ylim([0,1.2])
pylab.show()

# default_config_path = project_root / "config" / "expConfig.yaml"

# if not os.path.exists(default_config_path):
#     raise FileNotFoundError(f"Configuration file not found: {default_config_path}")
# print(f"Using config file: {default_config_path}")

# # Load config
# expConfig = utils.load_config(default_config_path)
# print(f"Using config: {default_config_path}")



#%%
# def create_line(win, pos=(0, 0), angle=90, length=40):
#     end1 = (
#         pos[0] - math.cos(math.radians(angle)) * length / 2,
#         pos[1] - math.sin(math.radians(angle)) * length / 2
#     )
#     end2 = (
#         pos[0] + math.cos(math.radians(angle)) * length / 2,
#         pos[1] + math.sin(math.radians(angle)) * length / 2
#     )
#     return visual.Line(win, start=end1, end=end2, lineColor='black', lineWidth=3.5)

# def load_stimuli(expConfig):
#     stim_dict = expConfig['stimuli']
#     processed_stimuli = {}
    
#     for stim_name, stim_list in stim_dict.items():
#         drawables = []
#         for entry in stim_list:
#             if entry['object'] == 'line':
#                 line_obj = create_line(
#                     win=None,
#                     pos=tuple(entry.get('pos', (0, 0))),
#                     angle=entry.get('angle', 90),
#                     length=entry.get('length', 100),
#                 )
#                 entry['line_obj'] = line_obj
#                 drawables.append(line_obj)
#         processed_stimuli[stim_name] = {
#             'components': stim_list,
#             'draw_lines': drawables
#         }
#     return processed_stimuli
    
# nTrials = expConfig["exp_blocks"]["baseline"]["test_trials"]
# nBlocks = expConfig["exp_blocks"]["baseline"]["n_blocks"]
# nullOdds = expConfig["fixed_params"]["nullOdds"]

# baselineCondition = [
#     {'label': 'target',
#      'startVal': 0.03,      # a bit above threshold to allow both up/down movement
#      'maxVal': 0.04,         # upper bound for the staircase
#      'minVal': 0.0015,      # lower bound
#      'stepSizes': [5.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1],  # big → small log steps
#      'stepType': 'log',
#      'nReversals': 20,      # enough for reliable slope + threshold
#      'nUp': 1,
#      'nDown': 1}            # targets ~50%
# ]
# base

# for stim_key in stim_keys:
#     if stim_key == "target":
#         # target-only condition (no flanker)
#         condition = {
#             "label": f"{stim_key}",
#             "stim_key": stim_key,
#             "startVal": baseline_threshold,
#             "maxVal": expConfig['fixed_params']["max_val"],
#             "minVal": expConfig['fixed_params']["min_val"],
#             "stepSizes": expConfig['fixed_params']["step_size"],
#             "stepType": expConfig['fixed_params']["step_type"],
#             "nReversals": expConfig['fixed_params']["reversals"],
#             "nUp": expConfig['fixed_params']["n_up"],
#             "nDown": expConfig['fixed_params']["n_down"],
#             "FC": 0
#         }
#         experimentConditions.append(condition) 

#     else:
#         # add all flanker conditions for other stim_key
#         for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
#             label = cond['label']
#             factor = cond['FC_factor']
            
#             if factor > 10:
#                 fc_value = 0.8
#                 print(f"baseline: {baseline_threshold}, fc_value: {fc_value}")
#             else:
#                 fc_value = baseline_threshold*factor
                
            
#             print(f"{stim_key}, {label}", {fc_value})
#             condition = {
#                 "label": f"{stim_key}_{label}",
#                 "stim_key": stim_key,
#                 "startVal": baseline_threshold,
#                 "maxVal": expConfig['fixed_params']["max_val"],
#                 "minVal": expConfig['fixed_params']["min_val"],
#                 "stepSizes": expConfig['fixed_params']["step_size"],
#                 "stepType": expConfig['fixed_params']["step_type"],
#                 "nReversals": expConfig['fixed_params']["reversals"],
#                 "nUp": expConfig['fixed_params']["n_up"],
#                 "nDown": expConfig['fixed_params']["n_down"],
#                 "FC": fc_value,
#             }

#             experimentConditions.append(condition)
            
# print(f"Len: {len(experimentConditions)} , Experiment conditions: {experimentConditions}")