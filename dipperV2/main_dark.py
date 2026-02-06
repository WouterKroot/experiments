#%%
import os
import sys
from pathlib import Path
import yaml
from utils import utils
from psychopy import core, visual, data, event, monitors, logging
import src.eyelink as eyelink
import src.eyelink_dummy as eyelink_dummy
from src.stimulus import Stimulus
from src.window import Window
from src.experiment import Experiment
import numpy as np
import random

# Set up settings for experiment:
is_test = False 

if is_test:
    tracker = False 
    run_baseline = True
    tutorial_done = False   
    sub_id = str(0)
else:
    tracker = True 
    run_baseline = True
    tutorial_done = True
    sub_id = str(utils.SubNumber("subNum.txt"))
    
#%% 

#%% Load configuration
default_config_dir = "./config"
default_config_filename = "expConfig.yaml"
default_config_path = os.path.join(default_config_dir, default_config_filename)

if not os.path.exists(default_config_path):
    raise FileNotFoundError(f"Configuration file not found: {default_config_path}")
print(f"Using config file: {default_config_path}")

# Load config
expConfig = utils.load_config(default_config_path)
print(f"Using config: {default_config_path}")

# get output path
base_dir = expConfig["paths"]["base_output_dir"]
exp_dir = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"])

if is_test == True:
    baseline_path = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"],expConfig["paths"]["test_output_dir"], expConfig["paths"]["baseline_name"],f"{sub_id}_baseline")
    main_path = os.path.join(base_dir, expConfig["paths"]["test_output_dir"], expConfig["paths"]["main_name"], f"{sub_id}_main")
    fullscr = False
    nTrials_base = expConfig["exp_blocks"]["baseline"]["test_trials"]
    nTrials_main = expConfig["exp_blocks"]["main"]["test_trials"]
    
else:
    baseline_path = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"], expConfig["paths"]["baseline_name"],f"{sub_id}_baseline")
    main_path = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"], expConfig["paths"]["main_name"], f"{sub_id}_main")
    fullscr = True
    nTrials_base = expConfig["exp_blocks"]["baseline"]["n_trials"]
    nTrials_main = expConfig["exp_blocks"]["main"]["n_trials"]    

nBlocks_base = expConfig["exp_blocks"]["baseline"]["n_blocks"]
nullOdds = expConfig["fixed_params"]["nullOdds"]
stepsizes = expConfig["fixed_params"]["step_sizes"]
    
nBlocks_main = expConfig["exp_blocks"]["main"]["n_blocks"]
min_val = expConfig["fixed_params"]["min_val"]
max_val = expConfig["fixed_params"]["max_val"]
start_val = expConfig["fixed_params"]["start_val"]
reversals = expConfig["fixed_params"]["reversals"]

background_colour = (min_val, min_val, min_val)

# Eyetracking
if tracker == True:
    eye_tracker = eyelink.EyeTracker(id=sub_id, doTracking=True, exp_dir=exp_dir)
    eye_tracker.startTracker()
else:
    eye_tracker = eyelink_dummy.DummyEyeTracker(id=sub_id, doTracking=False, exp_dir=None)
    
window = visual.Window(fullscr= fullscr,
                       monitor="Flanders", 
                       units="pix",
                       colorSpace='rgb',
                       color = background_colour,
                       bpc=(10,10,10),
                       depthBits=10
                       )

myWin = Window(window, expConfig)
myWin.stimuli = utils.load_stimuli(myWin)

#%%
baseline_thresholds = None
if run_baseline:
    baselineTargetCondition = [
        {
            'label': 'baseline_target',
            'stim_key': 'target',
            'startVal': start_val,
            'minVal': min_val,
            'maxVal': max_val,
            'stepSizes': stepsizes,
            'stepType': 'lin',
            'nReversals': reversals,
            'nUp': 1,
            'nDown': 1
        }
    ]

    redo = True 
    while redo:
        baseline = Experiment(
            myWin, sub_id,
            nTrials_base, nBlocks_base,
            eye_tracker,
            expConfig,
            baseline_path,
            nullOdds,
            baselineTargetCondition,
            min_val,
            baseline_thresholds=None
        )

        file_T = baseline.openDataFile()

        if not tutorial_done:
            baseline.run_tutorial()
            tutorial_done = True

        myWin.intro_baseline()
        baseline.run_baseline()
        baseline_thresholds = baseline.getThresholdFromBase(file_T)

        T_50 = baseline_thresholds[0.50]
        T_70 = baseline_thresholds[0.70]
        T_99 = baseline_thresholds[0.99]

        print(f"[BASELINE] Target threshold = {T_50:.4f}")
        redo = baseline.reDoBase(T_50)
        if redo:
            myWin.countdown()
            
    #if run_baseline == False:
#    T_70 = -0.9784
#    else:
#        pass

#     baselineFlankerCondition = [
#         {
#             'label': f'baseline_triple_flanker',
#             'stim_key': 'triple_flanker',
#             'startVal': -0.4,
#             'maxVal': 1.0,
#             'minVal': -1.0,
#             'stepSizes': stepsizes,
#             'stepType': 'lin',
#             'nReversals': 20,
#             'nUp': 1,
#             'nDown': 1,
#             'FC': -0.985          
#         }
#     ]

#     redo_F = True
#     while redo_F:
#         baseline_F = Experiment(
#             myWin, sub_id,
#             nTrials_base, nBlocks_base,
#             eye_tracker,
#             expConfig,
#             baseline_path,
#             nullOdds,
#             baselineFlankerCondition
#         )

#         file_F = baseline_F.openDataFile()

#         myWin.intro_baseline()
#         baseline_F.run_baseline()

#         thresholds_F = baseline_F.getThresholdFromBase(file_F)
#         F_50 = thresholds_F[0.50]
#         F_70 = thresholds_F[0.70]
#         F_99 = thresholds_F[0.99]

#         redo_F = baseline_F.reDoBase(F_50)

#         if redo_F:
#             myWin.countdown()

#     print(f"[BASELINE] Triple flanker threshold = {F_50:.4f}")

#     fc_levels = [
#         ("0", F_50), # TC at 10% percent detection of straight condition
#         ("1", F_70),
#         ("2", F_99),
#         ("3", F_99 / 2),
#         ("4", 0.0),
#         ("5", 1.0),
#     ]
# else:
#     fc_levels = [
#     ("0", -0.99), # TC at 10% percent detection of straight condition
#     ("1", -0.98),
#     ("2", -0.97),
#     ("3", -0.5),
#     ("4", 0.0),
#     ("5", 1.0),
# ]

#fc_levels = np.clip(fc_levels, -1.0, 1.0)
# print(f"[MAIN] Flanker contrast levels: {fc_levels}")

if baseline_thresholds is None:
    baseline_thresholds = {0.5: -0.75, 0.7: -0.70, 0.99: -0.65}
    T_50 = baseline_thresholds[0.50]
    T_70 = baseline_thresholds[0.70]
    T_99 = baseline_thresholds[0.99]
    print(f"No baseline thresholds found, using default T_50: {T_50})")
    #raise ValueError("No baseline thresholds found.")


experimentConditions = []
stim_keys = list(myWin.stimuli.keys())

for stim_key in stim_keys:
    if stim_key == "target":
        # continue
        condition = {
        "label": f"{stim_key}",
        "stim_key": stim_key,
        "startVal": start_val,
        "maxVal": expConfig['fixed_params']["max_val"],
        "minVal": expConfig['fixed_params']["min_val"],
        "stepSizes": expConfig['fixed_params']["step_sizes"],
        "stepType": expConfig['fixed_params']["step_type"],
        "nReversals": expConfig['fixed_params']["reversals"],
        "nUp": expConfig['fixed_params']["n_up"],
        "nDown": expConfig['fixed_params']["n_down"],
        "FC": min_val,
        }
        experimentConditions.append(condition)
            
    else:
        for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
            label = cond['label']
            factor = cond['FC_factor']
            
            if factor > 10:
                fc_value = 1.0
                print(f"Factor > 10 so baseline: {T_50}, fc_value: {fc_value}")
            else:
                fc_value = np.clip(min_val + (abs(T_50 - min_val) * factor), min_val, max_val)
                
            print(f"{stim_key}, {label}, {fc_value}")
            
            condition = {
                "label": f"{stim_key}_{label}",
                "stim_key": stim_key,
                "startVal": round(start_val + random.uniform(-0.3, 0.3), 4),
                "maxVal": expConfig['fixed_params']["max_val"],
                "minVal": expConfig['fixed_params']["min_val"],
                "stepSizes": expConfig['fixed_params']["step_sizes"],
                "stepType": expConfig['fixed_params']["step_type"],
                "nReversals": expConfig['fixed_params']["reversals"],
                "nUp": expConfig['fixed_params']["n_up"],
                "nDown": expConfig['fixed_params']["n_down"],
                "FC": fc_value,
            }

            experimentConditions.append(condition)

print(f"Len: {len(experimentConditions)} , Experiment conditions: {experimentConditions}")

if __name__ == "__main__":
    main = Experiment(
        myWin, sub_id,
        nTrials_main, nBlocks_main,
        eye_tracker,
        expConfig,
        main_path,
        nullOdds,
        experimentConditions,
        min_val,
        baseline_thresholds
    )

    main_output = main.openDataFile()
    myWin.intro_experiment()
    main.run_main(main_output)

# %%
