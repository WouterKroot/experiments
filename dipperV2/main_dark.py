# BASELINE is updating however MAIN is not. So have to check main update rules. make flanker contrast minimum be to -1.


# Based on yaml configurations create conditions and stimulus objects per condition, load all stimuli from yaml and associate condition with name, draw stimuli with visual.Line inside a window object, and run the experiment with baseline contrast in the main.py file.
#%%
import os
import sys
from pathlib import Path
import yaml
from utils import utils
from psychopy import core, visual, data, event, monitors, logging
import src.eyelink as eyelink
from src.stimulus import Stimulus
from src.window import Window
from src.experiment import Experiment
import numpy as np

is_test = False 
tracker = False 

#%% 
sub_id = str(utils.SubNumber("subNum.txt"))
# Eyetracking
if tracker == True:
    eye_tracker = eyelink.EyeTracker(id=sub_id, doTracking=True)
    eye_tracker.startTracker()
else:
    eye_tracker = eyelink.EyeTracker(id=sub_id, doTracking=False)

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

if is_test == True:
    baseline_path = os.path.join(base_dir, expConfig["paths"]["test_output_dir"], expConfig["paths"]["baseline_name"],f"{sub_id}_baseline")
    main_path = os.path.join(base_dir, expConfig["paths"]["test_output_dir"], expConfig["paths"]["main_name"], f"{sub_id}_main")
    
else:
    baseline_path = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"], expConfig["paths"]["baseline_name"],f"{sub_id}_baseline")
    main_path = os.path.join(base_dir, expConfig["paths"]["exp_output_dir"], expConfig["paths"]["main_name"], f"{sub_id}_main")

if is_test == True:
    fullscr = False 
else:
    fullscr = True
    
window = visual.Window(fullscr= fullscr,
                       monitor="Flanders", 
                       units="pix",
                       colorSpace='rgb',
                       color = [-1,-1,-1],
                       bpc=(10,10,10),
                       depthBits=10
                       )
#%%
# theta_F = -0.89
# theta_T = -0.8

myWin = Window(window, expConfig)
myWin.stimuli = utils.load_stimuli(myWin)

if is_test:
    nTrials_base = expConfig["exp_blocks"]["baseline"]["test_trials"]
else:
    nTrials_base = expConfig["exp_blocks"]["baseline"]["n_trials"]

nBlocks_base = expConfig["exp_blocks"]["baseline"]["n_blocks"]
nullOdds = expConfig["fixed_params"]["nullOdds"]
stepsizes = expConfig["fixed_params"]["step_sizes"]


baselineTargetCondition = [
    {
        'label': 'target',
        'startVal': 0.0,
        'maxVal': 1.0,
        'minVal': -1.0,
        'stepSizes': stepsizes,
        'stepType': 'lin',
        'nReversals': 20,
        'nUp': 1,
        'nDown': 1
    }
]

redo = True
tutorial_done = False 

while redo:
    baseline_T = Experiment(
        myWin, sub_id,
        nTrials_base, nBlocks_base,
        eye_tracker,
        expConfig,
        baseline_path,
        nullOdds,
        baselineTargetCondition,
        baseline_threshold=None
    )

    file_T = baseline_T.openDataFile()

    if not tutorial_done:
        baseline_T.run_tutorial()
        tutorial_done = True

    myWin.intro_baseline()
    baseline_T.run_baseline()

    theta_T = baseline_T.getThresholdFromBase(file_T)
    redo = baseline_T.reDoBase(theta_T)

    if redo:
        myWin.countdown()

print(f"[BASELINE] Target threshold θ_T = {theta_T:.4f}")

baselineFlankerCondition = [
    {
        'label': 'single_flanker_top',
        'stim_key': 'flanker_top',
        'startVal': 0.0,
        'maxVal': 1.0,
        'minVal': -1.0,
        'stepSizes': stepsizes,
        'stepType': 'lin',
        'nReversals': 20,
        'nUp': 1,
        'nDown': 1,
        'FC': theta_T          
    }
]

redo = True
while redo:
    baseline_F = Experiment(
        myWin, sub_id,
        nTrials_base, nBlocks_base,
        eye_tracker,
        expConfig,
        baseline_path,
        nullOdds,
        baselineFlankerCondition,
        baseline_threshold=None
    )

    file_F = baseline_F.openDataFile()

    myWin.intro_baseline()
    baseline_F.run_baseline()

    theta_F = baseline_F.getThresholdFromBase(file_F)
    redo = baseline_F.reDoBase(theta_F)

    if redo:
        myWin.countdown()

print(f"[BASELINE] Single flanker threshold θ_F = {theta_F:.4f}")

if is_test:
    nTrials_main = expConfig["exp_blocks"]["main"]["test_trials"]
else:
    nTrials_main = expConfig["exp_blocks"]["main"]["n_trials"]

nBlocks_main = expConfig["exp_blocks"]["main"]["n_blocks"]

delta = theta_T - theta_F

fc_levels = [
    ("F_50", theta_F),
    ("T_50", theta_T),
    ("T_half",      theta_T / 2),
    ("F_zero",        0.0),
    ("F_high",        0.8),
]

#fc_levels = np.clip(fc_levels, -1.0, 1.0)
print(f"[MAIN] Flanker contrast levels: {fc_levels}")

experimentConditions = []
stim_keys = list(myWin.stimuli.keys())

for stim_key in stim_keys:
    if stim_key == "target":
        # continue
                condition = {
            "label": f"{stim_key}",
            "stim_key": stim_key,
            "startVal": 0.0,
            "maxVal": expConfig['fixed_params']["max_val"],
            "minVal": expConfig['fixed_params']["min_val"],
            "stepSizes": expConfig['fixed_params']["step_sizes"],
            "stepType": expConfig['fixed_params']["step_type"],
            "nReversals": expConfig['fixed_params']["reversals"],
            "nUp": expConfig['fixed_params']["n_up"],
            "nDown": expConfig['fixed_params']["n_down"],
            "FC": -1.0,
            "FC_label": None
        }

    for fc_name, fc_value in fc_levels:
        condition = {
            "label": f"{stim_key}_{fc_name}",   # semantic, shared across subjects
            "stim_key": stim_key,
            "startVal": 0.0,
            "maxVal": expConfig['fixed_params']["max_val"],
            "minVal": expConfig['fixed_params']["min_val"],
            "stepSizes": expConfig['fixed_params']["step_sizes"],
            "stepType": expConfig['fixed_params']["step_type"],
            "nReversals": expConfig['fixed_params']["reversals"],
            "nUp": expConfig['fixed_params']["n_up"],
            "nDown": expConfig['fixed_params']["n_down"],
            "FC": float(fc_value),              
            "FC_label": fc_name                 
        }

        experimentConditions.append(condition)

print(f"[MAIN] Total conditions: {len(experimentConditions)}")

if __name__ == "__main__":
    main = Experiment(
        myWin, sub_id,
        nTrials_main, nBlocks_main,
        eye_tracker,
        expConfig,
        main_path,
        nullOdds,
        experimentConditions,
        baseline_threshold=theta_T
    )

    main_output = main.openDataFile()
    myWin.intro_experiment()
    main.run_main(main_output)
