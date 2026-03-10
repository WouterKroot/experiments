#%%
import os
import sys
from pathlib import Path
import yaml
from utils import utils
from psychopy import core, visual, data, event, monitors, logging
import src.eyelink as eyelink
import src.eyelink_dummy as eyelink_dummy
#from src.stimulus import Stimulus
from src.window import Window
from src.experiment import Experiment
import numpy as np
import random

# Set up settings for experiment:
is_test = True
background_config = "black" # "white" or "black"

if is_test:
    tracker = False 
    run_baseline = False 
    tutorial_done = True   
    sub_id = "000"
else:
    tracker = True 
    run_baseline = True 
    tutorial_done = False
    sub_id = str(utils.SubNumber("subNum.txt"))

#%% Load configuration
default_config_dir = "./config"

if background_config == "white":
    default_config_filename = "expConfig_white.yaml"
elif background_config == "black":
    default_config_filename = "expConfig_black.yaml"
    
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

background_val = expConfig["fixed_params"]["background_val"]


if background_val >= 0:
    max_val = 1.0 # this also clips contrast?
    min_val = -background_val # Negative of background, passed in contrast of line (line color = - background)
else: # Background is negative, so min val is clipped to background, and max val is 1.0
    max_val = 1.0
    min_val = background_val # Background is negative pass into contrast of line (line color = - background)
    
print(f"Background val: {background_val}, max_val: {max_val}, min_val: {min_val}")

start_val = expConfig["fixed_params"]["start_val"]
reversals = expConfig["fixed_params"]["reversals"]

background_colour = (background_val, background_val, background_val)

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
print(f"Background colour: {window.color}")

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
            baseline_thresholds = None
        )

        file_T = baseline.openDataFile()

        if not tutorial_done:
            baseline.run_tutorial()
            tutorial_done = True

        myWin.intro_baseline()
        baseline.run_baseline()
        baseline_thresholds_norm, baseline_thresholds = baseline.getThresholdFromBase(file_T)

        T_50_norm = baseline_thresholds_norm
        T_50 = baseline_thresholds

        print(f"[BASELINE] Target threshold = {T_50:.8f}")
        redo = baseline.reDoBase(T_50)
        if redo:
            myWin.countdown()
            

if baseline_thresholds is None:
    if background_val >= 0:
        baseline_thresholds = {0.5: -0.89, 0.7: -0.885, 0.99: -0.8}
        baseline_thresholds_norm = {0.5: 0.01, 0.7: 0.02, 0.99: 0.03}
    else:
        baseline_thresholds = {0.5: -0.796, 0.7: -0.786, 0.99: -0.780}
        baseline_thresholds_norm = {0.5: 0.01, 0.7: 0.02, 0.99: 0.03}
    
    T_50 = baseline_thresholds[0.50]
    # T_70 = baseline_thresholds[0.70]
    # T_99 = baseline_thresholds[0.99]

    T_50_norm = baseline_thresholds_norm[0.50]
    # T_70_norm = baseline_thresholds_norm[0.70]
    # T_99_norm = baseline_thresholds_norm[0.99]

    print(f"No baseline thresholds found, using default T_50: {T_50})")
    #raise ValueError("No baseline thresholds found.")

experimentConditions = []
stim_keys = list(myWin.stimuli.keys())
bg = myWin.background_val

for stim_key in stim_keys:
    if stim_key == "target":
        # continue
        condition = {
        "label": f"{stim_key}",
        "stim_key": stim_key,
        "startVal": start_val, 
        "maxVal": max_val,
        "minVal": min_val,
        "stepSizes": stepsizes,
        "stepType": expConfig['fixed_params']["step_type"],
        "nReversals": expConfig['fixed_params']["reversals"],
        "nUp": expConfig['fixed_params']["n_up"],
        "nDown": expConfig['fixed_params']["n_down"],
        "FC": background_val,
        }
        experimentConditions.append(condition)
            
    else:
        for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
            label = cond['label']
            factor = cond['FC_factor']
            
            # if factor == 10:
            #     fc_value = 0.2
            # elif factor == 100:
            #     fc_value = 1.0 # the full contrast of contour colour, negative max of -1 1 depending on background
            #     print(f"Factor > 20 so baseline: {T_50}, fc_value: {fc_value}")
            # else:
            #     delta_50 = (background_val + T_50) #T_50 is in contrast so will be negative of intensity of background, so + for difference
                
            #     if background_val >= 0:
            #         fc_value = -(background_val - (factor * delta_50))
            #     else:
            #         fc_value = (background_val - (factor * delta_50)) 
            
            if factor > 100:
                fc_value = 1.0 # the intensity of colour, negative max of -1 1 depending on background
                print(f"Factor > 20 so baseline: {T_50}, fc_value: {fc_value}")
            else:
                if background_val >= 0:
                    delta_50 = background_val + T_50 #+ 0.01 # add a small offset to avoid issues with contrast of 0
                    fc_value = -(background_val - (factor * delta_50)) # do we need the last -? 
                else:
                    delta_50 = T_50 - background_val #T_50 is in contrast so will be negative of intensity of background, so + for difference
                    fc_value = background_val + (factor * delta_50)

                    
            print(f"{stim_key}, {label}, {fc_value}")
            
            condition = {
                "label": f"{stim_key}_{label}",
                "stim_key": stim_key,
                "startVal": round(start_val + random.uniform(-0.3, 0), 4), # start_val
                "maxVal": max_val,
                "minVal": min_val,
                "stepSizes": stepsizes,
                "stepType": expConfig['fixed_params']["step_type"],
                "nReversals": expConfig['fixed_params']["reversals"],
                "nUp": expConfig['fixed_params']["n_up"],
                "nDown": expConfig['fixed_params']["n_down"],
                "FC": fc_value, #Intensity or contrast?
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
        baseline_thresholds
    )

    main_output = main.openDataFile()
    myWin.intro_experiment()
    main.run_main(main_output)

# %%
