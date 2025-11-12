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

is_test = False 
tracker = True 

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
                       color = [0,0,0],
                       bpc=(10,10,10),
                       depthBits=10
                       )
#%%
myWin = Window(window, expConfig)
myWin.stimuli = utils.load_stimuli(myWin) # A new drawable object is created and 

#%% start with introduction
myWin.intro()

#%% Load in baseline settings from dict and run:
if is_test == True:
    nTrials = expConfig["exp_blocks"]["baseline"]["test_trials"]
else: 
    nTrials =  expConfig["exp_blocks"]["baseline"]["n_trials"]

nBlocks = expConfig["exp_blocks"]["baseline"]["n_blocks"]
nullOdds = expConfig["fixed_params"]["nullOdds"]

stepsizes = expConfig["fixed_params"]["step_sizes"]

baselineCondition = [
    {'label': 'target',
     'startVal': 0.2,      # a bit above threshold to allow both up/down movement
     'maxVal': 1.0,         # upper bound for the staircase
     'minVal': 0.001,      # lower bound
     'stepSizes': stepsizes, # big → small log steps but psychophy implementation is linear
     'stepType': 'lin',
     'nReversals': 20,      # enough for reliable slope + threshold
     'nUp': 1,
     'nDown': 1}            # targets ~50%
]

redo = True
while redo:
    baseline = Experiment(myWin, sub_id, nTrials, nBlocks, eye_tracker, expConfig, baseline_path, nullOdds, baselineCondition, baseline_threshold=None)
    file_path = baseline.openDataFile()
    baseline.run_baseline()  # This appends to the same file
    baseline_threshold = baseline.getThresholdFromBase(file_path)
    redo = baseline.reDoBase(baseline_threshold)
    if redo:
        myWin.countdown()
        
# baseline_threshold = 0.02
        
#%% Load in main setting and run
if is_test == 1:
    nTrials = expConfig["exp_blocks"]["main"]["test_trials"]
else: 
    nTrials = expConfig["exp_blocks"]["main"]["n_trials"]

nBlocks = expConfig["exp_blocks"]["main"]["n_blocks"]

if baseline_threshold < 0 or baseline_threshold > 0.04:
        baseline_threshold = 0.03 #0.01
        
# get conditions:
experimentConditions = []
stim_keys = list(myWin.stimuli.keys())

for stim_key in stim_keys:
    if stim_key == "target":
        # target-only condition (no flanker)
        # condition = {
        #     "label": f"{stim_key}",
        #     "stim_key": stim_key,
        #     "startVal": 0.1,
        #     "maxVal": expConfig['fixed_params']["max_val"],
        #     "minVal": expConfig['fixed_params']["min_val"],
        #     "stepSizes": expConfig['fixed_params']["step_size"],
        #     "stepType": expConfig['fixed_params']["step_type"],
        #     "nReversals": expConfig['fixed_params']["reversals"],
        #     "nUp": expConfig['fixed_params']["n_up"],
        #     "nDown": expConfig['fixed_params']["n_down"],
        #     "FC": 0
        # }
        # experimentConditions.append(condition) 
        continue

    else:
        # add all flanker conditions for other stim_key
        for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
            label = cond['label']
            factor = cond['FC_factor']
            
            if factor > 10:
                fc_value = 0.8
                print(f"Factor > 10 so baseline: {baseline_threshold}, fc_value: {fc_value}")
            else:
                fc_value = baseline_threshold*factor
                
            
            print(f"{stim_key}, {label}", {fc_value})
            
            condition = {
                "label": f"{stim_key}_{label}",
                "stim_key": stim_key,
                "startVal": 0.2,
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



#%%
# Run the main experiment
if __name__ == "__main__":
    main = Experiment(myWin, sub_id, nTrials, nBlocks, eye_tracker, expConfig, main_path, nullOdds, experimentConditions, baseline_threshold)
    main_output = main.openDataFile()
    main.run_tutorial()
    main.run_main(main_output)
