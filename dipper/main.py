# Based on yaml configurations create conditions and stimulus objects per condition, load all stimuli from yaml and associate condition with name, draw stimuli with visual.Line inside a window object, and run the experiment with baseline contrast in the main.py file.
#%%
import os
import sys
from pathlib import Path
import yaml
from utils import utils
import src.eyelink as eyelink
from psychopy import core, visual, data, event, monitors, logging
import src.eyelink as eyelink
from src.stimulus import Stimulus
from src.window import Window
from src.experiment import Experiment

is_test = False
tracker = True

#%% Test or experiment
# while test not in ["test", "experiment"]:
#     try:
#         test = input("Run as 'test' or 'experiment'? ").strip().lower()
        
#         if test not in ["test", "experiment"]:
#             print("Invalid input. Please type 'test' or 'experiment'.")
#     except (EOFError, KeyboardInterrupt):
#         print("User cancelled. Exiting.")
#         sys.exit()

# is_test = 1 if test == "test" else 0

#%% 
sub_id = str(utils.SubNumber("subNum.txt"))
# Eyetracking
if tracker == True:
    eye_tracker = eyelink.EyeTracker(id=sub_id, doTracking=True)
    eye_tracker.startTracker()
else:
    from src.eyelink_dummy import DummyEyeTracker
    eye_tracker = DummyEyeTracker()

#%% Load configuration
default_config_dir = "./config"
default_config_filename = "expConfig.yaml"
default_config_path = os.path.join(default_config_dir, default_config_filename)

# Get config file from CLI argument (ignoring flags)
# user_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
# if user_args:
#     user_config = user_args[0]
    
#     if os.path.isabs(user_config):
#         config_path = user_config
#     elif os.path.exists(user_config):
#         config_path = user_config  # relative path provided and valid
#     else:
#         config_path = os.path.join(default_config_dir, user_config)
# else:
#     config_path = default_config_path
# # Expand and absolutize
# config_path = os.path.abspath(os.path.expanduser(config_path))

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

window = visual.Window(fullscr=True,
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
nullOdds = 0.3

baselineCondition = [
    {'label':'target','startVal':0.1,'maxVal':0.1,'minVal':0.0,
        'stepSizes':0.1,'stepType':'log','nReversals':1,
        'nUp':1,'nDown':1}]
        
redo = True
while redo:
    baseline = Experiment(myWin, sub_id, nTrials, nBlocks, eye_tracker, expConfig, baseline_path, nullOdds, baselineCondition)
    file_path = baseline.openDataFile()
    baseline.run_baseline()  # This appends to the same file
    baseline_threshold = baseline.getThresholdFromBase(file_path)
    redo = baseline.reDoBase(baseline_threshold)
    if redo:
        myWin.countdown()
        
#%% Load in main setting and run
if is_test == 1:
    nTrials = expConfig["exp_blocks"]["main"]["test_trials"]
else: 
    nTrials = expConfig["exp_blocks"]["main"]["n_trials"]

nBlocks = expConfig["exp_blocks"]["main"]["n_blocks"]

if baseline_threshold < 0 or baseline_threshold > 0.1:
        baseline_threshold = 0.02 #0.01
        
# get conditions:
experimentConditions = []
stim_keys = list(myWin.stimuli.keys())

for stim_key in stim_keys:
    for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
        label = cond['label']
        factor = cond['FC_factor']

        condition = {
            "label": f"{stim_key}_{label}",          
            "stim_key": stim_key,                      
            "startVal": baseline_threshold * 2,
            "maxVal": expConfig['fixed_params']["max_val"],
            "minVal": expConfig['fixed_params']["min_val"],
            "stepSizes": expConfig['fixed_params']["step_size"],
            "stepType": expConfig['fixed_params']["step_type"],
            "nReversals": expConfig['fixed_params']["reversals"],
            "nUp":  expConfig['fixed_params']["n_up"],
            "nDown": expConfig['fixed_params']["n_down"],
            "FC": baseline_threshold * factor
        }

        experimentConditions.append(condition)

# Run the main experiment
main = Experiment(myWin, sub_id, nTrials, nBlocks, eye_tracker, expConfig, main_path, nullOdds, experimentConditions)
main_output = main.openDataFile()
main.run_main(main_output)
main.end()
