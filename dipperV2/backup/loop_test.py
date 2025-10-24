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

#%%
sub_id = 0

default_config_dir = "./config"
default_config_filename = "expConfig.yaml"
default_config_path = os.path.join(default_config_dir, default_config_filename)

expConfig = utils.load_config(default_config_path)
base_dir = expConfig["paths"]["base_output_dir"]

baseline_path = os.path.join(base_dir, expConfig["paths"]["test_output_dir"], expConfig["paths"]["baseline_name"],f"{sub_id}_baseline")
#main_path = os.path.join(base_dir, expConfig["paths"]["test_output_dir"], expConfig["paths"]["main_name"], f"{sub_id}_main")
main_path = '/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/'
    
window = visual.Window(fullscr= False,
                       monitor="Flanders", 
                       units="pix",
                       colorSpace='rgb',
                       color = [0,0,0],
                       bpc=(10,10,10),
                       depthBits=10
                       )

myWin = Window(window, expConfig)
myWin.stimuli = utils.load_stimuli(myWin) # A new drawable object is created and 

nTrials = expConfig["exp_blocks"]["main"]["test_trials"]
nBlocks = expConfig["exp_blocks"]["main"]["n_blocks"]
nullOdds = expConfig["fixed_params"]["nullOdds"]
baseline_threshold = 0.03

experimentConditions = []
stim_keys = list(myWin.stimuli.keys())

for stim_key in stim_keys:
    if stim_key == "target":
        # target-only condition (no flanker)
        condition = {
            "label": f"{stim_key}",
            "stim_key": stim_key,
            "startVal": baseline_threshold,
            "maxVal": expConfig['fixed_params']["max_val"],
            "minVal": expConfig['fixed_params']["min_val"],
            "stepSizes": expConfig['fixed_params']["step_size"],
            "stepType": expConfig['fixed_params']["step_type"],
            "nReversals": expConfig['fixed_params']["reversals"],
            "nUp": expConfig['fixed_params']["n_up"],
            "nDown": expConfig['fixed_params']["n_down"],
            "FC": 0
        }
        experimentConditions.append(condition) 

    else:
        # add all flanker conditions for other stim_key
        for cond in expConfig['exp_blocks']['main']['flanker_conditions']:
            label = cond['label']
            factor = cond['FC_factor']

            condition = {
                "label": f"{stim_key}_{label}",
                "stim_key": stim_key,
                "startVal": baseline_threshold,
                "maxVal": expConfig['fixed_params']["max_val"],
                "minVal": expConfig['fixed_params']["min_val"],
                "stepSizes": expConfig['fixed_params']["step_size"],
                "stepType": expConfig['fixed_params']["step_type"],
                "nReversals": expConfig['fixed_params']["reversals"],
                "nUp": expConfig['fixed_params']["n_up"],
                "nDown": expConfig['fixed_params']["n_down"],
                "FC": baseline_threshold * factor,
            }

            experimentConditions.append(condition)
            
print(f"Len: {len(experimentConditions)} , Experiment conditions: {experimentConditions}")
eye_tracker = eyelink.EyeTracker(id=sub_id, doTracking=False)
main = Experiment(myWin, sub_id, nTrials, nBlocks, eye_tracker, expConfig, main_path, nullOdds, experimentConditions)

main_output = main.openDataFile()
main.run_main(main_output)
# %%
