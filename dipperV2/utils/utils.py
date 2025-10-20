import json
import yaml
import os
import math
import pandas as pd
from psychopy import visual

def SubNumber(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        content = int(file.read().strip())

    content_int = int(content)
    new_content = (content_int + 1)
    
    with open(filename, 'w') as file:
        file.write(str(new_content))
    return new_content

def load_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

def create_line(win, pos=(0, 0), angle=90, length=40):
    end1 = (
        pos[0] - math.cos(math.radians(angle)) * length / 2,
        pos[1] - math.sin(math.radians(angle)) * length / 2
    )
    end2 = (
        pos[0] + math.cos(math.radians(angle)) * length / 2,
        pos[1] + math.sin(math.radians(angle)) * length / 2
    )
    return visual.Line(win, start=end1, end=end2, lineColor='black', lineWidth=3.5)

def load_stimuli(myWin):
    win = myWin.win
    stim_dict = myWin.expConfig['stimuli']

    processed_stimuli = {}
    for stim_name, stim_list in stim_dict.items():
        drawables = []
        for entry in stim_list:
            if entry['object'] == 'line':
                line_obj = create_line(
                    win=win,
                    pos=tuple(entry.get('pos', (0, 0))),
                    angle=entry.get('angle', 90),
                    length=entry.get('length', 100),
                )
                entry['line_obj'] = line_obj
                drawables.append(line_obj)
        processed_stimuli[stim_name] = {
            'components': stim_list,
            'draw_lines': drawables
        }
    return processed_stimuli
        
def load_data(root_path):
    dfs = []
    
    # loop over folders inside root_path
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)  # full path
        
        # only process directories
        if not os.path.isdir(folder_path):
            continue
        
        # loop over files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(csv_path)
                print(f"Loaded: {csv_path}")
                dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        print("No CSV files found.")
    
    