import json
import yaml
import os
import math
import numpy as np
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
    return visual.Line(win, start=end1, end=end2, lineColor='white', lineWidth=3.5)

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
    
def contrast_steps_log(baseline, max_val=1.0, n_steps=7):
    """
    Generate contrast steps starting from baseline.
    - The first few steps are fixed ratios: 1, 1.5, 3
    - Remaining steps are log-spaced up to max_val (1.0)
    """
    first_ratios = np.array([1, 1.5, 3])
    first_steps = baseline * first_ratios
    first_steps = first_steps[first_steps <= max_val]

    if first_steps[-1] < max_val:
        num_remaining = n_steps - len(first_steps)
        if num_remaining > 0:
            remaining_steps = np.logspace(
                np.log10(first_steps[-1]),
                np.log10(max_val),
                num=num_remaining + 1,
            )[1:]  # skip first point
            contrasts = np.concatenate([first_steps, remaining_steps])
        else:
            contrasts = first_steps
    else:
        contrasts = first_steps

    return contrasts 

def fitfunction(TC, alpha, beta):
    """Weibull psychometric function.
    """
    return 1 - np.exp(-(TC / alpha) ** beta)

def fitfunctioninverse(p, alpha, beta):
    """Inverse Weibull function.
    """
    # avoid domain errors for p=1
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return alpha * (-np.log(1 - p)) ** (1 / beta)

def extract_condition(label):
    # Remove _null if present
    label = label.replace('_null', '')
    # Remove numeric/flanker suffix if it exists
    parts = label.split('_')
    if parts[-1].isdigit():
        parts = parts[:-1]
    return '_'.join(parts)
    
def clean_df(df):
    """
    Removes '_null' trials and computes false positive stats per label.

    Returns
    -------
    cleaned_df : DataFrame
        Data without '_null' trials.
    false_positive_stats : dict
        {base_label: {'false_positives': int, 'total_null_trials': int, 'false_positive_rate': float}}
    """
    df = df.copy()
    df['condition'] = df['label'].apply(extract_condition)
    

    # Boolean mask for null trials
    is_null_trial = df['label'].str.endswith('_null')
    # Extract only null trials
    null_trials = df[is_null_trial].copy()
    #null_trials['condition'] = null_trials['label'].apply(extract_condition)

    # Compute false positive counts per condition type
    false_positives = {}
    
    for condition, subset in null_trials.groupby('condition'):
        n_total = len(subset)
        n_fp = (subset['response'] == 1).sum()
        fp_rate = n_fp / n_total if n_total > 0 else 0.0

        false_positives[condition] = {
            'false_positives': int(n_fp),
            'total_null_trials': int(n_total),
            'false_positive_rate': fp_rate
        }

    # Keep only non-null trials
    cleaned_df = df[~is_null_trial].copy()

    # # Optional: add a column for condition grouping
    # cleaned_df['condition'] = cleaned_df['label'].str.rsplit('_', n=1).str[0]
    
    # Default flanker multiplier = 0 for 'target'
    cleaned_df['flanker_multiplier'] = 0  
    # For non-targets, extract numeric suffix and convert to int
    mask = cleaned_df['label'] != 'target'
    cleaned_df.loc[mask, 'flanker_multiplier'] = (
        cleaned_df.loc[mask, 'label'].str.rsplit('_', n=1).str[1].astype(int)
    )
    return cleaned_df, false_positives

def response_distribution(df, false_positive_dict, max_val=1.0, n_bins=30):
    """
    Returns the distribution of response=1 vs response=0 per binned intensity (TC)
    for every unique label, including adjusted proportion based on false positives.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'label', 'TC', 'response', and 'condition'.
    false_positive_dict : dict
        Dictionary with keys = condition, containing 'false_positive_rate'.
    max_val : float
        Maximum TC value to include in bins.
    n_bins : int
        Number of bins to divide the TC range into.

    Returns
    -------
    all_distributions : dict
        Dictionary of label_name -> DataFrame with columns:
        ['TC_bin', 'Response_0', 'Response_1', 'Proportion_yes', 'Adjusted_yes']
    combined_df : pandas.DataFrame
        All label distributions concatenated with 'label' column.
    """

    all_distributions = {}

    # Filter TC range
    df = df[(df['TC'] >= -1) & (df['TC'] <= max_val)].copy()

    # Define bins
    bins = np.linspace(df['TC'].min(), df['TC'].max(), n_bins + 1)
    df['TC_bin'] = pd.cut(df['TC'], bins=bins, include_lowest=True)

    combined_list = []

    for label_name in df['label'].unique():
        label_df = df[df['label'] == label_name]
        condition = label_df['condition'].iloc[0]  # use condition to get fp

        # Get false positive rate for this condition
        fp_rate = false_positive_dict.get(condition, {}).get('false_positive_rate', 0.0)

        data = []
        for b in label_df['TC_bin'].cat.categories:
            bin_df = label_df[label_df['TC_bin'] == b]
            n0 = (bin_df['response'] == 0).sum()
            n1 = (bin_df['response'] == 1).sum()
            total = n0 + n1
            proportion_yes = n1 / total if total > 0 else np.nan
            adjusted_yes = (proportion_yes - fp_rate) / (1 - fp_rate) if total > 0 else np.nan
            adjusted_yes = np.clip(adjusted_yes, 0, 1)  # keep in [0,1]

            data.append({
                'TC_bin': b,
                'Response_0': n0,
                'Response_1': n1,
                'Proportion_yes': proportion_yes,
                'Adjusted_yes': adjusted_yes
            })

        counts = pd.DataFrame(data)
        all_distributions[label_name] = counts

        counts['label'] = label_name
        combined_list.append(counts)

        print(f"\nLabel: {label_name}")
        print(counts)

    combined_df = pd.concat(combined_list, ignore_index=True)
    return all_distributions, combined_df

