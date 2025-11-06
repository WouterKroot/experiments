import numpy as np

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

# def clean_df(df):
#     """
#     Cleans the dataframe by removing no-response trials and false positives.
#     """
#     df = df[df["RT"] <= 90]  # remove no-response trials
    
#     false_positives = df['label'].str.endswith('_null') & (df['response'] == 1)
#     num_false_positives = false_positives.sum()
    
#     null_trials = df['label'].str.endswith('_null')
#     cleaned_df = df[~null_trials].copy()
    
#     cleaned_df['condition'] = cleaned_df['label'].str.rsplit('_', n=1).str[0]
#     cleaned_df['weight'] = cleaned_df.groupby(['TC', 'FC', 'condition'])['response'].transform('count')
#     cleaned_df['flanker_multiplier'] = 0  # default for 'target'
#     mask = cleaned_df['label'] != 'target'
#     cleaned_df.loc[mask, 'flanker_multiplier'] = cleaned_df['label'].str.rsplit('_', n=1).str[1]

#     return cleaned_df, num_false_positives

def clean_df(df):
    """
    Cleans the dataframe by removing no-response trials and false positives.
    """
    # Remove no-response trials
    df = df[df["RT"] <= 90]
    
    # Identify false positives (response=1 on _null trials)
    false_positives = df['label'].str.endswith('_null') & (df['response'] == 1)
    num_false_positives = false_positives.sum()
    
    # Remove _null trials entirely
    null_trials = df['label'].str.endswith('_null')
    cleaned_df = df[~null_trials].copy()
   
    cleaned_df = cleaned_df[cleaned_df['TC'] <= 0.6] 
    # Extract condition (everything before last "_")
    cleaned_df['condition'] = cleaned_df['label'].str.rsplit('_', n=1).str[0]
    
    # Compute trial weight
    cleaned_df['weight'] = cleaned_df.groupby(['TC', 'FC', 'condition'])['response'].transform('count')
    
    # Default flanker multiplier = 0 for 'target'
    cleaned_df['flanker_multiplier'] = 0  
    
    # For non-targets, extract numeric suffix and convert to int
    mask = cleaned_df['label'] != 'target'
    cleaned_df.loc[mask, 'flanker_multiplier'] = (
        cleaned_df.loc[mask, 'label'].str.rsplit('_', n=1).str[1].astype(int)
    )
    #cleaned_df['flanker_multiplier'] = cleaned_df['flanker_multiplier']/100.0  # convert to proportion (not necessary cause it's already wrt baseline)
    return cleaned_df, num_false_positives