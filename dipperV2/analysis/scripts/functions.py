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

def clean_df(df):
    """
    Cleans the dataframe by removing no-response trials and false positives.
    """
    df = df[df["RT"] <= 90]  # remove no-response trials
    
    false_positives = df['label'].str.endswith('_null') & (df['response'] == 1)
    num_false_positives = false_positives.sum()
    
    null_trials = df['label'].str.endswith('_null')
    cleaned_df = df[~null_trials].copy()
    
    cleaned_df['condition'] = cleaned_df['label'].str.rsplit('_', n=1).str[0]
    cleaned_df['weight'] = cleaned_df.groupby(['TC', 'FC', 'condition'])['response'].transform('count')
    
    return cleaned_df, num_false_positives