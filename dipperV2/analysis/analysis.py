# ToDo:
# per id create a summary plot fitting raw data using weighted Weibull function
# extract 0.75 threshold contrast from fit line per condition (dipper function)
# show that facilitation and inhibition effects can be modelled as multiplicative gain modulation
# take the mean of each condition across ids and plot group average with standard error shading
#%%
from pathlib import Path
import sys
import scripts.functions 

sys.path.append('/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/utils')
import utils

#%%
output_path = Path('/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/Output')

exp_path = output_path / 'Exp'
baseline_path = exp_path / 'Baseline'
main_path = exp_path / 'Main'
eyelink_path = output_path / 'Eyelink'

#%%
baseline_df = utils.load_data(baseline_path)
main_df = utils.load_data(main_path)

#%% Add per id a summary plot, dynamically generating for different conditions. Includes, weighted Weillbull fit of raw data, 0.75 threshold contrast from fit line per condition (dipper function).

# Goal is to show that the facilitation and inhibition effects can be modelled as a multiplicative gain modulation