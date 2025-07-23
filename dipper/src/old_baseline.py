import os
from psychopy import core, visual, data, event
from psychopy.tools.filetools import fromFile
import pylab

def run_baseline(
    win,
    subject_id,
    output_dir,
    t_fixation=0.3,
    t_stim=0.2,
    t_response=1.3,
    n_trials=40,
    n_up=1,
    n_down=1,
    start_val=0.1,
    min_val=0.0,
    max_val=0.1,
    step_size=0.1,
    step_type='log',
    reversals=1,
    expected_min=0.0,
    threshold_criterion=0.5
):
    stim = win.stimulus
    # Setup
    baseline_path = os.path.join(output_dir, f"{subject_id}_baseline.csv")
    data_file = open(baseline_path, 'w')
    data_file.write("id,trial,TC,response,RT\n")

    # Staircase
    stairs = data.StairHandler(
        startVal=start_val,
        minVal=min_val,
        maxVal=max_val,
        stepSizes=step_size,
        stepType=step_type,
        nReversals=reversals,
        nUp=n_up,
        nDown=n_down,
        nTrials=n_trials,
        name='baseline_stair'
    )

    # Trial Loop
    trial_clock = core.Clock()
    for trial in stairs:
        intensity = stairs.intensity
        stim.line_target.contrast = intensity

        stim.drawOrder(stim.fixation)
        core.wait(t_fixation)

        stim.drawOrder(stim.line_target)
        core.wait(t_stim)

        trial_clock.reset()
        stim.drawOrder(stim.blank)

        keys = event.waitKeys(maxWait=t_response, keyList=['left','right','num_4','num_6','q','escape'])
        rt = trial_clock.getTime()
        if rt < t_response:
            core.wait(t_response - rt)

        if keys:
            for key in keys:
                if key in ['left', 'num_4']:
                    resp = 0
                elif key in ['right', 'num_6']:
                    resp = 1
                elif key in ['q', 'escape']:
                    core.quit()
                else:
                    raise ValueError(f"Unexpected key: {key}")
        else:
            resp = 0
            rt = 99

        data_file.write(f"{subject_id},{stairs.thisTrialN},{intensity},{resp},{rt}\n")
        stairs.addResponse(resp)

    # Save staircase
    stair_file = os.path.join(output_dir, f"{subject_id}.psydat")
    stairs.saveAsPickle(stair_file)

    # Fit threshold
    dat = fromFile(stair_file)
    intensities = [dat.intensities]
    responses = [dat.data]

    i, r, n = data.functionFromStaircase(intensities, responses, bins='unique')
    n = pylab.array(n)
    fit = data.FitLogistic(i, r, expectedMin=expected_min, sems=1.0 / n)
    thresh = fit.inverse(threshold_criterion)

    print(f"-----------Threshold is: {thresh:.4f} -----------")
    return thresh