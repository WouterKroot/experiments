import os
import sys
import threading
from psychopy import data, core, visual, event
import pandas as pd
import pylab
import numpy as np
from typing import Literal
from utils import utils

class Experiment:
    def __init__(self, win,
                 subject_id, nTrials, nBlocks, eyeTracker,
                 expConfig, path, nullOdds, myConds, baseline_thresholds=None):
        self.myWin = win
        self.myConds = myConds
        self.id = subject_id
        self.nTrials = nTrials
        self.nBlocks = nBlocks
        self.expConfig = expConfig
        self.nullOdds = nullOdds
        self.eyeTracker = eyeTracker
        self.path = path
        self.base_name = os.path.basename(path)
        self.baseline_threshold = baseline_thresholds
        self.stairs = data.MultiStairHandler(stairType='simple',
                                             method='random',
                                             nTrials=self.nTrials,
                                             conditions=self.myConds, originPath=self.path)

    def getBreaks(self):
        totalTrials = int(len(self.myConds) * self.nTrials * (1 + self.nullOdds))
        breakTrials = np.linspace(start=0,stop=totalTrials, num=self.nBlocks,
                                  endpoint=False,dtype=int)[1:]
        return breakTrials, totalTrials

    
    def openDataFile(self):
        os.makedirs(self.path, exist_ok=True)

        fileName = os.path.join(self.path, f"{self.base_name}.csv")
        count = 1

        # Keep incrementing if file exists
        while os.path.exists(fileName):
            fileName = os.path.join(self.path, f"{self.base_name}_{count}.csv")
            count += 1

        print(f"CSV file opened at: {fileName}")
        
        self.dataFile = open(fileName, 'w', buffering=1)  # line-buffered
        self.dataFile.write("id,trial,label,FC,TC,FN,TN,response,RT\n")

        
        return fileName
    
    def run_baseline(self):
        dataFile = self.dataFile
        stairs = self.stairs
        trialClock = core.Clock()
        thisTrial = 0
        bg = self.myWin.background_val

        for trial, condition in stairs:
            print(f"Condition: {condition}")
            
            targetIntensity = round(float(stairs.currentStaircase.intensity), 8)
            thisStimulus = condition['stim_key']
            thisLabel = condition['label'] 
            flankerIntensity = round(float(stairs.currentStaircase.condition.get('FC', 0.0)), 8)

            targetContrast = utils.abs_contrast_from_bg(targetIntensity, bg)
            flankerContrast = utils.abs_contrast_from_bg(flankerIntensity, bg)

            stimulus = self.myWin.stimuli[thisStimulus]
            for entry in stimulus['components']:
                if entry.get('type') == 'target':
                    entry['line_obj'].contrast = targetIntensity
                    #delete
                    print(targetIntensity)
                else:
                    entry['line_obj'].contrast = flankerIntensity

            # Draw fixation
            self.myWin.win.flip()  # Clear previous frame
            self.myWin.fixation.draw()
            self.myWin.win.flip()
            core.wait(self.myWin.t_fixation)

            # Draw stimulus
            for line in stimulus['draw_lines']:
                line.draw()
            self.myWin.win.flip()
            core.wait(self.myWin.t_stim)

            trialClock.reset()
            self.myWin.drawOrder(self.myWin.blank)
            self.myWin.win.flip()
            
            allKeys = event.waitKeys(maxWait=self.myWin.t_response,
                        keyList=['left','num_4',
                            'right','num_6',
                            'q','escape'])
            
            thisRT = trialClock.getTime()
            if thisRT < self.myWin.t_response:
                core.wait(self.myWin.t_response - thisRT)

            if allKeys:
                for key in allKeys:
                    if key in ['left','num_4']:
                        thisResp = 0
                    elif key in ['right','num_6']:
                        thisResp = 1
                    elif key in ['q','escape']:
                        self.eyeTracker.closeTracker()
                        core.quit()
                    else:
                        raise ValueError(f"Unexpected key: {key}")
            else:
                thisResp = 0
                thisRT = 99
                
            if thisRT == 99:
                fb_stim = self.myWin.feedback_nan      
            elif thisResp == 1:
                fb_stim = self.myWin.feedback_yes     
            else:
                fb_stim = self.myWin.feedback_no       

            self.myWin.drawOrder(fb_stim)
            self.myWin.win.flip()
            core.wait(2/60)
            
            self.dataFile.write(f"{self.id},{thisTrial},{thisLabel},{flankerIntensity},{targetIntensity},{flankerContrast},{targetContrast},{thisResp},{thisRT}\n")
            self.dataFile.flush()
            
            if not thisLabel.endswith("_null") and thisRT != 99: 
                stairs.addResponse(thisResp) # Don't adjust the staircase for a null trial
           
            thisTrial += 1
            
        
        psydat_path = os.path.join(self.path, f"{self.id}_baseline.psydat")
        stairs.saveAsPickle(psydat_path)
        self.dataFile.close()
        print("Baseline done, break!")
        
    def compute_break_stats(self):
        # df = pd.read_csv(
        #     self.dataFile,
        #     names=['id','trial','label','FV','intensity','response','RT']
        # )
        self.dataFile.flush()
        df = pd.read_csv(os.path.join(self.path, f"{self.id}_main.csv"))

        # ---- Average RT (exclude timeouts + null trials) ----
        rt_mask = (df['RT'] != 99) & (~df['label'].str.endswith('_null'))
        avg_rt = df.loc[rt_mask, 'RT'].mean()

        # ---- False positives (null trials only) ----
        null_trials = df[df['label'].str.endswith('_null')]
        if len(null_trials) > 0:
            fp_rate = (null_trials['response'] == 1).mean()
        else:
            fp_rate = np.nan

        return avg_rt, fp_rate

    def doBreak(self,b, middle=False):
        print(f'this is middle: {middle}')
        
        avg_rt, fp_rate = self.compute_break_stats()

        rt_text = (
            "Avg RT: --"
            if np.isnan(avg_rt)
            else f"Avg RT: {avg_rt*1000:.0f} ms"
        )

        fp_text = (
            "False positives: --"
            if np.isnan(fp_rate)
            else f"False positives: {fp_rate*100:.1f}%"
        )
        
        if middle:
            m_break = visual.TextStim(self.myWin.win, color=self.myWin.stimulus_colour, height = 32, wrapWidth=600,
                        text= (
                f"You have finished block {b+1}.\n\n"
                f"{rt_text}\n"
                f"{fp_text}\n\n"
                "Take a LARGE break (~10 minutes).\n"
                "Stretch your legs, get some water, or rest your eyes.\n"
                "Press [RIGHT] when ready to continue."
            ))

            self.myWin.drawOrder(m_break)
            event.waitKeys(keyList=['right', 'num_6'])
            self.myWin.countdown()
            
        else:
            m_break = visual.TextStim(self.myWin.win, color=self.myWin.stimulus_colour, height = 32, wrapWidth=600,
                                    text=(f"You have finished block {b+1}.\nTime for a break. \nYou can stretch your legs or get some water.\nWait a bit before continuing.\n"
                                    "Press [RIGHT] to continue.\n\n"
                                    f"{rt_text}\n"
                                    f"{fp_text}\n\n"))

            self.myWin.drawOrder(m_break)
            event.waitKeys(keyList=['right','num_6'])
            self.myWin.countdown()
        
    def run_tutorial(self):
        win = self.myWin.win  

        def show_text(msg, wait_keys=['right', 'num_6'], color=self.myWin.stimulus_colour):
            text = visual.TextStim(win, color=color, text=msg)
            text.draw(); win.flip()
            event.waitKeys(keyList=wait_keys)

        show_text("Welcome to the Tutorial.\n\nTo begin, press the right arrow.")

        show_text("First, you will see a blank screen with a fixation point.\n\nTo continue, press the right arrow.")
        
        self.myWin.drawOrder(self.myWin.fixation)
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])

        show_text(
            "In the experiment this fixation point appears briefly.\n"
            "Directly afterwards, the stimulus will appear.\n"
            "It is important to focus on the line in the middle (the target).\n"
            "A red circle will mark it here, but not in the real experiment.\n\n"
            "To continue, press the right arrow."
        )

        top = visual.line.Line(win=self.myWin.win, start=(0, 30), end=(0, 70), pos=(0, 60), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour)
        middle = visual.line.Line(win=self.myWin.win, start=(0, -20), end=(0, 20), pos=(0, 0), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour)
        bottom = visual.line.Line(win=self.myWin.win, start=(0, -70), end=(0, -30), pos=(0, -60), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour)
        red_circle = visual.Circle(win=self.myWin.win, fillColor=None, radius=35, lineColor='red', lineWidth=3)

        self.myWin.drawOrder(self.myWin.fixation)
        core.wait(self.myWin.t_fixation)
        
        self.myWin.drawOrder([bottom, middle, top, red_circle])
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])

        show_text(
            "Now it's your turn!\n"
            "Focus on the target.\n"
            "Press RIGHT if the target is visible.\n"
            "Press LEFT if it is not visible.\n"
            "You have 1.2 seconds to respond.\n\n"
            "To continue, press the right arrow."
        )
        
        def show_trial(stims, visible=True):
                """Helper: show fixation, then stimuli, then collect response."""
                # Fixation
                self.myWin.drawOrder(self.myWin.fixation)
                core.wait(0.3)

                # Stimuli
                self.myWin.drawOrder(stims)
                core.wait(0.2)
                self.myWin.win.flip()  # blank screen

                # Response
                keys = event.waitKeys(
                    maxWait=1.2,
                    keyList=['left', 'num_4', 'right', 'num_6']
                )

                if not keys:
                    show_text("You need to press LEFT or RIGHT.\n\nPress RIGHT to continue.",
                            ['right', 'num_6'], 'red')
                    return 0

                key = keys[0]
                if (visible and key in ['right', 'num_6']) or (not visible and key in ['left', 'num_4']):
                    show_text("Correct!\n\nPress RIGHT to continue.", ['right', 'num_6'], 'green')
                    return 1
                else:
                    msg = ("Incorrect.\n" +
                        ("Target visible → press RIGHT." if visible else "Target invisible → press LEFT.") +
                        "\n\nPress RIGHT to continue.")
                    show_text(msg, ['right', 'num_6'], 'red')
                    return 0
        
        correct_streak = 0
        trial_num = 0

        while correct_streak < 3 and trial_num < 10:
            trial_num += 1
            correct_streak = 0  # restart streak each round

            # Trial 1: visible (single line)
            middle = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 0), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour, lineWidth=3
            )
            correct_streak += show_trial([middle], visible=True)

            # Trial 2: visible (3 lines)
            top = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 60), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour, lineWidth=3
            )
            middle = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 0), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour, lineWidth=3
            )
            bottom = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, -60), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour, lineWidth=3
            )
            correct_streak += show_trial([bottom, middle, top], visible=True)

            # Trial 3: invisible (9 lines grid)
            all_stims = []
            for y in [60, 0, -60]:
                for x in [-100, 0, 100]:
                    # Skip the center (target) completely
                    if x == 0 and y == 0:
                        continue
                    all_stims.append(
                        visual.Line(
                            win=self.myWin.win, start=(0, -20), end=(0, 20),
                            pos=(x, y), ori=0.0, contrast=-1.0, color=self.myWin.stimulus_colour, lineWidth=3
                        )
                    )
            correct_streak += show_trial(all_stims, visible=False)

        show_text(
            "Sometimes the target will be barely visible, or absent.\n"
            "Press RIGHT if you see it, LEFT if you do not.\n"
            "You won't receive feedback during the real experiment.\n\n"
            "To continue, press the right arrow."
        )

        show_text("You have finished the tutorial!\nGood luck with the experiment!\n"
                  "Press RIGHT to continue to the actual experiment.")
        core.wait(3)
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])
        
    def run_main(self, dataFile):

        breaks, totalTrials = self.getBreaks()
        stairs = self.stairs
        totalStaircaseTrials = int(len(self.myConds) * stairs.nTrials)

        middle_index = len(breaks) // 2
        middle_trial = breaks[middle_index] if len(breaks) > 0 else -1

        print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}, middle index: {middle_index}, middle trial: {middle_trial}")

        thisTrial = 0
        stairTrialCount = 0

        bg = self.myWin.background_val

        # ==========================================
        # MAIN LOOP
        # ==========================================
        while stairTrialCount < totalStaircaseTrials:

            self.myWin.checkQuit()

            print('===============')
            print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}")
            print(f"Trial: {thisTrial}, staircase count: {stairTrialCount}")

            # ==========================================
            # NULL TRIAL?
            # ==========================================
            isNull = np.random.random() <= self.nullOdds

            if isNull:

                print("Null trial")

                currentStair = stairs.currentStaircase
                condition = currentStair.condition

                thisLabel = condition['label'] + '_null'

                targetIntensity = None

            else:

                stairs.next()

                currentStair = stairs.currentStaircase
                condition = currentStair.condition

                thisLabel = condition['label']

                targetIntensity = float(currentStair.intensity)

            # ==========================================
            # BREAKS
            # ==========================================
            if thisTrial in breaks:

                b_idx = np.where(breaks == thisTrial)[0][0]

                middle_break = (thisTrial == middle_trial)

                self.doBreak(b=b_idx, middle=middle_break)

                os.makedirs(self.path, exist_ok=True)

                psydat_path = os.path.join(
                    self.path,
                    f"{self.id}_main.psydat"
                )

                stairs.saveAsPickle(
                    psydat_path,
                    fileCollisionMethod='overwrite'
                )

            # ==========================================
            # START EYELINK RECORDING
            # ==========================================
            if self.eyeTracker.doTracking:

                self.eyeTracker.tracker.startRecording(
                    1, 1, 1, 1
                )

            # ==========================================
            # PREPARE STIMULUS
            # ==========================================
            flankerIntensity = round(
                float(currentStair.condition['FC']),
                8
            )

            stim_key = condition['stim_key']

            stimulus = self.myWin.stimuli[stim_key]

            targetContrast = utils.abs_contrast_from_bg(
                targetIntensity,
                bg
            )

            flankerContrast = utils.abs_contrast_from_bg(
                flankerIntensity,
                bg
            )

            lines = []

            for entry in stimulus['components']:

                is_target = entry.get('type') == 'target'
                line = entry['line_obj']

                line_copy = line
                line_copy = line  # (or deepcopy if needed)

                if isNull and is_target:

                    if bg >= 0:
                        line_copy.contrast = -bg
                    else:
                        line_copy.contrast = bg

                else:

                    if is_target:
                        line_copy.contrast = targetIntensity
                    else:
                        line_copy.contrast = flankerIntensity

                lines.append(line_copy)

            print(
                f'Label: {thisLabel}, '
                f'target intensity (TC): {targetIntensity}, '
                f'flanker intensity (FC): {flankerIntensity}, '
                f'target contrast: {targetContrast}, '
                f'flanker contrast: {flankerContrast}'
            )

            # ==========================================
            # CLEAR KEYBOARD BUFFER
            # ==========================================
            event.clearEvents()

            # ==========================================
            # FIXATION PERIOD
            # ==========================================
            fixationClock = core.Clock()
            # diode OFF during fixation
            self.diodeOff()
            while fixationClock.getTime() < self.myWin.t_fixation:

                self.myWin.drawOrder(self.myWin.fixation)

                self.myWin.checkQuit()

            # ==========================================
            # STIMULUS PERIOD
            # ==========================================
            stimClock = core.Clock()
            self.diodeOn()
            # send EyeLink message exactly on flip
            if self.eyeTracker.doTracking:

                self.myWin.win.callOnFlip(
                    self.eyeTracker.stimOnset,
                    thisTrial,
                    thisLabel,
                    targetIntensity
                )

            while stimClock.getTime() < self.myWin.t_stim:

                # draw stimulus
                self.myWin.drawOrder(lines)

                self.myWin.checkQuit()

            # ==========================================
            # RESPONSE PERIOD
            # ==========================================
            responseClock = core.Clock()

            thisResp = 0
            thisRT = 99

            responded = False

            valid_keys = [
                'left',
                'num_4',
                'right',
                'num_6'
            ]

            quit_keys = ['q', 'escape']

            self.diodeOff()
            while responseClock.getTime() < self.myWin.t_response:

                # blank screen
                self.myWin.drawOrder(self.myWin.blank)

                keys = event.getKeys(
                    keyList=valid_keys + quit_keys,
                    timeStamped=responseClock
                )

                if keys and not responded:

                    key, t = keys[0]

                    if key in quit_keys:

                        self.eyeTracker.closeTracker()
                        core.quit()

                    if key in ['left', 'num_4']:

                        thisResp = 0

                    elif key in ['right', 'num_6']:

                        thisResp = 1

                    thisRT = round(t, 5)

                    responded = True

                self.myWin.checkQuit()

            # ==========================================
            # FEEDBACK
            # ==========================================
            if thisRT == 99:

                fb_stim = self.myWin.feedback_nan

            elif thisResp == 1:

                fb_stim = self.myWin.feedback_yes

            else:

                fb_stim = self.myWin.feedback_no

            fbClock = core.Clock()

            while fbClock.getTime() < (2 / 60):

                self.myWin.drawOrder(fb_stim)

            # ==========================================
            # LOG RESPONSE
            # ==========================================
            self.eyeTracker.logResponse(
                thisResp,
                thisRT
            )

            self.dataFile.write(
                f"{self.id},"
                f"{thisTrial},"
                f"{thisLabel},"
                f"{condition['FC']},"
                f"{currentStair.intensity},"
                f"{flankerContrast},"
                f"{targetContrast},"
                f"{thisResp},"
                f"{thisRT}\n"
            )

            self.dataFile.flush()

            # ==========================================
            # UPDATE STAIRCASE
            # ==========================================
            if not isNull:

                stairs.addResponse(thisResp)

                stairTrialCount += 1

            # increment displayed trial count
            thisTrial += 1

            # ==========================================
            # STOP EYELINK RECORDING
            # ==========================================
            if self.eyeTracker.doTracking:

                self.eyeTracker.tracker.stopRecording()

        # ==========================================
        # END EXPERIMENT
        # ==========================================
        self.myWin.checkQuit()

        self.myWin.end()

        self.eyeTracker.closeTracker()

        os.makedirs(self.path, exist_ok=True)

        psydat_path = os.path.join(
            self.path,
            f"{self.id}_main.psydat"
        )

        stairs.saveAsPickle(
            psydat_path,
            fileCollisionMethod='overwrite'
        )
     #old version, problem with the diode being stalled by core.wait() better for response collection to rewrite

    # def run_main(self, dataFile):
    #     breaks, totalTrials = self.getBreaks() #total trials with null trials for correct breaks
    #     stairs = self.stairs
    #     totalStaircaseTrials = int(len(self.myConds) * stairs.nTrials) # staircase trials
    
    #     middle_index = len(breaks) // 2
    #     middle_trial = breaks[middle_index] if len(breaks) > 0 else -1

    #     print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}, middle index: {middle_index}, middle trial: {middle_trial}")

    #     trialClock = core.Clock()
    #     thisTrial = 0         # counts all displayed trials (including nulls)
    #     stairTrialCount = 0   # counts only trials added to staircase

    #     bg = self.myWin.background_val
        
    #     # Loop until all staircase trials are completed
    #     while stairTrialCount < totalStaircaseTrials: 
    #         self.myWin.checkQuit()
    #         print('===============')
    #         #print(f"Total trials with null: {stairs.totalTrials},\n total staircase trials (no null): {totalStaircaseTrials}, total for breaks: {totalTrials}")
            
    #         print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}, middle index: {middle_index}, middle trial: {middle_trial}")
    #         #print(f"Total trials from the staircase: {stairs.totalTrials}")
    #         print(f"Total trials no null, totalStaircaseTrials: {totalStaircaseTrials}")
    #         print(f"Trial: {thisTrial}, staircase count: {stairTrialCount}")
            
    #         # --- Random null trial ---
    #         isNull = np.random.random() <= self.nullOdds
            
    #         if isNull:
    #             print("Null trial")
    #             currentStair = stairs.currentStaircase
    #             condition = currentStair.condition
    #             thisLabel = condition['label']
    #             thisLabel += '_null'
    #             targetIntensity = None # background (invisible)
    #         else:
    #             stairs.next()  
    #             currentStair = stairs.currentStaircase
    #             condition = currentStair.condition
    #             thisLabel = condition['label']
    #             targetIntensity = float(currentStair.intensity)
            
    #         # --- Handle breaks ---
    #         if thisTrial in breaks:
    #             b_idx = np.where(breaks == thisTrial)[0][0]
    #             middle_break = (thisTrial == middle_trial)
    #             self.doBreak(b=b_idx, middle=middle_break)
    #             # --- Save staircase periodically ---
    #             os.makedirs(self.path, exist_ok=True)
    #             psydat_path = os.path.join(self.path, f"{self.id}_main.psydat")
    #             stairs.saveAsPickle(psydat_path, fileCollisionMethod='overwrite')

    #         # --- Eye tracker start ---
    #         if self.eyeTracker.doTracking:
    #             self.eyeTracker.tracker.startRecording(1, 1, 1, 1)

    #         # --- Prepare stimulus ---
    #         lines = []
    #         flankerIntensity = round(float(currentStair.condition['FC']), 8)
    #         stim_key = condition['stim_key']
    #         stimulus = self.myWin.stimuli[stim_key]
            
    #         targetContrast = utils.abs_contrast_from_bg(targetIntensity, bg)
    #         flankerContrast = utils.abs_contrast_from_bg(flankerIntensity, bg)
            
    #         lines = []
    #         for entry in stimulus['components']:
    #             # Determine if this is the target or a flanker
    #             is_target = entry.get('type') == 'target'

    #             # --- Null trial: target should be invisible ---
    #             if isNull and is_target:
    #                 if bg >= 0:
    #                    entry['line_obj'].contrast = -bg 
    #                 else:
    #                     entry['line_obj'].contrast = bg  # exact background
    #             else:
    #                 # Normal trial or flanker: assign proper contrast
    #                 if is_target:
    #                     entry['line_obj'].contrast = targetIntensity
    #                 else:
    #                     entry['line_obj'].contrast = flankerIntensity

    #             lines.append(entry['line_obj'])

    #         print(f'Label: {thisLabel}, target intensity (TC): {targetIntensity}, flanker intensity (FC): {flankerIntensity}, target contrast: {targetContrast}, flanker contrast: {flankerContrast}')
    #         # --- Draw fixation ---
    #         #self.myWin.diode.color *= -1
    #         self.myWin.drawOrder(self.myWin.fixation)
    #         core.wait(self.myWin.t_fixation)
    #         self.blinkDiode()

    #         # --- Draw stimulus ---
    #         self.eyeTracker.stimOnset(thisTrial, thisLabel, targetIntensity)
    #         #self.myWin.diode.color *= -1
    #         diodeOn()
    #         self.myWin.diode.draw()

    #         self.myWin.drawOrder(lines)
    #         core.wait(self.myWin.t_stim)
    #         #self.blinkDiode()
    #         diodeOff()
    #         self.myWin.diode.draw()

    #         # --- Collect response ---
    #         trialClock.reset()
    #         self.myWin.drawOrder(self.myWin.blank)
    #         allKeys = event.waitKeys(maxWait=self.myWin.t_response,
    #                                 keyList=['left','num_4','right','num_6','q','escape'])
    #         thisRT = trialClock.getTime()
    #         if thisRT < self.myWin.t_response:
    #             core.wait(self.myWin.t_response - thisRT)

    #         if allKeys:
    #             for key in allKeys:
    #                 if key in ['left','num_4']:
    #                     thisResp = 0
    #                 elif key in ['right','num_6']:
    #                     thisResp = 1
    #                 else:
    #                     self.eyeTracker.closeTracker()
    #                     core.quit()
    #         else:
    #             thisResp = 0
    #             thisRT = 99

    #         if thisRT == 99:
    #             fb_stim = self.myWin.feedback_nan      
    #         elif thisResp == 1:
    #             fb_stim = self.myWin.feedback_yes     
    #         else:
    #             fb_stim = self.myWin.feedback_no       

    #         self.myWin.drawOrder(fb_stim)
    #         core.wait(2/60)
            
    #         # --- Log response ---
    #         self.eyeTracker.logResponse(thisResp, thisRT)
    #         self.dataFile.write(f"{self.id},{thisTrial},{thisLabel},{condition['FC']},{currentStair.intensity},{flankerContrast},{targetContrast},{thisResp},{thisRT}\n")
    #         self.dataFile.flush()
            
    #         # --- Add response only if not null ---
    #         if not isNull:
    #             stairs.addResponse(thisResp)
    #             stairTrialCount += 1

    #         # Increment total trial counter for breaks / logging
    #         thisTrial += 1

    #         if self.eyeTracker.doTracking:
    #             self.eyeTracker.tracker.stopRecording()

    #     # --- End of experiment ---
    #     self.myWin.checkQuit()
    #     self.myWin.end()
    #     self.eyeTracker.closeTracker()
        
    #     os.makedirs(self.path, exist_ok=True)
    #     psydat_path = os.path.join(self.path, f"{self.id}_main.psydat")
    #     stairs.saveAsPickle(psydat_path, fileCollisionMethod='overwrite')
    
    def getThresholdFromBase(self, file_path):
        """
        Fit psychometric function on absolute contrast relative to background.
        
        Parameters
        ----------
        file_path : str
            CSV with columns 'TN' (stimulus contrast normalised), 'TC' target contrast and 'response' (0/1)
        bg : float
            Background intensity (-1 or 1)
        
        Returns
        -------
        thresholds : dict
            Thresholds for requested probabilities
        """
        bg= self.myWin.background_val
        # Load and filter data
        thisDat = pd.read_csv(file_path)
        thisDat = thisDat[~thisDat['label'].str.endswith('_null')]

        # Convert raw intensities to absolute contrast
        allIntensities_norm = thisDat['TN'] 
        allIntensities_norm_transformed = utils.stim_from_abs_contrast(allIntensities_norm, bg)
        
        allIntensities = thisDat['TC']
        threshold_val = allIntensities[-25:].median()
        #threshold_val_norm = allIntensities_norm.median()
        threshold_val_norm_transformed = allIntensities_norm_transformed.median()
        
        print(f'--------All intensities (raw): {allIntensities.tolist()}')
        allResponses = thisDat['response'].tolist()

        # Collapse repetitions
        i, r, n = data.functionFromStaircase(allIntensities, allResponses, bins='unique')
        combinedN = pylab.array(n)

        # Fit logistic
        fit = data.FitLogistic(
            i, r,
            expectedMin=0.5,      # for 2AFV, 0.5 can also be used if you prefer
            sems=1.0 / combinedN,
            optimize_kws={'maxfev': int(1e6)}
        )
        return threshold_val_norm_transformed, threshold_val
        
    def reDoBase(self,thresh):
        m_redo = visual.TextStim(self.myWin.win, color=self.myWin.stimulus_colour, height = 32, wrapWidth=600,
                                 text = f"Please wait for the experimenter.\nParticipant {self.id} baseline detection threshold:\n{thresh}\n\nTry again [y / n]?")
        
        self.myWin.drawOrder(m_redo)
        keys = event.waitKeys(keyList=['y','n'])
        if keys:
            for key in keys:
                if key == 'y':
                    return True
                else:
                    return False


    # def blinkDiode(self,t=2/60):
    #     # Defaults to two frames blink (at 60fps)
    #     # Blinks the diode to indicate the offset of a stimulus
    #     # Does not draw any new stimuli, flips the window with existing stuff
    #     self.myWin.diode.color *= -1
    #     self.myWin.win.flip()
    #     core.wait(t) # 2 frames

    def diodeOn(self):
        self.myWin.diode.color = [-1, -1, -1]

    def diodeOff(self):
        self.myWin.diode.color = [1, 1, 1]
    

        